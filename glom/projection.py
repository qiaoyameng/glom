from collections import OrderedDict

from glom.core import (BadSpec,
                       Coalesce,
                       GlomError,
                       Path,
                       PathAccessError,
                       SKIP,
                       T,
                       TargetRegistry,
                       TType,
                       _MISSING,
                       glom)


class _ProjectNode:
    __slots__ = ('children', 'missing_error', 'value')

    def __init__(self):
        self.children = {}
        self.missing_error = None
        self.value = None


class _CompiledProjectField:
    __slots__ = ('name', 'path', 'ops', 'default', 'has_default')

    def __init__(self, name, path, default, has_default):
        self.name = name
        self.path = path
        self.ops = path.items()
        self.default = default
        self.has_default = has_default

    def glomit(self, target, scope):
        registry = scope[TargetRegistry]
        root = scope[_ProjectExecution]
        node = root
        cur = target

        for part_idx, (op, arg) in enumerate(self.ops):
            key = (op, arg)
            child_node = node.children.get(key)
            if child_node is not None:
                if child_node.missing_error is not None:
                    if self.has_default:
                        return self.default
                    exc = child_node.missing_error
                    raise PathAccessError(exc, self.path, part_idx)
                node = child_node
                cur = node.value
                continue

            if op == '.':
                pure = registry.is_pure(cur, 'attr')
                try:
                    nxt = getattr(cur, arg)
                except AttributeError as exc:
                    return self._access_error(exc, node, key, pure, part_idx)
            elif op == '[':
                pure = registry.is_pure(cur, 'item')
                try:
                    nxt = cur[arg]
                except (KeyError, IndexError, TypeError) as exc:
                    return self._access_error(exc, node, key, pure, part_idx)
            else:
                pure = registry.is_pure(cur, 'get')
                get = registry.get_handler('get', cur, path=self.path.values()[:part_idx + 1])
                try:
                    nxt = get(cur, arg)
                except Exception as exc:
                    return self._access_error(exc, node, key, pure, part_idx)

            child_node = _ProjectNode()
            child_node.value = nxt
            if pure:
                node.children[key] = child_node
                node = child_node
            else:
                node = _ProjectNode()
            cur = nxt

        return cur

    def _access_error(self, exc, node, key, pure, part_idx):
        child_node = _ProjectNode()
        child_node.missing_error = exc
        if pure:
            node.children[key] = child_node
        if self.has_default:
            return self.default
        raise PathAccessError(exc, self.path, part_idx)

    def __repr__(self):
        return repr(self.path)


class _ProjectExecution:
    pass


def _is_static_literal(value, allow_callable=False):
    if type(value) in (TType, Path) or hasattr(value, 'glomit'):
        return False
    if not allow_callable and callable(value):
        return False
    if isinstance(value, slice):
        return (_is_static_literal(value.start, allow_callable)
                and _is_static_literal(value.stop, allow_callable)
                and _is_static_literal(value.step, allow_callable))
    if isinstance(value, tuple):
        return all(_is_static_literal(item, allow_callable) for item in value)
    if isinstance(value, (list, dict, set)):
        return False
    return True


def _is_static_key(value):
    if type(value) in (TType, Path) or hasattr(value, 'glomit') or callable(value):
        return False
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _compile_path(spec):
    if type(spec) is Path:
        path = spec
    elif type(spec) is TType:
        path = Path(spec)
    else:
        raise BadSpec('Project fields must be Path or T access paths, not: %r' % (spec,))

    ops = path.path_t.__ops__
    if ops[0] is not T:
        raise BadSpec('Project paths must start from T, not: %r' % (path,))

    for op, arg in path.items():
        if op == '.':
            if type(arg) is not str:
                raise BadSpec('Project attribute access must use a static name: %r' % (path,))
        elif op in ('[', 'P'):
            if not _is_static_literal(arg, allow_callable=True):
                raise BadSpec('Project subscripts must be static values,'
                              ' not dynamic specs: %r' % (path,))
        else:
            raise BadSpec('Project paths may only contain attribute, item,'
                          ' or Path access: %r' % (path,))
    return path


def _compile_field(name, field_spec, defaults):
    if not _is_static_key(name):
        raise BadSpec('Project output names must be static keys, not: %r' % (name,))

    default = defaults.get(name, _MISSING) if defaults is not None else _MISSING
    if default is not _MISSING and not _is_static_literal(default):
        raise BadSpec('Project default for %r must be a static value' % (name,))

    if type(field_spec) is Coalesce:
        if len(field_spec.subspecs) != 1:
            raise BadSpec('Project defaults support exactly one path per'
                          ' field: %r' % (field_spec,))
        if field_spec.skip is not _MISSING or field_spec.skip_exc is not GlomError:
            raise BadSpec('Project fields do not support Coalesce skip or'
                          ' skip_exc: %r' % (field_spec,))
        if field_spec.default_factory is not _MISSING:
            raise BadSpec('Project fields do not support default_factory: %r' % (field_spec,))
        if default is not _MISSING and field_spec.default is not _MISSING:
            raise BadSpec('field %r has more than one default' % (name,))
        if field_spec.default is not _MISSING:
            default = field_spec.default
            if not _is_static_literal(default):
                raise BadSpec('Project default for %r must be a static value' % (name,))
        path = _compile_path(field_spec.subspecs[0])
    elif isinstance(field_spec, tuple):
        if len(field_spec) != 2:
            raise BadSpec('Project tuple fields must be (path, default), got: %r' % (field_spec,))
        path_spec, field_default = field_spec
        if default is not _MISSING:
            raise BadSpec('field %r has more than one default' % (name,))
        default = field_default
        if not _is_static_literal(default):
            raise BadSpec('Project default for %r must be a static value' % (name,))
        path = _compile_path(path_spec)
    else:
        path = _compile_path(field_spec)

    if default is not _MISSING and callable(default):
        raise BadSpec('Project default for %r must be a static value' % (name,))
    return _CompiledProjectField(name, path, default, default is not _MISSING)


class Project:
    """Compile a fixed mapping of output fields to explicit ``Path``/``T`` reads.

    >>> target = {'id': 1, 'address': {'city': 'Paris'}}
    >>> project = Project({'id': T['id'], 'city': T['address']['city']},
    ...                   defaults={'city': None})
    >>> project.glom(target)
    {'id': 1, 'city': 'Paris'}

    Per-field defaults can also be written as ``(path, default)`` or with a
    one-path :class:`Coalesce`. Calls, wildcard traversal, dynamic specs,
    default factories, and nested output structures raise :class:`BadSpec`
    when the project is constructed.

    Values read through a type registered with ``pure=True`` are shared in a
    prefix trie for one target only. Calling the project on another target
    creates a fresh trie, so target values are never cached across records.
    """
    def __init__(self, fields, defaults=None):
        if not isinstance(fields, dict):
            raise BadSpec('Project expects a mapping of output fields to paths')
        if defaults is not None and not isinstance(defaults, dict):
            raise BadSpec('Project defaults must be a mapping of output field names')
        self.result_type = type(fields)
        self.fields = tuple(_compile_field(name, spec, defaults)
                            for name, spec in fields.items())

    def glomit(self, target, scope):
        result = self.result_type()
        scope[_ProjectExecution] = _ProjectNode()
        for field in self.fields:
            value = scope[glom](target, field, scope)
            if value is SKIP:
                continue
            result[field.name] = value
        return result

    def glom(self, target, **kwargs):
        return glom(target, self, **kwargs)

    def many(self, targets):
        return [self.glom(target) for target in targets]

    def __call__(self, target):
        return self.glom(target)

    def __repr__(self):
        fields = OrderedDict((field.name, field.path) for field in self.fields)
        return 'Project(%r)' % (fields)


CompiledProjection = Project
Projection = Project
compile_project = Project
