"""Precompiled field projections.

This module provides :func:`compile_projection`, which turns a fixed
mapping of output field names to explicit :class:`~glom.Path` /
:data:`~glom.T` accessors into a reusable, callable
:class:`CompiledProjection`.

Projections are designed for the "same shape, many records" case:
many output fields reading from the same deep object, over and over.
Within a single record, reads through types registered as pure-read
(exact ``dict``/``list``/tuple, plus any user-declared *pure_types*)
are performed once and shared between fields. Nothing is cached
across records, and reads through undeclared custom types keep the
exact same order and cardinality as a plain glom dict spec, so
observable behavior never changes for the sake of speed.
"""

from .core import (
    glom,
    Path,
    TType,
    T,
    PathAccessError,
    BadSpec,
    TargetRegistry,
    _DEFAULT_SCOPE,
    _MISSING,
    bbrepr,
)

__all__ = ['Field', 'CompiledProjection', 'compile_projection']


# reads on exact instances of these builtin containers have no
# user-defined side effects, so they are always safe to share.
_BASE_PURE_TYPES = (dict, list, tuple)

# T operations supported by projections: attribute access ('.'),
# subscript access ('['), and Path-style registered get ('P').
_ALLOWED_OPS = ('.', '[', 'P')


class Field:
    """Declares a single output field for :func:`compile_projection`.

    Args:
       path (Path|T): An explicit :class:`~glom.Path` or
          :data:`~glom.T` accessor, using only attribute and
          subscript access.
       default: Value to use when *path* is missing from a record
          (i.e. a :exc:`~glom.PathAccessError` occurs anywhere along
          the path). When omitted, missing paths raise as usual.
          Note that a present value of ``None`` is a real value and
          never triggers the default.

    >>> Field(T['a']['b'], default='n/a')
    Field(T['a']['b'], default='n/a')
    """
    def __init__(self, path, default=_MISSING):
        self.path = path
        self.default = default

    def __repr__(self):
        cn = self.__class__.__name__
        if self.default is _MISSING:
            return f'{cn}({self.path!r})'
        return f'{cn}({self.path!r}, default={bbrepr(self.default)})'


class _CompiledField:
    __slots__ = ('name', 't_path', 'ops', 'default')

    def __init__(self, name, t_path, ops, default):
        self.name = name
        self.t_path = t_path
        self.ops = ops
        self.default = default


def _compile_path(spec, field_name):
    """Validate and flatten a Path/T accessor into (op, arg) steps."""
    if isinstance(spec, Path):
        t_path = spec.path_t
    elif isinstance(spec, TType):
        t_path = spec
    else:
        raise BadSpec(
            'projection field %r must be an explicit Path or T accessor,'
            ' not: %r' % (field_name, spec))
    t_ops = t_path.__ops__
    if t_ops[0] is not T:
        raise BadSpec(
            'projection field %r must be rooted at T, not: %r'
            % (field_name, spec))
    steps = []
    for i in range(1, len(t_ops), 2):
        op, arg = t_ops[i], t_ops[i + 1]
        if op not in _ALLOWED_OPS:
            raise BadSpec(
                'projection field %r only supports attribute and subscript'
                ' access, got unsupported operation %r in: %r'
                % (field_name, op, spec))
        if isinstance(arg, (TType, Path)) or callable(arg):
            raise BadSpec(
                'projection field %r does not support dynamic path'
                ' segments, got: %r' % (field_name, arg))
        steps.append((op, arg))
    return t_path, tuple(steps)


class CompiledProjection:
    """A precompiled, reusable field projection.

    Instances are callables taking a single record and returning a
    dict of output fields, in declaration order. They are also valid
    glom specs, so they compose with the rest of glom:

    >>> proj = compile_projection({
    ...     'name': T['user']['name'],
    ...     'email': Field(T['user']['email'], default=''),
    ... })
    >>> proj({'user': {'name': 'Ada'}})
    {'name': 'Ada', 'email': ''}
    >>> glom([{'user': {'name': 'Ada', 'email': 'a@b.c'}}], [proj])
    [{'name': 'Ada', 'email': 'a@b.c'}]

    Within one call, reads through pure-read types (exact
    ``dict``/``list``/``tuple`` and any *pure_types* declared at
    compile time) are shared between fields with common path
    prefixes. Reads through any other type are repeated per field,
    exactly as a plain glom dict spec would.
    """
    def __init__(self, fields, pure_types=()):
        if isinstance(fields, dict):
            field_items = list(fields.items())
        elif isinstance(fields, (list, tuple)):
            field_items = list(fields)
        else:
            raise BadSpec(
                'expected projection fields as a dict or list of'
                ' (name, spec) pairs, not: %r' % (fields,))
        pure_types = tuple(pure_types)
        for pure_type in pure_types:
            if not isinstance(pure_type, type):
                raise TypeError(
                    'expected pure_types entries to be types, not: %r'
                    % (pure_type,))
        self._pure_types = pure_types
        compiled = []
        for name, spec in field_items:
            default = _MISSING
            if isinstance(spec, Field):
                spec, default = spec.path, spec.default
            t_path, ops = _compile_path(spec, name)
            compiled.append(_CompiledField(name, t_path, ops, default))
        self._fields = tuple(compiled)

    @property
    def field_names(self):
        """Output field names, in declaration order."""
        return tuple(field.name for field in self._fields)

    def _is_pure(self, obj):
        return (type(obj) in _BASE_PURE_TYPES
                or isinstance(obj, self._pure_types))

    def _eval_field(self, target, field, memo, scope):
        cur = target
        for seg_idx, (op, arg) in enumerate(field.ops):
            mkey = None
            if self._is_pure(cur):
                try:
                    mkey = (id(cur), op, arg)
                except TypeError:
                    mkey = None  # unhashable segment, skip sharing
                if mkey is not None and mkey in memo:
                    cur = memo[mkey][1]
                    continue
            pae = None
            if op == '.':
                try:
                    nxt = getattr(cur, arg)
                except AttributeError as e:
                    pae = PathAccessError(e, Path(field.t_path), seg_idx)
            elif op == '[':
                try:
                    nxt = cur[arg]
                except (KeyError, IndexError, TypeError) as e:
                    pae = PathAccessError(e, Path(field.t_path), seg_idx)
            else:  # 'P'
                get = scope[TargetRegistry].get_handler('get', cur)
                try:
                    nxt = get(cur, arg)
                except Exception as e:
                    pae = PathAccessError(e, Path(field.t_path), seg_idx)
            if pae is not None:
                raise pae
            if mkey is not None:
                # keep cur alive in the memo so its id cannot be
                # recycled by a different object during this record
                memo[mkey] = (cur, nxt)
            cur = nxt
        return cur

    def glomit(self, target, scope):
        memo = {}
        ret = {}
        for field in self._fields:
            try:
                ret[field.name] = self._eval_field(target, field, memo, scope)
            except PathAccessError:
                if field.default is _MISSING:
                    raise
                ret[field.name] = field.default
        return ret

    def __call__(self, target):
        return self.glomit(target, _DEFAULT_SCOPE)

    def __repr__(self):
        cn = self.__class__.__name__
        specs = ', '.join(
            f'{field.name}: {_format_t_path(field.t_path)}'
            for field in self._fields)
        return f'{cn}({{{specs}}})'


def _format_t_path(t_path):
    t_ops = t_path.__ops__
    ret = 'T'
    for i in range(1, len(t_ops), 2):
        op, arg = t_ops[i], t_ops[i + 1]
        if op == '.':
            ret += f'.{arg}'
        else:
            ret += f'[{arg!r}]'
    return ret


def compile_projection(fields, pure_types=()):
    """Compile a fixed field projection for repeated use.

    Args:
       fields (dict): Mapping of output field names to explicit
          :class:`~glom.Path` / :data:`~glom.T` accessors. Use
          :class:`Field` to give an output field its own missing-value
          default. A list of ``(name, spec)`` pairs is also accepted.
       pure_types (tuple): Custom types whose attribute/item reads
          are declared side-effect-free and idempotent. Reads through
          these types are shared between fields within a single
          record. Types not declared here are always read in plain
          glom order, once per field.

    Only explicit attribute and subscript access is supported.
    Function calls, wildcards, arithmetic, and dynamic specs are
    rejected with :exc:`~glom.BadSpec` at compile time.

    >>> proj = compile_projection({'a': T['x']['a'], 'b': T['x']['b']})
    >>> proj({'x': {'a': 1, 'b': 2}})
    {'a': 1, 'b': 2}
    """
    return CompiledProjection(fields, pure_types=pure_types)
