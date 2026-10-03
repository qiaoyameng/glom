"""Explicit, round-trippable projections.

Glom's natural direction is target -> view: a glomspec *projects* an
internal data structure (the *target*) into some output shape.  When
that output is an editable form, applications traditionally write a
second, hand-maintained mapping to project edits back onto the target.
The two mappings inevitably drift.

:class:`Field` and :class:`Project` make both directions explicit in a
single, composable spec.  Each form field declares:

* its *source* path in the target,
* an optional forward transform (``get``), used when reading, and
* an optional reverse transform (``put``), used when writing back.

Reading is done with :meth:`Project.load` (or by using a
:class:`Project` directly as a glomspec); writing is done with
:meth:`Project.dump`.

Reverse transforms are never inferred: a one-way ``get`` function, or
a ``default`` that fabricates a value for a missing source, is not
silently assumed to be reversible.  When several form fields write to
the same source, their reverse transforms must agree on the resulting
value, or the whole write fails atomically, leaving the target
untouched.
"""

import copy

from boltons.typeutils import make_sentinel

from .core import Path, TType, glom, GlomError
from .mutation import Assign


_NO_DEFAULT = make_sentinel('_NO_DEFAULT')


class ProjectionError(GlomError):
    """Raised when a writable projection cannot be safely performed.

    This covers non-invertible fields that are included in a
    write-back, as well as conflicting reverse values produced by
    multiple form fields that share the same source path.
    """
    def __init__(self, msg, path=None):
        super(ProjectionError, self).__init__(msg)
        self.path = path

    def get_message(self):
        return self.args[0]


class Field(object):
    """A single editable form field backed by a target source.

    Args:
        source: Path to the value in the target. A ``'a.b'``-style
            string, a :class:`~glom.Path`, or a :data:`~glom.T`
            expression.
        get: Optional one-argument forward transform applied while
            reading the source into the form.
        put: Optional one-argument reverse transform applied while
            writing a form value back to the source. Required for a
            field carrying a ``get`` transform (or a ``default``) to
            participate in a write-back; it is never inferred.
        default: Value used for the form when the source path is
            missing. Because this fabricates information, a field with
            a default but no explicit ``put`` cannot be written back.

    >>> field = Field('config.port', get=str, put=int)
    >>> target = {'config': {'port': 8080}}
    >>> field.read(target)
    '8080'
    >>> field.write('9090')[1]
    9090
    """

    def __init__(self, source, get=None, put=None, default=_NO_DEFAULT):
        self.source = _coerce_path(source)
        self.get = get
        self.put = put
        self.default = default

    def read(self, target, source=None):
        source = self.source if source is None else source
        try:
            value = glom(target, source)
        except GlomError:
            if self.default is _NO_DEFAULT:
                raise
            value = self.default
        if self.get is not None:
            value = self.get(value)
        return value

    def write(self, value, source=None):
        """Return a ``(source, reverse_value)`` pair for this field.

        *source* overrides the field's own path, used when a parent
        :class:`Project` contributes a base path.
        """
        source = self.source if source is None else source
        if self.put is not None:
            value = self.put(value)
        elif self.get is not None:
            raise ProjectionError(
                'field at %r has forward transform %r but no reverse '
                'transform; one-way transforms are not assumed invertible'
                % (source, _func_repr(self.get)))
        elif self.default is not _NO_DEFAULT:
            raise ProjectionError(
                'field at %r has default %r but no reverse transform; '
                'lossy defaults are not assumed invertible'
                % (source, self.default))
        return source, value

    def __repr__(self):
        cn = self.__class__.__name__
        parts = [repr(self.source)]
        if self.get is not None:
            parts.append('get=%s' % _func_repr(self.get))
        if self.put is not None:
            parts.append('put=%s' % _func_repr(self.put))
        if self.default is not _NO_DEFAULT:
            parts.append('default=%r' % (self.default,))
        return '%s(%s)' % (cn, ', '.join(parts))


class _Leaf(object):
    __slots__ = ('source', 'form_path', 'field')

    def __init__(self, source, form_path, field):
        self.source = source
        self.form_path = form_path
        self.field = field


class Project(object):
    """An explicit, nested, round-trippable projection spec.

    *fields* maps form keys to :class:`Field` instances, raw source
    paths (strings, :class:`~glom.Path`, or :data:`~glom.T`), nested
    dicts, or further :class:`Project` instances. A nested
    :class:`Project` may carry its own *base*, which prefixes all of
    its source paths, so specs compose.

    >>> target = {'svc': {'host': 'example', 'port': 8080}}
    >>> spec = Project({'host': Field('host'),
    ...                 'port': Field('port', get=str, put=int)},
    ...                base='svc')
    >>> form = spec.load(target)
    >>> form == {'host': 'example', 'port': '8080'}
    True
    >>> new_target = spec.dump({'host': 'example', 'port': '9090'}, target)
    >>> new_target['svc']['port']
    9090
    """

    def __init__(self, fields, base=None):
        self.base = Path() if base is None else _coerce_path(base)
        self.fields = fields
        self._leaves = list(_flatten(fields, self.base, ()))

    def glomit(self, target, scope):
        return self.load(target)

    def load(self, target):
        """Read *target* into the form view.

        A missing source without a field ``default`` propagates the
        underlying :exc:`~glom.GlomError`.
        """
        result = {}
        for leaf in self._leaves:
            sub = result
            for key in leaf.form_path[:-1]:
                sub = sub.setdefault(key, {})
            sub[leaf.form_path[-1]] = leaf.field.read(target, leaf.source)
        return result

    def dump(self, form, target=None, in_place=False):
        """Write *form* back onto *target* and return the result target.

        Only source paths covered by the projection are touched; every
        other part of *target* is preserved verbatim. The whole write
        is validated before anything is mutated, so a non-invertible
        field or a disagreement between form fields sharing a source
        fails atomically.

        By default a deep copy is returned and *target* is left
        untouched. Pass ``in_place=True`` to mutate *target*.
        """
        assignments = _plan_write(self._leaves, form)

        if in_place:
            result = {} if target is None else target
        else:
            result = {} if target is None else copy.deepcopy(target)

        for source, value in assignments:
            glom(result, Assign(source, value, missing=dict))
        return result

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.fields)


def _flatten(fields, base, form_path):
    if isinstance(fields, Project):
        sub_base = _join_paths(base, fields.base)
        for leaf in _flatten(fields.fields, sub_base, form_path):
            yield leaf
        return
    if not isinstance(fields, dict):
        raise TypeError('Project fields must be a dict or Project, not %r'
                        % (type(fields).__name__,))
    for key, val in fields.items():
        if not isinstance(key, str):
            raise TypeError('form keys must be strings, not %r' % (key,))
        cur_path = form_path + (key,)
        if isinstance(val, Field):
            yield _Leaf(_join_paths(base, val.source), cur_path, val)
        elif isinstance(val, Project) or isinstance(val, dict):
            for leaf in _flatten(val, base, cur_path):
                yield leaf
        else:
            field = Field(_coerce_path(val))
            yield _Leaf(_join_paths(base, field.source), cur_path, field)


def _plan_write(leaves, form):
    planned = {}  # path key -> (source Path, value, first form path)
    for leaf in leaves:
        sub = form
        try:
            for key in leaf.form_path:
                sub = sub[key]
        except (KeyError, IndexError, TypeError):
            continue  # form key not provided; leave its source as-is
        form_value = sub
        source, value = leaf.field.write(form_value, leaf.source)
        key = source.path_t.__ops__
        if key in planned:
            prior_source, prior_value, prior_path = planned[key]
            if prior_value != value:
                raise ProjectionError(
                    'conflicting write-back values for source %r: %r '
                    '(from %r) != %r (from %r)'
                    % (source, prior_value, '.'.join(prior_path),
                       value, '.'.join(leaf.form_path)))
        else:
            planned[key] = (source, value, leaf.form_path)
    return [(source, value) for source, value, _ in planned.values()]


def _coerce_path(path):
    if isinstance(path, Path):
        return path
    if type(path) is TType:
        return Path(path)
    if isinstance(path, str):
        return Path.from_text(path)
    raise TypeError('expected source path to be a string, Path, or T, not %r'
                    % (type(path).__name__,))


def _join_paths(base, extra):
    if len(base) == 0:
        return extra
    if len(extra) == 0:
        return base
    return Path(base, extra)


def _func_repr(func):
    return getattr(func, '__name__', repr(func))


def load(target, fields, base=None):
    """Convenience wrapper around :class:`Project` and
    :meth:`Project.load`.
    """
    return Project(fields, base=base).load(target)


def dump(form, target, fields, base=None, in_place=False):
    """Convenience wrapper around :class:`Project` and
    :meth:`Project.dump`.
    """
    return Project(fields, base=base).dump(form, target, in_place=in_place)
