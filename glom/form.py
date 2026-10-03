"""``glom`` specs describe how to *read* a projection out of a
target. Interfaces which project an internal config into an editing
form usually end up hand-writing a second, inverse mapping to save
the form values back.

The :class:`Form` and :class:`Field` specifier types in this module
make that inverse mapping explicit: every form field declares its
source path together with the pair of conversion functions used to
project the value out (``forward``) and to write it back
(``backward``).

Reading a :class:`Form` produces the form view; writing a view back
takes the original target as the base and updates only the parts
covered by the projection, leaving the original object untouched::

  >>> target = {'user': {'name': 'Ada', 'age': 36}, 'theme': 'dark'}
  >>> form = Form({
  ...     'name': Field('user.name'),
  ...     'age': Field('user.age', forward=str, backward=int),
  ... })
  >>> form.read(target)
  {'name': 'Ada', 'age': '36'}
  >>> updated = form.write(target, {'name': 'Grace', 'age': '37'})
  >>> updated
  {'user': {'name': 'Grace', 'age': 37}, 'theme': 'dark'}
  >>> target  # the original target is never mutated
  {'user': {'name': 'Ada', 'age': 36}, 'theme': 'dark'}

Reversibility is never assumed. A field with only a ``forward``
function is read-only, and a lossy ``default`` does not make a field
writable either; only an explicitly declared ``backward`` function
(or no conversion at all) makes a field writable. This first version
deliberately does not try to infer the inverse of arbitrary Python
callables.
"""
import copy
from collections.abc import Mapping

from .core import glom, GlomError, BadSpec, bbrepr, _MISSING, Path, TType
from .mutation import assign


class FormError(GlomError):
    "Base exception for :class:`Form` and :class:`Field` failures."


class FieldNotWritableError(FormError, TypeError):
    """Raised when a form view writes a changed value back through a
    :class:`Field` which has no explicit ``backward`` function.

    One-way conversions and lossy defaults are never treated as
    reversible, so such fields must round-trip unchanged.
    """


class FormConflictError(FormError, ValueError):
    """Raised when two or more form fields sharing the same source
    path produce differing values on write-back. The whole write is
    aborted and the target is left untouched.
    """
    def __init__(self, source, field_names, values):
        self.source = source
        self.field_names = list(field_names)
        self.values = list(values)

    def get_message(self):
        return ('fields %r share source %s but reversed to differing values: %r'
                % (self.field_names, bbrepr(self.source), self.values))

    def __repr__(self):
        cn = self.__class__.__name__
        return f'{cn}({self.source!r}, {self.field_names!r}, {self.values!r})'


def _source_key(source):
    # Path and TType are unhashable, so group sources by their repr;
    # two specs with the same repr denote the same source.
    return source if isinstance(source, str) else bbrepr(source)


class Field:
    """Declares a single form field as an explicit, writable
    projection of one source path.

    Args:
       source: Path to the source value, as a string, :class:`~glom.Path`,
          or :data:`~glom.T` expression.
       forward (callable): Converts the source value into the form
          value on read. Defaults to the identity.
       backward (callable): Converts the form value back into a source
          value on write. Must be declared explicitly for the field to
          be writable; it is never inferred from *forward*.
       default: Value to read when the source path is missing. A
          default only affects reads and does not make a one-way field
          writable.

    A field with neither *forward* nor *backward* is a plain
    pass-through and is writable both ways::

      >>> Field('a.b').writable
      True
      >>> Field('a.b', forward=str).writable
      False
      >>> Field('a.b', forward=str, backward=int).writable
      True
      >>> Field('a.b', forward=str, default='n/a').writable
      False
    """
    def __init__(self, source, forward=None, backward=None, default=_MISSING):
        if not isinstance(source, (str, Path, TType)):
            raise BadSpec('Field source must be a path string, Path, or T expression,'
                          ' got: %r' % (source,))
        for name, func in (('forward', forward), ('backward', backward)):
            if func is not None and not callable(func):
                raise BadSpec('Field %s must be callable, got: %r' % (name, func))
        self.source = source
        self.forward = forward
        self.backward = backward
        self.default = default

    @property
    def writable(self):
        """Whether this field can be written back: either it has an
        explicit ``backward`` function, or it applies no conversion at
        all (identity in both directions)."""
        return self.backward is not None or self.forward is None

    def read(self, target):
        "Project this field's value out of *target*."
        if self.default is _MISSING:
            value = glom(target, self.source)
        else:
            value = glom(target, self.source, default=self.default)
        if self.forward is not None:
            value = self.forward(value)
        return value

    def glomit(self, target, scope):
        return self.read(target)

    def reverse(self, form_value):
        """Convert a form value back into a source value. Raises
        :exc:`FieldNotWritableError` if the field is not writable."""
        if not self.writable:
            raise FieldNotWritableError(
                'field with source %s is not writable; declare an explicit'
                ' backward function to make it reversible' % bbrepr(self.source))
        if self.backward is not None:
            return self.backward(form_value)
        return form_value

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, bbrepr(self.source))


class Form:
    """A composable, explicitly writable projection spec.

    *fields* maps form field names to :class:`Field` instances, or to
    nested ``Form`` instances for grouping::

      >>> target = {'user': {'name': 'Ada', 'age': 36}, 'theme': 'dark'}
      >>> form = Form({
      ...     'profile': Form({'name': Field('user.name')}),
      ...     'age': Field('user.age', forward=str, backward=int),
      ... })
      >>> form.read(target)
      {'profile': {'name': 'Ada'}, 'age': '36'}

    Writing a view back validates the whole view before touching
    anything: several fields may share one source path, but only if
    their reversed values agree, otherwise the write fails with
    :exc:`FormConflictError` and the target is preserved as-is::

      >>> form = Form({
      ...     'age_int': Field('user.age', forward=str, backward=int),
      ...     'age_float': Field('user.age', forward=str, backward=float),
      ... })
      >>> form.write(target, {'age_int': '40', 'age_float': '40.5'})
      Traceback (most recent call last):
      ...
      FormConflictError: fields ['age_int', 'age_float'] share source 'user.age'...
    """
    def __init__(self, fields):
        if not isinstance(fields, Mapping):
            raise BadSpec('Form fields must be a mapping of name to Field or Form,'
                          ' got: %r' % (fields,))
        self.fields = dict(fields)
        for name, spec in self.fields.items():
            if not isinstance(spec, (Field, Form)):
                raise BadSpec('Form field %r must be a Field or Form, got: %r'
                              % (name, spec))

    def read(self, target):
        "Project *target* into a form view (a nested dict)."
        return {name: spec.read(target) for name, spec in self.fields.items()}

    def glomit(self, target, scope):
        return self.read(target)

    def write(self, target, view):
        """Write a form *view* back, returning a new target based on
        *target* with only the projected paths updated. The original
        target is never mutated, and unprojected values are preserved.
        """
        updates = []
        self._plan(target, view, form_path=(), updates=updates)
        new_target = copy.deepcopy(target)
        for source, value in updates:
            assign(new_target, source, value)
        return new_target

    def _plan(self, target, view, form_path, updates):
        if not isinstance(view, Mapping):
            raise FormError('expected a mapping for form view at %r, got: %r'
                            % ('.'.join(form_path) or '(root)', view))
        missing = [name for name in self.fields if name not in view]
        if missing:
            raise FormError('form view at %r is missing fields: %r'
                            % ('.'.join(form_path) or '(root)', sorted(missing)))
        extra = [name for name in view if name not in self.fields]
        if extra:
            raise FormError('form view at %r has unexpected fields: %r'
                            % ('.'.join(form_path) or '(root)', sorted(extra)))

        by_source = {}
        for name, spec in self.fields.items():
            if isinstance(spec, Form):
                spec._plan(target, view[name], form_path + (name,), updates)
                continue
            new_value = view[name]
            if new_value == spec.read(target):
                continue  # unchanged round-trip needs no write, writable or not
            reversed_value = spec.reverse(new_value)
            key = _source_key(spec.source)
            entry = by_source.setdefault(key, [spec.source, [], []])
            entry[1].append(reversed_value)
            entry[2].append('.'.join(form_path + (name,)))

        for entry in by_source.values():
            source, values, field_names = entry
            first = values[0]
            if any(value != first for value in values[1:]):
                raise FormConflictError(source, field_names, values)
            updates.append((source, first))

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.fields)
