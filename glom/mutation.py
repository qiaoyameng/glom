"""By default, glom aims to safely return a transformed copy of your
data. But sometimes you really need to transform an existing object.

When you already have a large or complex bit of nested data that you
are sure you want to modify in-place, glom has you covered, with the
:func:`~glom.assign` function, and the :func:`~glom.Assign` specifier
type.

.. warning::

   glom's deep assignment is powerful, and incorrect use can result in 
   unintended assignments to global state, including class and module 
   attributes, as well as function defaults. 
   
   Be careful when writing assignment specs, and especially careful when 
   any part of the spec is data-driven or provided by an end user.

"""
import operator
from pprint import pprint

from .core import Path, T, S, Spec, glom, UnregisteredTarget, GlomError, PathAccessError, UP
from .core import TType, register_op, TargetRegistry, bbrepr, PathAssignError, arg_val, _assign_op


try:
    basestring
except NameError:
    basestring = str


if getattr(__builtins__, '__dict__', None) is not None:
    # pypy's __builtins__ is a module, as is CPython's REPL, but at
    # normal execution time it's a dict?
    __builtins__ = __builtins__.__dict__


class PathDeleteError(PathAssignError):
    """This :exc:`GlomError` subtype is raised when an assignment fails,
    stemming from an :func:`~glom.delete` call or other
    :class:`~glom.Delete` usage.

    One example would be deleting an out-of-range position in a list::

      >>> delete(["short", "list"], Path(5))
      Traceback (most recent call last):
      ...
      PathDeleteError: could not delete 5 on object at Path(), got error: IndexError(...

    Other assignment failures could be due to deleting a read-only
    ``@property`` or exception being raised inside a ``__delattr__()``.

    """
    def get_message(self):
        return ('could not delete %r on object at %r, got error: %r'
                % (self.dest_name, self.path, self.exc))


def _apply_for_each(func, path, val):
    layers = path.path_t.__stars__()
    if layers:
        for i in range(layers - 1):
            val = sum(val, [])  # flatten out the extra layers
        for inner in val:
            func(inner)
    else:
        func(val)


class Assign:
    """*New in glom 18.3.0*

    The ``Assign`` specifier type enables glom to modify the target,
    performing a "deep-set" to mirror glom's original deep-get use
    case.

    ``Assign`` can be used to perform spot modifications of large data
    structures when making a copy is not desired::

      # deep assignment into a nested dictionary
      >>> target = {'a': {}}
      >>> spec = Assign('a.b', 'value')
      >>> _ = glom(target, spec)
      >>> pprint(target)
      {'a': {'b': 'value'}}

    The value to be assigned can also be a :class:`~glom.Spec`, which
    is useful for copying values around within the data structure::

      # copying one nested value to another
      >>> _ = glom(target, Assign('a.c', Spec('a.b')))
      >>> pprint(target)
      {'a': {'b': 'value', 'c': 'value'}}

    Another handy use of Assign is to deep-apply a function::

      # sort a deep nested list
      >>> target={'a':{'b':[3,1,2]}}
      >>> _ = glom(target, Assign('a.b', Spec(('a.b',sorted))))
      >>> pprint(target)
      {'a': {'b': [1, 2, 3]}}

    Like many other specifier types, ``Assign``'s destination path can be
    a :data:`~glom.T` expression, for maximum control::

      # changing the error message of an exception in an error list
      >>> err = ValueError('initial message')
      >>> target = {'errors': [err]}
      >>> _ = glom(target, Assign(T['errors'][0].args, ('new message',)))
      >>> str(err)
      'new message'

    ``Assign`` has built-in support for assigning to attributes of
    objects, keys of mappings (like dicts), and indexes of sequences
    (like lists). Additional types can be registered through
    :func:`~glom.register()` using the ``"assign"`` operation name.

    Attempting to assign to an immutable structure, like a
    :class:`tuple`, will result in a
    :class:`~glom.PathAssignError`. Attempting to assign to a path
    that doesn't exist will raise a :class:`~PathAccessError`.

    To automatically backfill missing structures, you can pass a
    callable to the *missing* argument. This callable will be called
    for each path segment along the assignment which is not
    present.

       >>> target = {}
       >>> assign(target, 'a.b.c', 'hi', missing=dict)
       {'a': {'b': {'c': 'hi'}}}

    """
    def __init__(self, path, val, missing=None):
        # TODO: an option like require_preexisting or something to
        # ensure that a value is mutated, not just added. Current
        # workaround is to do a Check().
        if isinstance(path, basestring):
            path = Path.from_text(path)
        elif type(path) is TType:
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError('path argument must be a .-delimited string, Path, T, or S')

        try:
            self.op, self.arg = path.items()[-1]
        except IndexError:
            raise ValueError('path must have at least one element')
        self._orig_path = path
        self.path = path[:-1]

        if self.op not in '[.P':
            # maybe if we add null-coalescing this should do something?
            raise ValueError('last part of path must be setattr or setitem')
        self.val = val

        if missing is not None:
            if not callable(missing):
                raise TypeError(f'expected missing to be callable, not {missing!r}')
        self.missing = missing

    def glomit(self, target, scope):
        val = arg_val(target, self.val, scope)

        op, arg, path = self.op, self.arg, self.path
        if self.path.startswith(S):
            dest_target = scope[UP]
            dest_path = self.path.from_t()
        else:
            dest_target = target
            dest_path = self.path
        try:
            dest = scope[glom](dest_target, dest_path, scope)
        except PathAccessError as pae:
            if not self.missing:
                raise

            remaining_path = self._orig_path[pae.part_idx + 1:]
            val = scope[glom](self.missing(), Assign(remaining_path, val, missing=self.missing), scope)

            op, arg = self._orig_path.items()[pae.part_idx]
            path = self._orig_path[:pae.part_idx]
            dest = scope[glom](dest_target, path, scope)

        # TODO: forward-detect immutable dest?
        _apply = lambda dest: _assign_op(
            dest=dest, op=op, arg=arg, val=val, path=path, scope=scope)
        _apply_for_each(_apply, path, dest)

        return target

    def __repr__(self):
        cn = self.__class__.__name__
        if self.missing is None:
            return f'{cn}({self._orig_path!r}, {self.val!r})'
        return f'{cn}({self._orig_path!r}, {self.val!r}, missing={bbrepr(self.missing)})'


def assign(obj, path, val, missing=None):
    """*New in glom 18.3.0*

    The ``assign()`` function provides convenient "deep set"
    functionality, modifying nested data structures in-place::

      >>> target = {'a': [{'b': 'c'}, {'d': None}]}
      >>> _ = assign(target, 'a.1.d', 'e')  # let's give 'd' a value of 'e'
      >>> pprint(target)
      {'a': [{'b': 'c'}, {'d': 'e'}]}

    Missing structures can also be automatically created with the
    *missing* parameter. For more information and examples, see the
    :class:`~glom.Assign` specifier type, which this function wraps.
    """
    return glom(obj, Assign(path, val, missing=missing))


class Update:
    """*New in glom 25.12.1*

    The ``Update`` specifier type performs a "deep update", modifying
    each value found at *path* based on its own old value. Where
    :class:`~glom.Assign` writes a single value computed once from the
    whole target, ``Update`` evaluates its *val* spec once per matched
    location, with the old value at that location as the input::

      # apply a discount to every price, each computed from its own old value
      >>> target = {'orders': [{'price': 100}, {'price': 40}]}
      >>> _ = glom(target, Update('orders.*.price', T * 0.9))
      >>> pprint(target)
      {'orders': [{'price': 90.0}, {'price': 36.0}]}

    Within the *val* spec, ``T`` refers to the old value at the
    current location. The original target is available through the
    explicit ``'target'`` scope entry, and parameters passed to the
    call (e.g. ``glom(target, spec, scope={'discount': 0.9})``) are
    available through :data:`~glom.S` as usual::

      >>> target = {'tax_rate': 0.2, 'orders': [{'price': 100}, {'price': 50}]}
      >>> spec = Update('orders.*.price', T * (S['target']['tax_rate'] + 1) * S['discount'])
      >>> _ = glom(target, spec, scope={'discount': 0.5})
      >>> pprint(target)
      {'orders': [{'price': 60.0}, {'price': 30.0}], 'tax_rate': 0.2}

    When a wildcard (``*``/``**``) path matches the same physical
    location more than once (e.g. a shared subobject reachable along
    several routes), the location is only updated once, in order of
    first discovery::

      >>> shared = {'price': 10}
      >>> target = {'a': shared, 'b': shared}
      >>> _ = glom(target, Update('*.price', T + 1))
      >>> shared
      {'price': 11}

    Whether missing paths are created is explicitly chosen by the
    caller through the *missing* argument, which works just like
    :class:`~glom.Assign`'s. When *missing* is set and a location does
    not exist yet, the *val* spec is evaluated with ``None`` as the
    old value::

      >>> target = {}
      >>> _ = glom(target, Update('a.b', (T, lambda v: (v or 0) + 1), missing=dict))
      >>> pprint(target)
      {'a': {'b': 1}}

    Without *missing*, updating a path that does not exist raises
    :class:`~glom.PathAccessError`.

    ``Update`` is not atomic: locations are updated one at a time, in
    order of discovery, and if evaluating *val* or writing a result
    fails, updates already applied to earlier locations are retained
    (not rolled back). Failures keep glom's usual path diagnostics:
    a failed read raises :class:`~glom.PathAccessError`, a failed
    *val* evaluation propagates its original error, and a failed
    write raises :class:`~glom.PathAssignError`.

    Like :class:`~glom.Assign`, ``Update`` supports assigning to
    attributes of objects, keys of mappings, and indexes of sequences
    out of the box. Additional types can be registered through
    :func:`~glom.register()` using the ``"assign"`` operation name
    (and ``"get"`` for reading the old value).

    """
    def __init__(self, path, val, missing=None):
        if isinstance(path, basestring):
            path = Path.from_text(path)
        elif type(path) is TType:
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError('path argument must be a .-delimited string, Path, T, or S')

        try:
            self.op, self.arg = path.items()[-1]
        except IndexError:
            raise ValueError('path must have at least one element')
        self._orig_path = path
        self.path = path[:-1]

        if self.op not in '[.P':
            raise ValueError('last part of path must be setattr or setitem')
        self.val = val

        if missing is not None:
            if not callable(missing):
                raise TypeError(f'expected missing to be callable, not {missing!r}')
        self.missing = missing
        self._bind_target = True

    def _get_old_val(self, dest, op, arg, scope):
        if op == '[':
            return dest[arg]
        elif op == '.':
            return getattr(dest, arg)
        # op == 'P'
        get = scope[TargetRegistry].get_handler('get', dest)
        return get(dest, arg)

    def _update_one(self, dest, op, arg, path, seen, scope):
        if id(dest) in seen:
            return
        seen.add(id(dest))
        try:
            old_val = self._get_old_val(dest, op, arg, scope)
        except GlomError:
            raise
        except Exception as e:
            if not self.missing:
                raise PathAccessError(e, self._orig_path, len(self._orig_path.items()) - 1)
            old_val = None
        new_val = scope[glom](old_val, self.val, scope)
        _assign_op(dest=dest, op=op, arg=arg, val=new_val, path=path, scope=scope)

    def glomit(self, target, scope):
        if self._bind_target:
            scope.update({'target': target})

        op, arg, path = self.op, self.arg, self.path
        if self.path.startswith(S):
            dest_target = scope[UP]
            dest_path = self.path.from_t()
        else:
            dest_target = target
            dest_path = self.path
        try:
            dest = scope[glom](dest_target, dest_path, scope)
        except PathAccessError as pae:
            if not self.missing:
                raise

            remaining_path = self._orig_path[pae.part_idx + 1:]
            sub_update = Update(remaining_path, self.val, missing=self.missing)
            sub_update._bind_target = False  # keep the original target binding
            val = scope[glom](self.missing(), sub_update, scope)

            op, arg = self._orig_path.items()[pae.part_idx]
            path = self._orig_path[:pae.part_idx]
            dest = scope[glom](dest_target, path, scope)

            # the backfilled subtree is already fully computed, assign
            # it directly, just like Assign
            _apply = lambda dest: _assign_op(
                dest=dest, op=op, arg=arg, val=val, path=path, scope=scope)
            _apply_for_each(_apply, path, dest)

            return target

        seen = set()
        _apply_for_each(lambda dest: self._update_one(dest, op, arg, path, seen, scope),
                        path, dest)

        return target

    def __repr__(self):
        cn = self.__class__.__name__
        if self.missing is None:
            return f'{cn}({self._orig_path!r}, {self.val!r})'
        return f'{cn}({self._orig_path!r}, {self.val!r}, missing={bbrepr(self.missing)})'


def update(obj, path, val, missing=None):
    """*New in glom 25.12.1*

    The ``update()`` function provides convenient "deep update"
    functionality, modifying each value found at *path* based on its
    own old value::

      >>> target = {'orders': [{'price': 100}, {'price': 40}]}
      >>> _ = update(target, 'orders.*.price', lambda price: price * 0.9)
      >>> pprint(target)
      {'orders': [{'price': 90.0}, {'price': 36.0}]}

    For more information and examples, see the :class:`~glom.Update`
    specifier type, which this function wraps.
    """
    return glom(obj, Update(path, val, missing=missing))


_ALL_BUILTIN_TYPES = [v for v in __builtins__.values() if isinstance(v, type)]
_BUILTIN_BASE_TYPES = [v for v in _ALL_BUILTIN_TYPES
                       if not issubclass(v, tuple([t for t in _ALL_BUILTIN_TYPES
                                                   if t not in (v, type, object)]))]
_UNASSIGNABLE_BASE_TYPES = tuple(set(_BUILTIN_BASE_TYPES)
                                 - {dict, list, BaseException, object, type})


def _set_sequence_item(target, idx, val):
    target[int(idx)] = val


def _assign_autodiscover(type_obj):
    # TODO: issubclass or "in"?
    if issubclass(type_obj, _UNASSIGNABLE_BASE_TYPES):
        return False

    if callable(getattr(type_obj, '__setitem__', None)):
        if callable(getattr(type_obj, 'index', None)):
            return _set_sequence_item
        return operator.setitem

    return setattr


register_op('assign', auto_func=_assign_autodiscover, exact=False)


class Delete:
    """
    In addition to glom's core "deep-get" and ``Assign``'s "deep-set",
    the ``Delete`` specifier type performs a "deep-del", which can
    remove items from larger data structures by key, attribute, and
    index.

    >>> target = {'dict': {'x': [5, 6, 7]}}
    >>> glom(target, Delete('dict.x.1'))
    {'dict': {'x': [5, 7]}}
    >>> glom(target, Delete('dict.x'))
    {'dict': {}}

    If a target path is missing, a :exc:`PathDeleteError` will be
    raised. To ignore missing targets, use the ``ignore_missing``
    flag:

    >>> glom(target, Delete('does_not_exist', ignore_missing=True))
    {'dict': {}}

    ``Delete`` has built-in support for deleting attributes of
    objects, keys of dicts, and indexes of sequences
    (like lists). Additional types can be registered through
    :func:`~glom.register()` using the ``"delete"`` operation name.

    .. versionadded:: 20.5.0
    """
    def __init__(self, path, ignore_missing=False):
        if isinstance(path, basestring):
            path = Path.from_text(path)
        elif type(path) is TType:
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError('path argument must be a .-delimited string, Path, T, or S')

        try:
            self.op, self.arg = path.items()[-1]
        except IndexError:
            raise ValueError('path must have at least one element')
        self._orig_path = path
        self.path = path[:-1]

        if self.op not in '[.P':
            raise ValueError('last part of path must be an attribute or index')

        self.ignore_missing = ignore_missing

    def _del_one(self, dest, op, arg, scope):
        if op == '[':
            try:
                del dest[arg]
            except IndexError as e:
                if not self.ignore_missing:
                    raise PathDeleteError(e, self.path, arg)
        elif op == '.':
            try:
                delattr(dest, arg)
            except AttributeError as e:
                if not self.ignore_missing:
                    raise PathDeleteError(e, self.path, arg)
        elif op == 'P':
            _delete = scope[TargetRegistry].get_handler('delete', dest)
            try:
                _delete(dest, arg)
            except Exception as e:
                if not self.ignore_missing:
                    raise PathDeleteError(e, self.path, arg)

    def glomit(self, target, scope):
        op, arg, path = self.op, self.arg, self.path
        if self.path.startswith(S):
            dest_target = scope[UP]
            dest_path = self.path.from_t()
        else:
            dest_target = target
            dest_path = self.path
        try:
            dest = scope[glom](dest_target, dest_path, scope)
        except PathAccessError as pae:
            if not self.ignore_missing:
                raise
        else:
            _apply_for_each(lambda dest: self._del_one(dest, op, arg, scope), path, dest)

        return target

    def __repr__(self):
        cn = self.__class__.__name__
        return f'{cn}({self._orig_path!r})'


def delete(obj, path, ignore_missing=False):
    """
    The ``delete()`` function provides "deep del" functionality,
    modifying nested data structures in-place::

      >>> target = {'a': [{'b': 'c'}, {'d': None}]}
      >>> delete(target, 'a.0.b')
      {'a': [{}, {'d': None}]}

    Attempting to delete missing keys, attributes, and indexes will
    raise a :exc:`PathDeleteError`. To ignore these errors, use the
    *ignore_missing* argument::

      >>> delete(target, 'does_not_exist', ignore_missing=True)
      {'a': [{}, {'d': None}]}

    For more information and examples, see the :class:`~glom.Delete`
    specifier type, which this convenience function wraps.

    .. versionadded:: 20.5.0
    """
    return glom(obj, Delete(path, ignore_missing=ignore_missing))


def _del_sequence_item(target, idx):
    del target[int(idx)]


def _delete_autodiscover(type_obj):
    if issubclass(type_obj, _UNASSIGNABLE_BASE_TYPES):
        return False

    if callable(getattr(type_obj, '__delitem__', None)):
        if callable(getattr(type_obj, 'index', None)):
            return _del_sequence_item
        return operator.delitem
    return delattr


register_op('delete', auto_func=_delete_autodiscover, exact=False)
