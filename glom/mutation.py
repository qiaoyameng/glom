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

from .core import (Path, T, S, Spec, glom, Call, UnregisteredTarget, GlomError,
                   PathAccessError, UP, ROOT)
from .core import TType, register_op, TargetRegistry, bbrepr, PathAssignError, arg_val, _assign_op
from boltons.typeutils import make_sentinel


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


UPDATE_TARGET = make_sentinel('UPDATE_TARGET')
UPDATE_TARGET.__doc__ = """
Scope key used by :class:`~glom.Update` to expose the original
top-level target of the enclosing :func:`~glom.glom` call to each
leaf's update spec. Access it with ``S[UPDATE_TARGET]``.
"""

UPDATE_ARGS = make_sentinel('UPDATE_ARGS')
UPDATE_ARGS.__doc__ = """
Scope key used by :class:`~glom.Update` to expose the call arguments
(the *scope* mapping passed to :func:`~glom.glom`) to each leaf's
update spec. Access it with ``S[UPDATE_ARGS]``.
"""


class PathUpdateError(PathAssignError):
    """This :exc:`GlomError` subtype is raised when a leaf-wise
    :class:`~glom.Update` fails, either while evaluating a leaf's
    update spec ("converting" the old value) or while writing the
    result back.

    Because :class:`~glom.Update` processes its destinations one at a
    time, later leaves are not processed after a failure, but leaves
    updated earlier are left in place -- the operation is not rolled
    back. The error message states how many leaves were updated
    successfully and that those prior modifications are retained.

    Like other glom errors, the target-spec trace leading up to the
    failure is preserved.
    """
    def __init__(self, exc, path, dest_name, updated_count=0, total_count=None):
        super(PathUpdateError, self).__init__(exc, path, dest_name)
        self.updated_count = updated_count
        self.total_count = total_count

    def get_message(self):
        msg = ('could not update %r on object at %r, got error: %r'
               % (self.dest_name, self.path, self.exc))
        if self.updated_count:
            msg += ('; %d leaf target(s) were updated before this failure'
                    ' and those modifications were retained (not rolled back)'
                    % self.updated_count)
        elif self.total_count:
            msg += '; no leaf targets were updated, so no prior modifications were retained'
        return msg

    def __repr__(self):
        cn = self.__class__.__name__
        return (f'{cn}({self.exc!r}, {self.path!r}, {self.dest_name!r},'
                f' updated_count={self.updated_count!r}, total_count={self.total_count!r})')


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


_MISSING_SENTINEL = make_sentinel('UPDATE_MISSING_UNSET')


class Update:
    """*New in glom 25.12*

    The ``Update`` specifier type performs a leaf-wise in-place update:
    it finds every target at *path* (including wildcard matches) and,
    for each one, evaluates *spec* with the leaf's **current value as
    the target**, then writes the result back to that same leaf.

    This differs from :class:`~glom.Assign`, which evaluates its value
    once against the whole top-level target and writes that one result
    to every match. With ``Update``, each leaf is computed from its own
    old value, so a wildcard never collapses all leaves onto a single
    computed result::

      >>> target = {'orders': [{'price': 100}, {'price': 200}]}
      >>> _ = glom(target, Update('orders.*.price', lambda price: price * 2))
      >>> pprint(target)
      {'orders': [{'price': 200}, {'price': 400}]}

    The update *spec* runs with the leaf value as its target (``T``),
    and two explicit scope entries are available for reading data that
    is not on the leaf:

      * ``S[UPDATE_TARGET]`` -- the original top-level target of the
        :func:`~glom.glom` call.
      * ``S[UPDATE_ARGS]`` -- the call arguments (the *scope* mapping
        passed to :func:`~glom.glom`).

    For example, applying a per-call multiplier to each price using a
    glom ``T``-expression (which can reference ``S``)::

      >>> target = {'orders': [{'price': 100}, {'price': 200}]}
      >>> spec = Update('orders.*.price', T * S['factor'])
      >>> _ = glom(target, spec, scope={'factor': 0.9})
      >>> pprint(target)
      {'orders': [{'price': 90.0}, {'price': 180.0}]}

    The call arguments are also reachable explicitly as
    ``S[UPDATE_ARGS]['factor']``, and the original target as
    ``S[UPDATE_TARGET]``. Combine the two with :class:`~glom.Call`,
    for instance to subtract a per-order count derived from the
    original target::

      >>> def adjust(price, root_target, factor):
      ...     return round(price * factor) - len(root_target['orders'])
      >>> target = {'orders': [{'price': 100}, {'price': 200}]}
      >>> spec = Update('orders.*.price', Call(adjust, args=(
      ...     T, S[UPDATE_TARGET], S[UPDATE_ARGS]['factor'])))
      >>> _ = glom(target, spec, scope={'factor': 0.9})
      >>> pprint(target)
      {'orders': [{'price': 88}, {'price': 178}]}

    Note that a plain Python ``lambda`` receives only the leaf value;
    reference ``S`` from a glom spec like a ``T``-expression or
    :class:`~glom.Call`, as above, rather than from inside a raw
    ``lambda`` body.

    When a wildcard path resolves to the same physical leaf more than
    once (for example, because two keys reference the same shared
    object), that leaf is updated only **once**, at the position at
    which it was first discovered.

    By default a missing parent or leaf raises
    :class:`~glom.PathAccessError`, just like a read. Pass
    ``skip_missing=True`` to leave missing leaves untouched. Pass a
    callable as *missing* to backfill missing intermediate structures
    (see :class:`~glom.Assign` for the semantics); a still-missing
    final leaf is created when *missing* is provided.

    Updates are applied one leaf at a time. If evaluating *spec* or
    writing the result fails, processing stops and a
    :class:`~glom.PathUpdateError` is raised. Leaves updated earlier are
    not rolled back; the error reports how many leaves were updated and
    that those modifications were retained.

    ``Update`` supports attribute, key, index, and registered target
    types, just like :class:`~glom.Assign`.
    """
    def __init__(self, path, spec, missing=_MISSING_SENTINEL, skip_missing=False):
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
        if self.op not in '[.P':
            raise ValueError('last part of path must be setattr or setitem')

        self._orig_path = path
        self.path = path[:-1]
        self.spec = spec
        self.skip_missing = skip_missing

        if missing is _MISSING_SENTINEL:
            self.missing = None
            self._missing_explicit = False
        else:
            if missing is not None and not callable(missing):
                raise TypeError(f'expected missing to be callable, not {missing!r}')
            self.missing = missing
            self._missing_explicit = True

    def _resolve_dests(self, target, scope):
        """Return a list of (dest_object, op, arg, write_path) tuples,
        one per unique physical leaf, in first-discovery order."""
        path = self.path
        if path.startswith(S):
            dest_target = scope[UP]
            dest_path = path.from_t()
        else:
            dest_target = target
            dest_path = path

        layers = dest_path.path_t.__stars__()
        op, arg = self.op, self.arg

        if not layers:
            # single (non-wildcard) destination; emulate Assign's
            # missing-structure backfill behavior
            write_path = self._orig_path
            try:
                dest = scope[glom](dest_target, dest_path, scope)
            except PathAccessError as pae:
                if self._missing_explicit:
                    # backfill the missing intermediate containers only,
                    # without assuming a value for the final leaf
                    dest = self._backfill(dest_target, pae, scope)
                    return [(dest, op, arg, write_path)]
                if self.skip_missing:
                    return []
                raise
            return [(dest, op, arg, write_path)]

        # wildcard destinations: evaluate the parent path, which yields
        # nested lists according to the star layers
        try:
            resolved = scope[glom](dest_target, dest_path, scope)
        except PathAccessError:
            if self.skip_missing:
                return []
            raise

        flat = resolved
        for _ in range(layers - 1):
            flat = sum(flat, [])

        seen = set()
        results = []
        for dest in flat:
            key = (id(dest), op, arg)
            if key in seen:
                continue
            seen.add(key)
            results.append((dest, op, arg, self._orig_path))
        return results

    def _backfill(self, dest_target, pae, scope):
        part_idx = pae.part_idx
        parent_items = self._orig_path.items()
        if part_idx < len(parent_items) - 1:
            cur = dest_target
            for op, arg in parent_items[:part_idx]:
                cur = self._step(cur, op, arg, scope)
            for op, arg in parent_items[part_idx:-1]:
                nxt = self.missing()
                _assign_op(dest=cur, op=op, arg=arg, val=nxt,
                           path=self._orig_path, scope=scope)
                cur = nxt
            return cur
        # the missing part is the final leaf's own container
        cur = dest_target
        for op, arg in parent_items[:-1]:
            cur = self._step(cur, op, arg, scope)
        return cur

    def _step(self, cur, op, arg, scope):
        if op == '[':
            return cur[arg]
        if op == '.':
            return getattr(cur, arg)
        return scope[TargetRegistry].get_handler('get', cur)(cur, arg)

    def glomit(self, target, scope):
        dests = self._resolve_dests(target, scope)

        root_scope = scope[ROOT]
        scope.maps[0][UPDATE_TARGET] = target
        scope.maps[0][UPDATE_ARGS] = root_scope
        leaf_path = self._orig_path[-1:]
        updated_count = 0
        try:
            for dest, op, arg, write_path in dests:
                try:
                    old_val = scope[glom](dest, leaf_path, scope)
                except PathAccessError:
                    if self._missing_explicit:
                        old_val = self.missing()
                    elif self.skip_missing:
                        continue
                    else:
                        raise

                try:
                    new_val = scope[glom](old_val, self.spec, scope)
                except Exception as e:
                    raise PathUpdateError(e, write_path, arg,
                                          updated_count=updated_count,
                                          total_count=len(dests)) from e
                try:
                    _assign_op(dest=dest, op=op, arg=arg, val=new_val,
                               path=write_path, scope=scope)
                except PathAssignError as pae:
                    raise PathUpdateError(pae.exc, write_path, arg,
                                          updated_count=updated_count,
                                          total_count=len(dests)) from pae
                except Exception as e:
                    raise PathUpdateError(e, write_path, arg,
                                          updated_count=updated_count,
                                          total_count=len(dests)) from e
                updated_count += 1
        finally:
            scope.maps[0].pop(UPDATE_TARGET, None)
            scope.maps[0].pop(UPDATE_ARGS, None)
        return target

    def __repr__(self):
        cn = self.__class__.__name__
        extra = ''
        if self.skip_missing:
            extra += f', skip_missing={self.skip_missing!r}'
        if self._missing_explicit:
            extra += f', missing={bbrepr(self.missing)}'
        return f'{cn}({self._orig_path!r}, {self.spec!r}{extra})'


def update(obj, path, spec, missing=_MISSING_SENTINEL, skip_missing=False, scope=None):
    """The ``update()`` function is the convenient wrapper around
    :class:`~glom.Update`, leaf-wise in-place mutation of an existing
    data structure::

      >>> target = {'orders': [{'price': 100}, {'price': 200}]}
      >>> _ = update(target, 'orders.*.price', lambda price: price + 1)
      >>> pprint(target)
      {'orders': [{'price': 101}, {'price': 201}]}

    *scope* holds the call arguments made available to each leaf's
    update spec as ``S[...]`` and ``S[UPDATE_ARGS]``. See
    :class:`~glom.Update` for full semantics.

    .. versionadded:: 25.12
    """
    kwargs = {}
    if scope is not None:
        kwargs['scope'] = scope
    return glom(obj, Update(path, spec, missing=missing,
                           skip_missing=skip_missing), **kwargs)


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
