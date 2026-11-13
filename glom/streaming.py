"""glom's helpers for streaming use cases.

Specifier types which yield their results incrementally so that they
can be applied to targets which are themselves streaming (e.g. chunks
of rows from a database, lines from a file) without excessive memory
usage.

glom's streaming functionality revolves around a single :class:`Iter`
Specifier type, which has methods to transform the target stream.
"""

import sys
from itertools import islice, dropwhile, takewhile, chain
from functools import partial
try:
    from itertools import imap, ifilter
except ImportError:
    # py3
    imap = map
    ifilter = filter

from boltons.iterutils import split_iter, chunked_iter, windowed_iter, unique_iter, first
from boltons.funcutils import FunctionBuilder

from .core import (glom, T, STOP, SKIP, _MISSING, Path, PathAccessError,
                   GlomError,
                   TargetRegistry, Call, Spec, Pipe, S, bbrepr,
                   format_invocation)
from .matching import Check


def _close_iterator(iterator):
    close = getattr(iterator, 'close', None)
    if close is not None:
        close()


def _error_chain(err):
    """Return the exception and all wrapped/caused exceptions in order."""
    chain = []
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        chain.append(err)
        wrapped = getattr(err, '_GlomError__wrapped', None)
        if wrapped is None:
            wrapped = getattr(err, '__wrapped', None)
        err = wrapped or getattr(err, '__cause__', None) or getattr(err, '__context__', None)
    return tuple(chain)


def _approx_size(value, limit=None):
    """Estimate an object's shallow recursive size, stopping at *limit*."""
    total = 0
    stack = [value]
    seen = set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        try:
            total += sys.getsizeof(cur)
        except Exception:
            total += 1
        if limit is not None and total > limit:
            return total
        if isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple, set, frozenset)):
            stack.extend(cur)
    return total


def _iter_error_path(error, item_path):
    path = getattr(error, 'path', None)
    if isinstance(path, Path):
        parts = list(item_path.values())
        for op, value in path.items():
            if op not in ('.', 'P', '['):
                break
            if isinstance(value, (str, int, float, bool, type(None))):
                parts.append(value)
            else:
                break
        return Path(*parts)
    return item_path


def _path_for_item(base_path, index):
    # The root scope often carries the current Iter spec in its Path.
    # Keep enclosing indexes, but leave the spec object out of data
    # locations reported for individual records.
    parts = [part for part in base_path if not isinstance(part, Iter)]
    parts.append(index)
    return Path(*parts)


class IterFailure:
    """The failure item yielded by :meth:`Iter.attempt`.

    It keeps the source index, the spec location, and the original
    glom error chain without interrupting the rest of the stream.
    """
    __slots__ = ('index', 'path', 'item_path', 'kind', 'error', '_original',
                 'error_chain', 'original_size', 'original_omitted')

    def __init__(self, index, path, kind, error=None, original=_MISSING,
                 error_chain=(), original_size=None, original_omitted=False,
                 item_path=None):
        self.index = index
        self.path = path
        self.item_path = item_path if item_path is not None else path
        self.kind = kind
        self.error = error
        self._original = original
        self.error_chain = tuple(error_chain)
        self.original_size = original_size
        self.original_omitted = original_omitted

    @property
    def source_index(self):
        return self.index

    @property
    def status(self):
        return self.kind

    @property
    def reason(self):
        return self.kind

    @property
    def exception(self):
        return self.error

    @property
    def exc(self):
        return self.error

    @property
    def root_error(self):
        return self.error_chain[-1] if self.error_chain else self.error

    @property
    def errors(self):
        return self.error_chain

    @property
    def original_error(self):
        return self.root_error

    @property
    def cause(self):
        return self.root_error

    @property
    def error_type(self):
        return type(self.error) if self.error is not None else None

    @property
    def spec_path(self):
        path = getattr(self.error, 'path', None)
        if isinstance(path, Path):
            parts = []
            for op, value in path.items():
                if op not in ('.', 'P', '['):
                    break
                if isinstance(value, (str, int, float, bool, type(None))):
                    parts.append(value)
                else:
                    break
            return Path(*parts)
        return self.path

    @property
    def location(self):
        return self.path

    @property
    def original_value(self):
        return self.original

    @property
    def original(self):
        return None if self._original is _MISSING else self._original

    @property
    def source(self):
        return self.original

    @property
    def value(self):
        return self.original

    @property
    def has_original(self):
        return self._original is not _MISSING

    @property
    def ok(self):
        return False

    @property
    def success(self):
        return False

    @property
    def failed(self):
        return True

    @property
    def is_failure(self):
        return True

    @property
    def is_missing(self):
        return self.kind == 'missing'

    @property
    def is_skip(self):
        return self.kind == 'skip'

    @property
    def is_stop(self):
        return self.kind == 'stop'

    @property
    def is_error(self):
        return self.kind == 'error'

    def as_dict(self):
        return {
            'index': self.index,
            'path': self.path,
            'item_path': self.item_path,
            'kind': self.kind,
            'error': self.error,
            'error_chain': self.error_chain,
            'original': self.original,
            'original_size': self.original_size,
            'original_omitted': self.original_omitted,
        }

    def __getitem__(self, key):
        try:
            return self.as_dict()[key]
        except KeyError:
            raise KeyError(key) from None

    def __repr__(self):
        return (f'{self.__class__.__name__}(index={self.index!r}, '
                f'path={self.path!r}, kind={self.kind!r}, '
                f'error={self.error!r})')


AttemptFailure = IterFailure
IterAttemptFailure = IterFailure
IterError = IterFailure
Failure = IterFailure
Attempt = IterFailure
AttemptItem = IterFailure


class _CloseableIterator:
    """Close the whole internal iterator chain when the consumer stops."""

    __slots__ = ('_iterator', '_upstream', '_closed')

    def __init__(self, iterator, upstream):
        self._iterator = iterator
        self._upstream = upstream
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            _close_iterator(self._iterator)
        finally:
            if self._upstream is not self._iterator:
                _close_iterator(self._upstream)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        self.close()


class Iter:
    """``Iter()`` is glom's counterpart to Python's built-in :func:`iter()`
    function. Given an iterable target, ``Iter()`` yields the result
    of applying the passed spec to each element of the target, similar
    to the built-in ``[]`` spec, but streaming.

    The following turns a list of strings into integers using Iter(),
    before deduplicating and converting it to a tuple:

    >>> glom(['1', '2', '1', '3'], (Iter(int), set, tuple))
    (1, 2, 3)

    ``Iter()`` also has many useful methods which can be chained to
    compose a stream processing pipeline. The above can also be
    written as:

    >>> glom(['1', '2', '1', '3'], (Iter().map(int).unique(), tuple))
    (1, 2, 3)

    ``Iter()`` also respects glom's :data:`~glom.SKIP` and
    :data:`~glom.STOP` singletons for filtering and breaking
    iteration.

    Args:

       subspec: A subspec to be applied on each element from the iterable.
       sentinel: Keyword-only argument, which, when found in the
         iterable stream, causes the iteration to stop. Same as with the
         built-in :func:`iter`.
       attempt: When true, apply *subspec* through :meth:`attempt`,
         yielding :class:`IterFailure` items instead of raising for
         per-record spec failures.

    """
    def __init__(self, subspec=T, **kwargs):
        self.subspec = subspec
        self._iter_stack = kwargs.pop('_iter_stack', [])
        self._attempt = kwargs.pop('_attempt', None)

        self.sentinel = kwargs.pop('sentinel', STOP)
        attempt_opts = {
            'keep_original': kwargs.pop('keep_original', False),
            'original_budget': kwargs.pop('original_budget', None),
            'max_original_items': kwargs.pop('max_original_items', None),
        }
        if attempt_opts['original_budget'] is None:
            attempt_opts['original_budget'] = kwargs.pop('max_original_bytes', None)
        attempt = kwargs.pop('attempt', _MISSING)
        if self._attempt is None:
            if attempt is not _MISSING:
                if attempt is True:
                    self._set_attempt(**attempt_opts)
                elif attempt is not False:
                    self.subspec = attempt
                    self._set_attempt(**attempt_opts)
            elif any(value is not None and value is not False
                     for value in attempt_opts.values()):
                self._set_attempt(**attempt_opts)
        if kwargs:
            raise TypeError('unexpected keyword arguments: %r' % sorted(kwargs))
        return

    def __repr__(self):
        base_args = ()
        if self.subspec != T:
            base_args = (self.subspec,)
        base = format_invocation(self.__class__.__name__, base_args, repr=bbrepr)
        chunks = [base]
        for fname, args, _ in reversed(self._iter_stack):
            meth = getattr(self, fname)
            fb = FunctionBuilder.from_func(meth)
            fb.args = fb.args[1:]  # drop self
            arg_names = fb.get_arg_names()
            # TODO: something fancier with defaults:
            kwargs = []
            if len(args) > 1 and arg_names:
                args, kwargs = (), zip(arg_names, args)
            chunks.append('.' + format_invocation(fname, args, kwargs, repr=bbrepr))
        if self._attempt is not None:
            attempt_args = ()
            if self._attempt['subspec'] != self.subspec:
                attempt_args = (self._attempt['subspec'],)
            chunks.append('.' + format_invocation('attempt', attempt_args, repr=bbrepr))
        return ''.join(chunks)

    def glomit(self, target, scope):
        base_iterator = self._iterate(target, scope)
        iterator = base_iterator

        for _, _, callback in reversed(self._iter_stack):
            iterator = callback(iterator, scope)

        return _CloseableIterator(iterator, base_iterator)

    def _iterate(self, target, scope):
        iterate = scope[TargetRegistry].get_handler('iterate', target, path=scope[Path])
        try:
            iterator = iterate(target)
        except Exception as e:
            raise TypeError('failed to iterate on instance of type %r at %r (got %r)'
                            % (target.__class__.__name__, Path(*scope[Path]), e))

        base_path = scope[Path]
        original_budget = None
        original_items = 0
        if self._attempt is not None:
            original_budget = self._attempt['original_budget']
            original_items = self._attempt['max_original_items']
        try:
            i = -1
            while True:
                try:
                    t = next(iterator)
                except StopIteration:
                    break
                i += 1
                scope[Path] = base_path + [i]
                item_path = _path_for_item(base_path, i)
                yld = t
                if self.subspec is not T:
                    try:
                        yld = scope[glom](t, self.subspec, scope)
                    except Exception as e:
                        if self._attempt is None:
                            raise
                        if not isinstance(e, GlomError):
                            e = GlomError.wrap(e)
                        original = _MISSING
                        original_size = None
                        original_omitted = False
                        if self._attempt['keep_original']:
                            original_size = _approx_size(t, original_budget)
                            can_keep = (original_items is None
                                        or original_items > 0)
                            if original_budget is not None:
                                can_keep = can_keep and original_size <= original_budget
                                if can_keep:
                                    original_budget -= original_size
                            if can_keep:
                                original = t
                                if original_items is not None:
                                    original_items -= 1
                            else:
                                original_omitted = True
                        kind = 'missing' if isinstance(e, PathAccessError) else 'error'
                        yield IterFailure(i, _iter_error_path(e, item_path), kind, e,
                                          original=original,
                                          error_chain=_error_chain(e),
                                          original_size=original_size,
                                          original_omitted=original_omitted,
                                          item_path=item_path)
                        continue
                if yld is SKIP:
                    continue
                elif yld is self.sentinel or yld is STOP:
                    # NB: sentinel defaults to STOP so I was torn whether
                    # to also check for STOP, and landed on the side of
                    # never letting STOP through.
                    return
                yield yld
        finally:
            _close_iterator(iterator)
        return

    def _set_attempt(self, keep_original=False, original_budget=None,
                     max_original_items=None):
        if isinstance(keep_original, int) and not isinstance(keep_original, bool):
            if original_budget is None:
                original_budget = keep_original
            keep_original = True
        self._attempt = {
            'subspec': self.subspec,
            'keep_original': keep_original,
            'original_budget': original_budget,
            'max_original_items': max_original_items,
        }

    def attempt(self, subspec=_MISSING, keep_original=False,
                max_original_bytes=None, original_budget=None,
                max_original_items=None, **kwargs):
        """Return a new :class:`Iter()` which yields one result per record.

        Failures do not abort the stream. Instead of raising, a
        :class:`IterFailure` is yielded with the source index, spec
        location, and original :class:`~glom.GlomError` chain. Successful
        records are yielded unchanged.

        >>> target = [{'id': 1}, {}, {'id': 3}]
        >>> spec = Iter().attempt(T['id']).all()
        >>> results = glom(target, spec)
        >>> [r for r in results if isinstance(r, IterFailure)][0].kind
        'missing'

        *keep_original* controls whether failed source values are
        retained. *max_original_bytes* puts a byte budget on those
        retained values across the stream; once the budget is spent,
        later failures omit their originals. *max_original_items* can
        limit the number of retained originals. SKIP and STOP keep their
        normal filtering/termination behavior.
        """
        if kwargs:
            # A few intuitive aliases, kept keyword-only in spirit.
            if max_original_bytes is None:
                max_original_bytes = kwargs.pop('max_original_size', None)
            if max_original_bytes is None:
                max_original_bytes = kwargs.pop('max_original_bytes', None)
            if original_budget is None:
                original_budget = kwargs.pop('original_limit', None)
            if not kwargs:
                pass
            else:
                raise TypeError('unexpected keyword arguments: %r' % sorted(kwargs))

        if original_budget is None:
            original_budget = max_original_bytes
        if isinstance(keep_original, int) and not isinstance(keep_original, bool):
            if original_budget is None:
                original_budget = keep_original
            keep_original = True

        new_iter = type(self)(subspec=self.subspec, sentinel=self.sentinel,
                              _iter_stack=self._iter_stack,
                              _attempt=self._attempt)
        if subspec is not _MISSING:
            new_iter.subspec = subspec
        new_iter._set_attempt(keep_original=keep_original,
                              original_budget=original_budget,
                              max_original_items=max_original_items)
        return new_iter

    def _add_op(self, opname, args, callback):
        return type(self)(subspec=self.subspec, sentinel=self.sentinel,
                          _iter_stack=[(opname, args, callback)] + self._iter_stack,
                          _attempt=self._attempt)

    def map(self, subspec):
        """Return a new :class:`Iter()` spec which will apply the provided
        *subspec* to each element of the iterable.

        >>> glom(range(5), Iter().map(lambda x: x * 2).all())
        [0, 2, 4, 6, 8]

        Because a spec can be a callable, :meth:`Iter.map()` does
        everything the built-in :func:`map` does, but with the full
        power of glom specs.

        >>> glom(['a', 'B', 'C'], Iter().map(T.islower()).all())
        [True, False, False]
        """
        # whatever validation you want goes here
        # TODO: DRY the self._add_op with a decorator?
        return self._add_op(
            'map',
            (subspec,),
            lambda iterable, scope: imap(
                lambda t: scope[glom](t, subspec, scope), iterable))

    def filter(self, key=T):
        """Return a new :class:`Iter()` spec which will include only elements matching the
        given *key*.

        >>> glom(range(6), Iter().filter(lambda x: x % 2).all())
        [1, 3, 5]

        Because a spec can be a callable, :meth:`Iter.filter()` does
        everything the built-in :func:`filter` does, but with the full
        power of glom specs. For even more power, combine,
        :meth:`Iter.filter()` with :class:`Check()`.

        >>> # PROTIP: Python's ints know how many binary digits they require, using the bit_length method
        >>> glom(range(9), Iter().filter(Check(T.bit_length(), one_of=(2, 4), default=SKIP)).all())
        [2, 3, 8]

        """
        # NB: Check's validate function defaults to bool, and
        # *default* is returned on access errors as well validation
        # errors, so the lambda passed to ifilter below works fine.
        check_spec = key if isinstance(key, Check) else Check(key, default=SKIP)
        return self._add_op(
            'filter',
            (key,),
            lambda iterable, scope: ifilter(
                lambda t: scope[glom](t, check_spec, scope) is not SKIP, iterable))

    def chunked(self, size, fill=_MISSING):
        """Return a new :class:`Iter()` spec which groups elements in the iterable
        into lists of length *size*.

        If the optional *fill* argument is provided, iterables not
        evenly divisible by *size* will be padded out by the *fill*
        constant. Otherwise, the final chunk will be shorter than *size*.

        >>> list(glom(range(10), Iter().chunked(3)))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(glom(range(10), Iter().chunked(3, fill=None)))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]
        """
        kw = {'size': size}
        args = size,
        if fill is not _MISSING:
            kw['fill'] = fill
            args += (fill,)
        return self._add_op(
            'chunked', args, lambda it, scope: chunked_iter(it, **kw))

    def windowed(self, size):
        """Return a new :class:`Iter()` spec which will yield a sliding window of
        adjacent elements in the iterable. Each tuple yielded will be
        of length *size*.

        Useful for getting adjacent pairs and triples.

        >>> list(glom(range(4), Iter().windowed(2)))
        [(0, 1), (1, 2), (2, 3)]
        """
        return self._add_op(
            'windowed', (size,), lambda it, scope: windowed_iter(it, size))

    def split(self, sep=None, maxsplit=None):
        """Return a new :class:`Iter()` spec which will lazily split an iterable based
        on a separator (or list of separators), *sep*. Like
        :meth:`str.split()`, but for all iterables.

        ``split_iter()`` yields lists of non-separator values. A separator will
        never appear in the output.

        >>> target = [1, 2, None, None, 3, None, 4, None]
        >>> list(glom(target, Iter().split()))
        [[1, 2], [3], [4]]

        Note that ``split_iter`` is based on :func:`str.split`, so if
        *sep* is ``None``, ``split()`` **groups** separators. If empty lists
        are desired between two contiguous ``None`` values, simply use
        ``sep=[None]``:

        >>> list(glom(target, Iter().split(sep=[None])))
        [[1, 2], [], [3], [4], []]

        A max number of splits may also be set:

        >>> list(glom(target, Iter().split(maxsplit=2)))
        [[1, 2], [3], [4, None]]

        """
        return self._add_op(
            'split',
            (sep, maxsplit),
            lambda it, scope: split_iter(it, sep=sep, maxsplit=maxsplit))

    def flatten(self):
        """Returns a new :class:`Iter()` instance which combines iterables into a
        single iterable.

        >>> target = [[1, 2], [3, 4], [5]]
        >>> list(glom(target, Iter().flatten()))
        [1, 2, 3, 4, 5]
        """
        return self._add_op(
            'flatten',
            (),
            lambda it, scope: chain.from_iterable(it))

    def unique(self, key=T):
        """Return a new :class:`Iter()` spec which lazily filters out duplicate
        values, i.e., only the first appearance of a value in a stream will
        be yielded.

        >>> target = list('gloMolIcious')
        >>> out = list(glom(target, Iter().unique(T.lower())))
        >>> print(''.join(out))
        gloMIcus
        """
        return self._add_op(
            'unique',
            (key,),
            lambda it, scope: unique_iter(it, key=lambda t: scope[glom](t, key, scope)))


    def slice(self, *args):
        """Returns a new :class:`Iter()` spec which trims iterables in the
        same manner as :func:`itertools.islice`.

        >>> target = [0, 1, 2, 3, 4, 5]
        >>> glom(target, Iter().slice(3).all())
        [0, 1, 2]
        >>> glom(target, Iter().slice(2, 4).all())
        [2, 3]

        This method accepts only positional arguments.
        """
        # TODO: make a kwarg-compatible version of this (islice takes no kwargs)
        # TODO: also support slice syntax Iter()[::]
        try:
            islice([], *args)
        except TypeError:
            raise TypeError(f'invalid slice arguments: {args!r}')
        return self._add_op('slice', args, lambda it, scope: islice(it, *args))

    def limit(self, count):
        """A convenient alias for :meth:`~Iter.slice`, which takes a single
        argument, *count*, the max number of items to yield.
        """
        return self._add_op('limit', (count,), lambda it, scope: islice(it, count))

    def takewhile(self, key=T):
        """Returns a new :class:`Iter()` spec which stops the stream once
        *key* becomes falsy.

        >>> glom([3, 2, 0, 1], Iter().takewhile().all())
        [3, 2]

        :func:`itertools.takewhile` for more details.
        """
        return self._add_op(
            'takewhile',
            (key,),
            lambda it, scope: takewhile(
                lambda t: scope[glom](t, key, scope), it))

    def dropwhile(self, key=T):
        """Returns a new :class:`Iter()` spec which drops stream items until
        *key* becomes falsy.

        >>> glom([0, 0, 3, 2, 0], Iter().dropwhile(lambda t: t < 1).all())
        [3, 2, 0]

        Note that while similar to :meth:`Iter.filter()`, the filter
        only applies to the beginning of the stream. In a way,
        :meth:`Iter.dropwhile` can be thought of as
        :meth:`~str.lstrip()` for streams. See
        :func:`itertools.dropwhile` for more details.

        """

        return self._add_op(
            'dropwhile',
            (key,),
            lambda it, scope: dropwhile(
                lambda t: scope[glom](t, key, scope), it))

    # Terminal methods follow

    def all(self):
        """A convenience method which returns a new spec which turns an
        iterable into a list.

        >>> glom(range(5), Iter(lambda t: t * 2).all())
        [0, 2, 4, 6, 8]

        Note that this spec will always consume the whole iterable, and as
        such, the spec returned is *not* an :class:`Iter()` instance.
        """
        return Pipe(self, list)

    def first(self, key=T, default=None):
        """A convenience method for lazily yielding a single truthy item from
        an iterable.

        >>> target = [False, 1, 2, 3]
        >>> glom(target, Iter().first())
        1

        This method takes a condition, *key*, which can also be a
        glomspec, as well as a *default*, in case nothing matches the
        condition.

        As this spec yields at most one item, and not an iterable, the
        spec returned from this method is not an :class:`Iter()` instance.
        """
        return (self, First(key=key, default=default))


class First:
    """Get the first element of an iterable which matches *key*, if there
    is one, otherwise return *default* (``None`` if unset).

    >>> is_odd = lambda x: x % 2
    >>> glom([0, 1, 2, 3], First(is_odd))
    1
    >>> glom([0, 2, 4], First(is_odd, default=False))
    False
    """
    # The impl of this ain't pretty and basically just exists for a
    # nicer-looking repr. (Iter(), First()) is the equivalent of doing
    # (Iter().filter(spec), Call(first, args=(T,), kwargs={'default':
    # default}))
    __slots__ = ('_spec', '_default', '_first')

    def __init__(self, key=T, default=None):
        self._spec = key
        self._default = default

        spec_glom = Spec(Call(partial, args=(Spec(self._spec).glom,), kwargs={'scope': S}))
        self._first = Call(first, args=(T,), kwargs={'default': default, 'key': spec_glom})

    def glomit(self, target, scope):
        return self._first.glomit(target, scope)

    def __repr__(self):
        cn = self.__class__.__name__
        if self._default is None:
            return f'{cn}({bbrepr(self._spec)})'
        return f'{cn}({bbrepr(self._spec)}, default={bbrepr(self._default)})'
