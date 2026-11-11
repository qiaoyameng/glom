"""glom's helpers for streaming use cases.

Specifier types which yield their results incrementally so that they
can be applied to targets which are themselves streaming (e.g. chunks
of rows from a database, lines from a file) without excessive memory
usage.

glom's streaming functionality revolves around a single :class:`Iter`
Specifier type, which has methods to transform the target stream.
"""

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
from boltons.typeutils import make_sentinel

from .core import (glom, T, STOP, SKIP, _MISSING, Path, TargetRegistry, Call,
                   Spec, Pipe, S, GlomError, PathAccessError, bbrepr,
                   format_invocation)
from .matching import Check


# outcome tags for :meth:`Iter.attempt()`; see :class:`AttemptFailure`
ATTEMPT_MISSING = make_sentinel('ATTEMPT_MISSING')
ATTEMPT_SKIP = make_sentinel('ATTEMPT_SKIP')
ATTEMPT_STOP = make_sentinel('ATTEMPT_STOP')
ATTEMPT_ERROR = make_sentinel('ATTEMPT_ERROR')

_DEFAULT_ATTEMPT_VALUE_BUDGET = 10

class AttemptSuccess:
    """The success outcome yielded by :meth:`Iter.attempt()`.

    Attributes:
       index (int): Zero-based position of the source element in the
         iterable being processed.
       value: The result of applying the spec to the source element.

    >>> glom(['1', '2'], Iter(int).attempt().all())
    [AttemptSuccess(index=0, value=1), AttemptSuccess(index=1, value=2)]
    """
    __slots__ = ('index', 'value')

    def __init__(self, index, value):
        self.index = index
        self.value = value

    @property
    def ok(self):
        """Always ``True`` for :class:`AttemptSuccess`."""
        return True

    def __repr__(self):
        return '%s(index=%r, value=%r)' % (self.__class__.__name__,
                                           self.index, self.value)

    def __eq__(self, other):
        if type(other) is not AttemptSuccess:
            return NotImplemented
        return self.index == other.index and self.value == other.value

    def __hash__(self):
        return hash((self.__class__, self.index))


class AttemptFailure:
    """The failure outcome yielded by :meth:`Iter.attempt()`, produced
    whenever a single source element does not yield a normal value.

    Attributes:
       index (int): Zero-based position of the source element in the
         iterable being processed.
       kind: One of the outcome sentinels describing why no value was
         produced:

         * :data:`ATTEMPT_MISSING` -- the spec raised a
           :class:`~glom.PathAccessError` while looking up a value
         * :data:`ATTEMPT_SKIP` -- the spec evaluated to
           :data:`~glom.SKIP`
         * :data:`ATTEMPT_STOP` -- the spec evaluated to
           :data:`~glom.STOP` (or the configured sentinel)
         * :data:`ATTEMPT_ERROR` -- the spec raised any other error
       path: A copy of the glom target path at the point of failure,
         including the source element's index.
       error: The original exception raised while applying the spec,
         kept intact with its full ``GlomError`` chain (including any
         wrapped exception on its ``__wrapped`` attribute). ``None``
         for the non-exceptional SKIP and STOP outcomes.
       value: The original source element, or the :data:`~glom._MISSING`
         sentinel when value retention is disabled or the configured
         value budget has been spent.

    Note that :class:`AttemptFailure` never swallows errors from the
    source iterator itself: if fetching the next source element raises,
    that error propagates out of the iteration instead.
    """
    __slots__ = ('index', 'kind', 'path', 'error', 'value')

    # expose the outcome sentinels as class attributes for convenience
    MISSING = ATTEMPT_MISSING
    SKIP = ATTEMPT_SKIP
    STOP = ATTEMPT_STOP
    ERROR = ATTEMPT_ERROR

    def __init__(self, index, kind, path, error, value=_MISSING):
        self.index = index
        self.kind = kind
        self.path = path
        self.error = error
        self.value = value

    @property
    def ok(self):
        """Always ``False`` for :class:`AttemptFailure`."""
        return False

    def __repr__(self):
        cn = self.__class__.__name__
        parts = ['index=%r' % self.index, 'kind=%r' % self.kind]
        if self.error is not None:
            parts.append('error=%r' % self.error)
        if self.value is not _MISSING:
            parts.append('value=%r' % self.value)
        return '%s(%s)' % (cn, ', '.join(parts))

    def __eq__(self, other):
        if type(other) is not AttemptFailure:
            return NotImplemented
        return (self.index == other.index and self.kind is other.kind
                and self.error == other.error and self.value == other.value)

    def __hash__(self):
        return hash((self.__class__, self.index, self.kind))


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

    """
    def __init__(self, subspec=T, **kwargs):
        self.subspec = subspec
        self._iter_stack = kwargs.pop('_iter_stack', [])

        self.sentinel = kwargs.pop('sentinel', STOP)
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
        return ''.join(chunks)

    def glomit(self, target, scope):
        iterator = self._iterate(target, scope)

        for _, _, callback in reversed(self._iter_stack):
            iterator = callback(iterator, scope)

        return iter(iterator)

    def _iterate(self, target, scope):
        iterate = scope[TargetRegistry].get_handler('iterate', target, path=scope[Path])
        try:
            iterator = iterate(target)
        except Exception as e:
            raise TypeError('failed to iterate on instance of type %r at %r (got %r)'
                            % (target.__class__.__name__, Path(*scope[Path]), e))

        base_path = scope[Path]
        for i, t in enumerate(iterator):
            scope[Path] = base_path + [i]
            yld = (t if self.subspec is T else scope[glom](t, self.subspec, scope))
            if yld is SKIP:
                continue
            elif yld is self.sentinel or yld is STOP:
                # NB: sentinel defaults to STOP so I was torn whether
                # to also check for STOP, and landed on the side of
                # never letting STOP through.
                return
            yield yld
        return

    def _add_op(self, opname, args, callback):
        return type(self)(subspec=self.subspec, _iter_stack=[(opname, args, callback)] + self._iter_stack)

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

    def attempt(self, subspec=T, keep_values=False, value_budget=_DEFAULT_ATTEMPT_VALUE_BUDGET):
        """Return a new :class:`Iter()` spec which evaluates *subspec*
        once per source element and always yields exactly one outcome
        object, instead of letting the first bad element abort the
        whole batch.

        Successful elements come out as :class:`AttemptSuccess`
        instances carrying the source index and the spec result;
        anything else comes out as :class:`AttemptFailure`, tagged
        with a ``kind`` sentinel:

        * :data:`ATTEMPT_MISSING` -- a :class:`~glom.PathAccessError`
          from the spec,
        * :data:`ATTEMPT_ERROR` -- any other exception from the spec,
          preserved on the failure as the original error with its full
          ``GlomError`` chain,
        * :data:`ATTEMPT_SKIP` and :data:`ATTEMPT_STOP` -- when the
          spec returns the :data:`~glom.SKIP` or :data:`~glom.STOP`
          sentinels, respectively. A STOP outcome is terminal, just
          like plain :class:`Iter()` behavior.

        >>> target = [{'a': 1}, {}, {'a': 3}]
        >>> out = glom(target, Iter().attempt(T['a']).all())
        >>> [(r.index, r.ok, r.value if r.ok else None) for r in out]
        [(0, True, 1), (1, False, None), (2, True, 3)]
        >>> out[1].kind is AttemptFailure.MISSING
        True
        >>> out[1].error  # doctest: +ELLIPSIS
        PathAccessError(KeyError('a'), T['a'], 0)
        >>> out[1].path  # doctest: +ELLIPSIS
        [Iter().attempt(T['a']), 1]
        >>> from glom.core import _MISSING
        >>> out[1].value is _MISSING
        True

        Only errors raised while applying *subspec* to an element are
        captured. Specs applied by earlier stages of the pipeline
        (e.g. the :class:`Iter()` subspec or an earlier
        :meth:`map()`) are outside this protection, and if the source
        iterator itself raises while fetching the next element, that
        error propagates and terminates the stream -- a broken source
        is not another bad record:

        >>> def source():
        ...     yield {'a': 1}
        ...     raise RuntimeError('source exploded')
        >>> out = glom(source(), Iter().attempt(T['a']))
        >>> next(out)
        AttemptSuccess(index=0, value=1)
        >>> next(out)
        Traceback (most recent call last):
        ...
        RuntimeError: source exploded

        Args:
           subspec: A subspec to be applied on each element from the
             iterable. Defaults to :data:`~glom.T`, i.e. the element
             itself.
           keep_values (bool): When true, failures retain a reference
             to the original source element on
             :attr:`AttemptFailure.value`. When false (the default),
             the attribute holds the :data:`~glom._MISSING` sentinel,
             so failing streams do not pin data in memory.
           value_budget (int): Upper bound on how many failure values
             are retained when *keep_values* is true, so a stream of
             mostly-bad records cannot accumulate without limit. Once
             the budget is spent, later failures omit their value.
        """
        if value_budget < 0:
            raise ValueError('expected value_budget >= 0, got: %r' % value_budget)
        args = (subspec,)
        if keep_values or value_budget != _DEFAULT_ATTEMPT_VALUE_BUDGET:
            # NB: with >1 args, __repr__ renders them all as kwargs
            args = (subspec, keep_values, value_budget)
        return self._add_op(
            'attempt', args,
            lambda iterable, scope: _attempt_iter(
                iterable, scope, subspec, keep_values, value_budget,
                self.sentinel))

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


def _attempt_iter(iterable, scope, subspec, keep_values, value_budget,
                  sentinel=STOP):
    """Generator behind :meth:`Iter.attempt()`.

    Wraps only the per-element spec evaluation in try/except; pulling
    the next element from *iterable* happens outside the handler so a
    source-iterator error propagates to the caller. When the consumer
    closes the generator early, ``finally`` closes the wrapped
    iterator instead of converting its shutdown signals into results.
    """
    iterator = iter(iterable)
    retained = 0
    base_path = scope[Path]
    index = 0
    try:
        while True:
            # outside the try: source failures must terminate, not
            # get reported as another bad record
            try:
                element = next(iterator)
            except StopIteration:
                return

            scope[Path] = base_path + [index]
            try:
                result = (element if subspec is T
                          else scope[glom](element, subspec, scope))
            except PathAccessError as error:
                yield _make_failure(index, ATTEMPT_MISSING, scope, error,
                                    element, keep_values, retained, value_budget)
                retained += 1
            except GlomError as error:
                yield _make_failure(index, ATTEMPT_ERROR, scope, error,
                                    element, keep_values, retained, value_budget)
                retained += 1
            except Exception as error:
                # errors raised by plain callables are not wrapped
                # until the top-level glom() call, so catch them here
                # to keep one bad element from aborting the stream
                yield _make_failure(index, ATTEMPT_ERROR, scope, error,
                                    element, keep_values, retained, value_budget)
                retained += 1
            else:
                if result is SKIP:
                    yield _make_failure(index, ATTEMPT_SKIP, scope, None,
                                        element, keep_values, retained,
                                        value_budget)
                    retained += 1
                elif result is STOP or result is sentinel:
                    # like plain Iter(), STOP is never let through
                    yield _make_failure(index, ATTEMPT_STOP, scope, None,
                                        element, keep_values, retained,
                                        value_budget)
                    retained += 1
                    return
                else:
                    yield AttemptSuccess(index, result)
            index += 1
    finally:
        # release the wrapped iterator's resources on early close;
        # do not swallow any error it raises while shutting down
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
        scope[Path] = base_path


def _make_failure(index, kind, scope, error, element,
                  keep_values, retained, value_budget):
    value = element if (keep_values and retained < value_budget) else _MISSING
    return AttemptFailure(index, kind, list(scope[Path]), error, value)


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
