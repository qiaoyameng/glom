import pytest

from itertools import count, dropwhile, chain

from glom import Iter
from glom import glom, SKIP, STOP, T, Call, Spec, Glommer, Check, SKIP, Path
from glom import (AttemptSuccess, AttemptFailure,
                   ATTEMPT_MISSING, ATTEMPT_SKIP, ATTEMPT_STOP, ATTEMPT_ERROR,
                   GlomError, PathAccessError)
from glom.core import _MISSING
from glom.streaming import _attempt_iter


RANGE_5 = list(range(5))


def test_iter():
    assert list(glom(['1', '2', '3'], Iter(int))) == [1, 2, 3]
    cnt = count()
    cnt_1 = glom(cnt, Iter(lambda t: t + 1))
    assert (next(cnt_1), next(cnt_1)) == (1, 2)
    assert next(cnt) == 2

    assert list(glom(['1', '2', '3'], (Iter(int), enumerate))) == [(0, 1), (1, 2), (2, 3)]

    assert list(glom([1, SKIP, 2], Iter())) == [1, 2]
    assert list(glom([1, STOP, 2], Iter())) == [1]

    with pytest.raises(TypeError):
        Iter(nonexistent_kwarg=True)


def test_filter():
    is_odd = lambda x: x % 2
    odd_spec = Iter().filter(is_odd)
    out = glom(RANGE_5, odd_spec)
    assert list(out) == [1, 3]

    # let's just make sure we're actually streaming just in case
    counter = count()
    out = glom(counter, odd_spec)
    assert next(out) == 1
    assert next(out) == 3
    assert next(counter) == 4
    assert next(counter) == 5
    assert next(out) == 7

    bools = [True, False, False, True, False]
    spec = Iter().filter().all()
    out = glom(bools, spec)
    assert out == [True, True]

    imags = [0j, 1j, 2, 2j, 3j]
    spec = Iter().filter(Check(T.imag.real, type=float, one_of=(0, 2), default=SKIP)).all()
    out = glom(imags, spec)
    assert out == [0j, 2j]

    assert repr(Iter().filter(T.a.b)) == 'Iter().filter(T.a.b)'
    assert repr(Iter(list).filter(sum)) == 'Iter(list).filter(sum)'


def test_map():
    spec = Iter().map(lambda x: x * 2)
    out = glom(RANGE_5, spec)
    assert list(out) == [0, 2, 4, 6, 8]
    assert repr(Iter().map(T.a.b)) == 'Iter().map(T.a.b)'


def test_split_flatten():
    falsey_stream = [1, None, None, 2, 3, None, 4]
    spec = Iter().split()
    out = glom(falsey_stream, spec)
    assert list(out) == [[1], [2, 3], [4]]

    spec = Iter().split().flatten()
    out = glom(falsey_stream, spec)
    assert list(out) == [1, 2, 3, 4]

    assert repr(Iter().split(sep=None, maxsplit=2)) == 'Iter().split(sep=None, maxsplit=2)'
    assert repr(Iter(T.a.b[1]).flatten()) == 'Iter(T.a.b[1]).flatten()'


def test_chunked():
    int_list = list(range(9))

    spec = Iter().chunked(3)
    out = glom(int_list, spec)
    assert list(out) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    spec = Iter().chunked(3).map(sum)
    out = glom(int_list, spec)
    assert list(out) == [3, 12, 21]


def test_windowed():
    int_list = list(range(5))

    spec = Iter().windowed(3)
    out = glom(int_list, spec)
    assert list(out) == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert repr(spec) == 'Iter().windowed(3)'

    spec = spec.filter(lambda x: bool(x[0] % 2)).map(sum)
    out = glom(int_list, spec)
    assert next(out) == 6

    out = glom(range(10), spec)
    assert list(out) == [6, 12, 18, 24]


def test_unique():
    int_list = list(range(10))

    spec = Iter().unique()
    out = glom(int_list, spec)
    assert list(out) == int_list

    spec = Iter(lambda x: x % 4).unique()
    out = glom(int_list, spec)
    assert list(out) == int_list[:4]
    assert repr(Iter().unique(T.a)) == 'Iter().unique(T.a)'


def test_slice():
    cnt = count()

    spec = Iter().slice(3)
    out = glom(cnt, spec)

    assert list(out) == [0, 1, 2]
    assert next(cnt) == 3

    out = glom(range(10), Iter().slice(1, 5))
    assert list(out) == [1, 2, 3, 4]

    out = glom(range(10), Iter().slice(1, 6, 2))
    assert list(out) == [1, 3, 5]
    assert repr(Iter().slice(1, 6, 2)) == 'Iter().slice(1, 6, 2)'

    out = glom(range(10), Iter().limit(3))
    assert list(out) == [0, 1, 2]
    assert repr(Iter().limit(3)) == 'Iter().limit(3)'

    out = glom(range(5), Iter().limit(10))
    assert list(out) == [0, 1, 2, 3, 4]

    # test broken args
    with pytest.raises(TypeError):
        Iter().slice(1, 2, 3, 4)


def test_while():
    cnt = count()
    out = glom(cnt, Iter().takewhile(lambda x: x < 3))
    assert list(out) == [0, 1, 2]
    assert next(cnt) == 4
    assert repr(Iter().takewhile(T.a) == 'Iter().takewhile(T.a)')

    range_iter = iter(range(7))
    out = glom(range_iter, Iter().dropwhile(lambda x: x < 3 or x > 5))
    assert list(out) == [3, 4, 5, 6]  # 6 still here despite the x>5 above

    out = glom(range(10), Iter().dropwhile(lambda x: x >= 0).limit(10))
    assert list(out) == []

    out = glom(range(8), Iter().dropwhile((T.bit_length(), lambda x: x < 3)))
    assert list(out) == [4, 5, 6, 7]
    assert repr(Iter().dropwhile(T.a) == 'Iter().dropwhile(T.a)')


def test_iter_composition():
    int_list = list(range(10))
    out = glom(int_list, (Iter(), Iter(), list))
    assert out == int_list

    out = glom([int_list] * 3, Iter(Iter(lambda x: x % 4)).flatten().unique())
    assert list(out) == [0, 1, 2, 3]


def test_faulty_iterate():
    glommer = Glommer()

    def bad_iter(obj):
        raise RuntimeError('oops')

    glommer.register(str, iterate=bad_iter)

    with pytest.raises(TypeError):
        glommer.glom('abc', (Iter(), list))


def test_first():
    spec = Iter().first(T.imag)

    target = iter([1, 2, 3j, 4])
    out = glom(target, spec)
    assert out == 3j
    assert next(target) == 4
    assert repr(spec) == '(Iter(), First(T.imag))'

    spec = Iter().first(T.imag, default=0)
    target = iter([1, 2, 4])
    out = glom(target, spec)
    assert out == 0
    assert repr(spec) == '(Iter(), First(T.imag, default=0))'


def test_all():
    int_iter = iter(range(10))

    out = glom(int_iter, Iter().all())
    assert out == list(range(10))
    assert next(int_iter, None) is None
    assert repr(Iter().all()) == 'Pipe(Iter(), list)'


def test_attempt_basic():
    target = [{'a': 1}, {}, {'a': 3}]
    out = glom(target, Iter().attempt(T['a']).all())

    assert [r.ok for r in out] == [True, False, True]
    assert [r.index for r in out] == [0, 1, 2]
    assert out[0] == AttemptSuccess(index=0, value=1)
    assert out[2].value == 3

    failure = out[1]
    assert isinstance(failure, AttemptFailure)
    assert failure.kind is ATTEMPT_MISSING
    assert failure.kind is AttemptFailure.MISSING
    # source index and spec location are both recoverable
    assert failure.index == 1
    assert failure.path[-1] == 1
    # original GlomError chain is preserved, not re-wrapped
    assert isinstance(failure.error, PathAccessError)
    assert isinstance(failure.error, GlomError)
    assert isinstance(failure.error.exc, KeyError)
    assert failure.error.path == Path(T['a'])


def test_attempt_error_kind():
    # non-GlomError exceptions from plain callables are captured, too
    out = list(glom(['1', 'x', '3'], Iter().attempt(int)))
    assert [r.ok for r in out] == [True, False, True]
    assert out[1].kind is ATTEMPT_ERROR
    assert isinstance(out[1].error, ValueError)

    # a GlomError raised by the spec keeps its type
    out = list(glom([{'a': 1}, {'a': 2}], Iter().attempt(T['b'])))
    assert all(r.kind is ATTEMPT_MISSING for r in out)


def test_attempt_skip_and_stop():
    # SKIP/STOP returned by the subspec get their own failure kinds
    spec = Iter().attempt(lambda t: SKIP if t == 1 else (STOP if t == 3 else t * 10))
    out = list(glom([0, 1, 2, 3, 4], spec))

    assert [r.ok for r in out] == [True, False, True, False]
    assert out[1].kind is ATTEMPT_SKIP
    assert out[1].error is None
    assert out[3].kind is ATTEMPT_STOP
    assert out[3].error is None
    # STOP is terminal: element 4 was never processed
    assert len(out) == 4

    # SKIP/STOP source elements are still handled by Iter itself
    assert list(glom([1, SKIP, 2], Iter().attempt())) == [
        AttemptSuccess(0, 1), AttemptSuccess(1, 2)]
    assert list(glom([1, STOP, 2], Iter().attempt())) == [AttemptSuccess(0, 1)]

    # a custom Iter sentinel is also a terminal STOP outcome
    spec = Iter(sentinel='END').attempt(lambda t: 'END' if t == 2 else t)
    out = list(glom([0, 1, 2, 3], spec))
    assert [r.ok for r in out] == [True, True, False]
    assert out[2].kind is ATTEMPT_STOP


def test_attempt_keep_values_and_budget():
    # by default, failures do not retain the original value
    out = list(glom(['x', 'y'], Iter().attempt(int)))
    assert all(r.value is _MISSING for r in out)

    # keep_values retains values up to the budget, then stops
    spec = Iter().attempt(int, keep_values=True, value_budget=2)
    out = list(glom(['1', 'x', 'y', 'z', '5'], spec))
    failures = [r for r in out if not r.ok]
    assert [f.value for f in failures] == ['x', 'y', _MISSING]
    assert [r.value for r in out if r.ok] == [1, 5]

    with pytest.raises(ValueError):
        Iter().attempt(int, value_budget=-1)


def test_attempt_callback_once_per_element():
    calls = []

    def boom(v):
        calls.append(v)
        if v == 2:
            raise ValueError('nope')
        return v

    out = list(glom([1, 2, 3], Iter().attempt(boom)))
    assert [r.ok for r in out] == [True, False, True]
    # exactly one invocation per element, success or failure
    assert calls == [1, 2, 3]


def test_attempt_source_error_terminates():
    # one-shot source: one good record, one bad record, then the
    # source itself raises. The source error must terminate the
    # stream, not surface as another failure item.
    def source():
        yield {'a': 1}
        yield {'b': 2}
        raise RuntimeError('source exploded')

    out = glom(source(), Iter().attempt(T['a']))

    good = next(out)
    assert good == AttemptSuccess(index=0, value=1)

    bad = next(out)
    assert isinstance(bad, AttemptFailure)
    assert bad.kind is ATTEMPT_MISSING
    assert bad.index == 1
    assert bad.path[-1] == 1
    assert isinstance(bad.error, PathAccessError)

    with pytest.raises(RuntimeError, match='source exploded'):
        next(out)


def test_attempt_close_cleans_up():
    # closing the consumer early closes the internal iterator
    closed = []

    class Closeable(object):
        def __iter__(self):
            return self

        def __next__(self):
            return 1

        def close(self):
            closed.append(True)

    scope = {Path: [], glom: None}
    stream = _attempt_iter(Closeable(), scope, T, False, 0)
    next(stream)
    stream.close()
    assert closed == [True]

    # a close() that raises propagates; it is not converted to None
    class BadClose(object):
        def __iter__(self):
            return self

        def __next__(self):
            return 1

        def close(self):
            raise RuntimeError('close boom')

    stream = _attempt_iter(BadClose(), {Path: [], glom: None}, T, False, 0)
    next(stream)
    with pytest.raises(RuntimeError, match='close boom'):
        stream.close()


def test_attempt_composition():
    # outcomes flow through downstream Iter ops for filtering,
    # batching, and summarizing
    out = list(glom([{'a': 1}, {}, {'a': 3}],
                    Iter().attempt(T['a']).filter(lambda r: r.ok).map(lambda r: r.value)))
    assert out == [1, 3]

    out = list(glom([1, 'x', 2, 'y'], Iter().attempt(int).chunked(2)))
    assert [[r.ok for r in chunk] for chunk in out] == [[True, False], [True, False]]

    summary = {'ok': 0, 'failed': 0}
    for r in glom([1, 'x', 2, 'y'], Iter().attempt(int)):
        summary['ok' if r.ok else 'failed'] += 1
    assert summary == {'ok': 2, 'failed': 2}

    assert repr(Iter().attempt(int)) == 'Iter().attempt(int)'
    assert (repr(Iter().attempt(int, keep_values=True, value_budget=2))
            == 'Iter().attempt(subspec=int, keep_values=True, value_budget=2)')
