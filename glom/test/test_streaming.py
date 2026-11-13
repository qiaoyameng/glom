import pytest

from itertools import count, dropwhile, chain

from glom import Iter, IterFailure
from glom import glom, SKIP, STOP, T, Call, Spec, Glommer, Check, Path, GlomError


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


class SourceError(Exception):
    pass


class OneShotSource:
    def __init__(self, records, error):
        self.records = records
        self.error = error
        self.index = 0
        self.closed = False
        self.iterated = False

    def __iter__(self):
        if self.iterated:
            raise RuntimeError('source was iterated twice')
        self.iterated = True
        return self

    def __next__(self):
        if self.index >= len(self.records):
            raise self.error
        value = self.records[self.index]
        self.index += 1
        return value

    def close(self):
        self.closed = True


def test_attempt_failure_then_source_error():
    source = OneShotSource(['1', 'bad', '3'], SourceError('source failed'))
    out = glom(source, Iter().attempt(int))

    assert next(out) == 1

    failure = next(out)
    assert isinstance(failure, IterFailure)
    assert failure.index == 1
    assert failure.path == Path(1)
    assert failure.kind == 'error'
    assert failure.error is not None
    assert isinstance(failure.error, GlomError)
    assert failure.error_chain
    assert isinstance(failure.root_error, ValueError)

    # The failure did not stop the stream or consume the next record.
    assert next(out) == 3

    # A failure from the source itself terminates iteration. It is not
    # converted into another per-record failure item.
    with pytest.raises(SourceError):
        next(out)
    assert source.closed


def test_attempt_missing_skip_and_stop():
    target = [{'id': 1}, {}, {'id': 3}]
    out = list(glom(target, Iter().attempt(T['id'])))

    assert out[0] == 1
    assert isinstance(out[1], IterFailure)
    assert out[1].kind == 'missing'
    assert out[1].path == Path(1, 'id')
    assert out[1].spec_path == Path('id')
    assert out[2] == 3

    # SKIP still filters and STOP still terminates in attempt mode.
    assert list(glom([1, SKIP, 2, STOP, 3], Iter(attempt=True))) == [1, 2]


def test_attempt_callback_runs_once_per_record():
    calls = []

    def parse(value):
        calls.append(value)
        return int(value)

    out = glom(['1', 'bad', '3'], Iter().attempt(parse))
    assert next(out) == 1
    assert calls == ['1']
    assert isinstance(next(out), IterFailure)
    assert calls == ['1', 'bad']
    assert next(out) == 3
    assert calls == ['1', 'bad', '3']


def test_attempt_original_budget_and_early_close():
    out = list(glom(['bad'], Iter().attempt(
        int, keep_original=True, max_original_bytes=0).all()))
    assert out[0].original is None or not out[0].has_original
    assert out[0].original_omitted

    source = OneShotSource(['1', '2'], SourceError('finished'))
    out = glom(source, Iter().attempt(int))
    assert next(out) == 1
    out.close()
    assert source.closed
    with pytest.raises(StopIteration):
        next(out)
