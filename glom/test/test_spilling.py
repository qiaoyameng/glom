import os
import glob
import tempfile

from pytest import raises

from glom import glom, T, SKIP, STOP, BadSpec, GlomError
from glom.grouping import Group, Min, Max, Avg
from glom.reduction import Count, Sum
from glom.spilling import SpillGroup


def _spill_files():
    return glob.glob(os.path.join(tempfile.gettempdir(), 'glom-spill-*'))


def test_spill_matches_group_builtin_aggs():
    # dozens of records, round-robin across more groups than the budget
    # allows, forcing constant spill/swap-in churn
    data = list(range(60))
    for leaf in (Count(), Sum(), Min(), Max()):
        spec = {T % 7: leaf}
        expected = glom(data, Group(spec))
        sg = SpillGroup(spec, max_active=2)
        result = glom(data, sg)
        assert result == expected
        assert list(result) == list(expected)  # first-appearance order
        # the shared budget was respected and spilling actually happened
        assert sg.max_active_seen <= 2
        assert sg.active_count <= 2
        assert sg.spill_count > 0
        assert sg.load_count > 0
    assert _spill_files() == []


def test_spill_nested_and_tuple_keys():
    data = [{'a': i % 3, 'b': i % 2, 'v': i * 1.5} for i in range(45)]
    spec = {T['a']: {lambda t: (t['a'], t['b']): Sum(T['v'])}}
    expected = glom(data, Group(spec))
    sg = SpillGroup(spec, max_active=1)
    result = glom(data, sg)
    assert result == expected
    assert list(result) == list(expected)
    for key in result:
        assert list(result[key]) == list(expected[key])
    assert sg.max_active_seen == 1
    assert _spill_files() == []


def test_spill_multi_keyspec_dict():
    data = list(range(30))
    spec = {T % 4: Count(), lambda t: 'total': Sum()}
    expected = glom(data, Group(spec))
    sg = SpillGroup(spec, max_active=2)
    assert glom(data, sg) == expected
    assert _spill_files() == []


def test_spill_bare_aggregator():
    assert glom(list(range(10)), SpillGroup(Sum(), max_active=1)) == 45
    assert glom([3, 1, 2], SpillGroup(Min())) == 1
    assert glom([], SpillGroup(Count())) is None
    assert _spill_files() == []


def test_spill_skip_leaves_no_counts():
    data = list(range(20))
    # SKIPped records must not affect the accumulator
    expected = glom(data, Group({T % 3: [lambda t: SKIP if t % 2 else t]}))
    expected = {k: len(v) for k, v in expected.items()}
    # same skip behavior via keyspec returning SKIP on odd values
    skip_spec = {lambda t: SKIP if t % 2 else t % 3: Count()}
    sg = SpillGroup(skip_spec, max_active=1)
    assert glom(data, sg) == expected
    assert _spill_files() == []


def test_spill_stop_cleans_up():
    def keyspec(t):
        if t >= 5:
            return STOP
        return t % 2

    sg = SpillGroup({keyspec: Count()}, max_active=1)
    result = glom(list(range(100)), sg)
    assert result == {0: 3, 1: 2}  # only records 0..4 counted
    assert sg._db is None
    assert _spill_files() == []


def test_spill_agg_failure_propagates_and_cleans_up():
    # the third record fails inside Sum's update, after group state exists
    data = [(0, 1), (1, 2), (0, 'not-a-number'), (1, 4)]
    sg = SpillGroup({T[0]: Sum(T[1])}, max_active=2)
    with raises(GlomError):
        glom(data, sg)
    assert sg._db is None
    # the failed record left no partial counts behind
    assert sg._active[(0, (0,))] == 1
    assert sg._active[(0, (1,))] == 2
    assert _spill_files() == []


def test_spill_failed_record_leaves_state_intact():
    class MutatingFail:
        """custom aggregator which mutates its state before raising"""
        def agg(self, target, tree):
            state = tree.get(self, [])
            state.append(target)
            tree[self] = state
            if target == 'boom':
                raise ValueError('boom')
            return state

        def encode_state(self, state):
            return list(state)

        def decode_state(self, data):
            return list(data)

    sg = SpillGroup({lambda t: 'g': MutatingFail()}, max_active=1)
    with raises(ValueError):
        glom(['ok', 'boom'], sg)
    # the failed record's mutation was rolled back, not counted
    assert sg._active[(0, ('g',))] == ['ok']
    assert _spill_files() == []


def test_spill_custom_aggregator_protocol():
    class AvgSpill:
        def agg(self, target, tree):
            acc = tree.get(self, [0.0, 0])
            acc[0] += target
            acc[1] += 1
            tree[self] = acc
            return acc

        def encode_state(self, state):
            return state

        def decode_state(self, data):
            return [float(data[0]), int(data[1])]

        def finalize_state(self, state):
            return state[0] / state[1]

    data = list(range(1, 41))
    expected = glom(data, Group({T % 5: Avg()}))
    sg = SpillGroup({T % 5: AvgSpill()}, max_active=2)
    assert glom(data, sg) == expected
    assert sg.spill_count > 0
    assert _spill_files() == []


def test_spill_rejects_aggregators_without_protocol():
    class Opaque:
        def agg(self, target, tree):
            return target

    with raises(BadSpec):
        SpillGroup({T: Opaque()})
    with raises(BadSpec):
        SpillGroup({T: [T]})  # list leaves unsupported
    with raises(BadSpec):
        SpillGroup({})
    with raises(BadSpec):
        SpillGroup('nope')
    with raises(ValueError):
        SpillGroup({T: Count()}, max_active=0)


def test_spill_rejects_unsupported_keys():
    for bad_key in ([1], {'a': 1}, {1, 2}, b'bytes', object()):
        sg = SpillGroup({lambda t: bad_key: Count()}, max_active=1)
        with raises(BadSpec):
            glom([1], sg)
    assert _spill_files() == []
    # tuple keys of supported scalars are fine
    sg = SpillGroup({lambda t: ('a', 1, 2.5, None, True): Count()}, max_active=1)
    assert glom([1, 2, 3], sg) == {('a', 1, 2.5, None, True): 3}


def test_spill_explicit_close_and_spill_path():
    path = os.path.join(tempfile.gettempdir(), 'glom-spill-explicit-test.sqlite')
    sg = SpillGroup({T % 3: Count()}, max_active=1, spill_path=path)
    assert glom(list(range(20)), sg) == {0: 7, 1: 7, 2: 6}
    assert not os.path.exists(path)  # cleaned up after the run
    sg.close()  # idempotent
    sg.close()

    sg2 = SpillGroup({T: Count()}, spill_path=path)
    sg2._begin()
    assert os.path.exists(path)
    sg2.close()
    assert not os.path.exists(path)

    with SpillGroup({T % 2: Sum()}, max_active=1) as sg3:
        assert glom(list(range(10)), sg3) == {0: 20, 1: 25}
    assert sg3._db is None
    assert _spill_files() == []


def test_spill_repr():
    spec = SpillGroup({T % 2: Count()}, max_active=4)
    assert repr(spec) == 'SpillGroup({T % 2: Count()}, max_active=4)'


def test_spill_spec_reuse():
    sg = SpillGroup({T % 3: Sum()}, max_active=1)
    first = glom(list(range(15)), sg)
    second = glom(list(range(15)), sg)
    assert first == second == glom(list(range(15)), Group({T % 3: Sum()}))
    assert _spill_files() == []
