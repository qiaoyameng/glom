import os
from collections import OrderedDict

import pytest

from glom import (
    glom, T, SKIP, STOP, Val,
    SpillGroup, SpillableAgg, SpillError, SpillCount, SpillSum, SpillMin, SpillMax,
)
from glom.grouping import Group, Min, Max, Avg, First, Sample
from glom.reduction import Count, Sum
from glom.spill import encode_state, restore_state


def _plain(spec):
    """map spill aggregators to their ordinary Group counterparts."""
    if isinstance(spec, SpillCount):
        return Count()
    if isinstance(spec, SpillSum):
        return Sum()
    if isinstance(spec, SpillMin):
        return Min()
    if isinstance(spec, SpillMax):
        return Max()
    if type(spec) is dict:
        return {k: _plain(v) for k, v in spec.items()}
    if type(spec) is list:
        return [_plain(v) for v in spec]
    return spec


def _native(value):
    if isinstance(value, OrderedDict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    return value


SPECS = [
    SpillCount(),
    SpillSum(),
    SpillMin(),
    SpillMax(),
    {T % 2: SpillCount()},
    {T % 3: SpillSum()},
    {T % 2: SpillMin()},
    {T % 2: SpillMax()},
    {T % 3: {T % 2: SpillCount()}},
    {T % 3: {T % 2: SpillSum()}},
    {T % 3: {T % 2: SpillMin()}},
    {T % 3: {T % 2: SpillMax()}},
    [SpillCount(), SpillSum()],
    {T % 2: [SpillCount(), SpillSum(), SpillMin(), SpillMax()]},
    {T % 4: {T % 2: [SpillCount(), SpillMax()]}},
]


@pytest.mark.parametrize('spec', SPECS)
@pytest.mark.parametrize('budget', [1, 2, 3, 5, 100])
def test_parity_with_group(spec, budget):
    # tiny data plus a tiny budget forces repeated spill/load cycles
    data = [n % 7 for n in range(48)]
    expected = glom(data, Group(_plain(spec)))
    got = glom(data, SpillGroup(spec, budget=budget))
    assert _native(got) == _native(expected)


def test_first_seen_order_preserved():
    keys = [5, 2, 7, 2, 5, 7, 1]
    result = glom(keys, SpillGroup({T: SpillCount()}, budget=1))
    assert list(result.keys()) == [5, 2, 7, 1]
    assert _native(result) == {5: 2, 2: 2, 7: 2, 1: 1}



# ---------- budget and spill statistics --------------------------------

def test_budget_bounds_active_groups():
    for budget in (1, 2, 3):
        sg = SpillGroup({T % 5: SpillCount()}, budget=budget)
        glom(range(100), sg)
        assert sg.peak_active_groups <= budget
        assert sg.spill_count > 0 and sg.load_count > 0


def test_large_budget_never_spills():
    sg = SpillGroup({T % 5: SpillCount()}, budget=64)
    glom(range(100), sg)
    assert sg.spill_count == 0
    assert sg.peak_active_groups == 5


def test_stats_survive_close():
    sg = SpillGroup({T % 4: SpillCount()}, budget=1)
    glom(range(40), sg)
    # close() already ran inside glom(); stats remain readable
    assert sg.spill_count > 0
    assert sg.peak_active_groups == 1
    assert sg.active_group_count == 0


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        SpillGroup({T: SpillCount()}, budget=0)


# ---------- staging cleanup ---------------------------------------------

def _spill_glob():
    import glob
    import tempfile
    return glob.glob(os.path.join(tempfile.gettempdir(), 'glom-spill-*'))


def test_normal_run_cleans_staging():
    before = set(_spill_glob())
    glom(range(30), SpillGroup({T % 3: SpillCount()}, budget=1))
    assert set(_spill_glob()) <= before


def test_exception_cleans_staging():
    before = set(_spill_glob())

    def explode(target):
        if target == 5:
            raise ValueError('boom')
        return target % 3

    with pytest.raises(Exception):
        glom(range(30), SpillGroup({explode: SpillCount()}, budget=1))
    assert set(_spill_glob()) <= before


def test_early_stop_cleans_staging():
    before = set(_spill_glob())
    glom(range(30), SpillGroup({T: lambda t: STOP if t > 5 else t}, budget=1))
    assert set(_spill_glob()) <= before


def test_explicit_close_removes_database(tmp_path):
    path = str(tmp_path / 'stage.db')
    sg = SpillGroup({T % 2: SpillCount()}, budget=1, spill_path=path)
    sg._open()
    assert os.path.exists(path)
    sg.close()
    assert not os.path.exists(path)
    assert not os.path.exists(path + '-wal')
    assert not os.path.exists(path + '-shm')


def test_context_manager_closes(tmp_path):
    path = str(tmp_path / 'stage.db')
    with SpillGroup({T: SpillCount()}, budget=1, spill_path=path) as sg:
        sg._store  # opened
        assert os.path.exists(path)
    assert not os.path.exists(path)


# ---------- SKIP / STOP semantics ---------------------------------------

def test_skip_key_matches_group():
    spec = {lambda t: SKIP if t < 3 else t: SpillCount()}
    assert _native(glom(range(6), SpillGroup(spec, budget=1))) == {3: 1, 4: 1, 5: 1}


def test_skip_value_matches_group():
    spec = {T: lambda t: t if t % 2 else SKIP}
    got = _native(glom(range(6), SpillGroup(spec, budget=1)))
    expected = _native(glom(range(6), Group({T: lambda t: t if t % 2 else SKIP})))
    assert got == expected


def test_immediate_stop_returns_empty():
    assert glom(range(5), SpillGroup({Val(STOP): SpillCount()}, budget=1)) == OrderedDict()


def test_stop_value_matches_group():
    spec = {T: lambda t: STOP if t > 2 else t}
    got = _native(glom(range(6), SpillGroup(spec, budget=1)))
    expected = _native(glom(range(6), Group(spec)))
    assert got == expected


# ---------- key restrictions --------------------------------------------

def test_scalar_and_tuple_keys_accepted():
    rows = [('a', 1), ('b', 2), ('a', 3), ('b', 4)]
    result = glom(rows, SpillGroup(
        {lambda t: (t[0], t[1] % 2): SpillCount()}, budget=1))
    assert _native(result) == {('a', 1): 2, ('b', 0): 2}


def test_unsupported_scalar_key_rejected():
    with pytest.raises(SpillError):
        glom([{'x': 1}], SpillGroup({T: SpillCount()}, budget=1))


def test_unsupported_tuple_component_rejected():
    with pytest.raises(SpillError):
        glom([{'x': 1}],
             SpillGroup({lambda t: (t, t): SpillCount()}, budget=1))


# ---------- custom aggregator protocol ----------------------------------

class _Avg(SpillableAgg):
    state_type = 'test_avg'

    def agg(self, target, tree):
        acc = tree.setdefault(self, [0.0, 0])
        acc[0] += target
        acc[1] += 1
        return acc[0] / acc[1]

    def encode_state(self, tree):
        return list(tree.get(self, [0.0, 0]))

    def restore_state(self, payload, tree):
        tree[self] = list(payload)

    def finalize(self, tree):
        total, count = tree.get(self, [0.0, 0])
        return total / count if count else None


def test_custom_aggregator_round_trips():
    data = list(range(21))
    got = _native(glom(data, SpillGroup({T % 3: _Avg()}, budget=1)))
    expected = {k: sum(data[k::3]) / len(data[k::3]) for k in range(3)}
    assert got == expected


class _NoProtocol:
    def agg(self, target, tree):
        return target


def test_aggregator_without_protocol_rejected():
    with pytest.raises(SpillError):
        glom([1, 2, 3], SpillGroup(_NoProtocol(), budget=1))


@pytest.mark.parametrize('built_in', [Avg(), First(), Sample(2)])
def test_unsupported_builtin_aggregators_rejected(built_in):
    with pytest.raises(SpillError):
        glom([1, 2, 3], SpillGroup(built_in, budget=1))


# ---------- explicit state encoding (no pickle) --------------------------

def test_state_protocol_is_json_friendly():
    ks, mn = T % 2, SpillMin()
    spec = {ks: mn}
    tree = {
        id(spec): OrderedDict([(0, {mn: 4}), (1, {mn: 1})]),
        '__key_index__': {0: id(ks), 1: id(ks)},
        '__keyspecs__': {id(ks): (ks, mn)},
        '__emitted__': {0, 1},
        '__stopped__': set(),
    }
    state = encode_state(spec, tree)
    import json
    text = json.dumps(state)  # must serialize without custom encoders
    restored = {}
    restore_state(spec, json.loads(text), restored)
    assert encode_state(spec, restored) == state


def test_empty_input_matches_group():
    assert _native(glom([], SpillGroup({T: SpillCount()}, budget=1))) == {}
    assert glom([], SpillGroup(SpillCount())) is None
    assert glom([], SpillGroup([SpillCount()], budget=1)) == []


def test_all_skip_matches_group():
    spec = {lambda t: SKIP: SpillCount()}
    plain = {lambda t: SKIP: Count()}
    got = _native(glom(range(3), SpillGroup(spec, budget=1)))
    expected = _native(glom(range(3), Group(plain)))
    assert got == expected


def test_deep_nesting_and_high_churn():
    data = [n % 7 for n in range(48)]
    spec = {T % 3: {T % 2: [SpillCount(), SpillSum()]}}
    plain = {T % 3: {T % 2: [Count(), Sum()]}}
    for budget in (1, 2, 4, 50):
        got = _native(glom(data, SpillGroup(spec, budget=budget)))
        expected = _native(glom(data, Group(plain)))
        assert got == expected


def test_high_churn_tiny_budget():
    data = [n % 50 for n in range(500)]
    spec = {T % 50: [SpillCount(), SpillSum(), SpillMin(), SpillMax()]}
    plain = {T % 50: [Count(), Sum(), Min(), Max()]}
    sg = SpillGroup(spec, budget=2)
    assert _native(glom(data, sg)) == _native(glom(data, Group(plain)))
    assert sg.spill_count > 100
    assert sg.peak_active_groups <= 2
