import copy

import pytest

from glom import (BidiSpec,
                  Coalesce,
                  DuplicateSourceError,
                  NotInvertibleError,
                  ParentChildConflictError,
                  Path,
                  PathAccessError,
                  SequenceConflictError,
                  T,
                  UnmappedFieldError,
                  glom)


def test_nested_rearrange_and_dotted_key_roundtrip():
    target = {
        'user': {'name': 'Ada', 'address': {'city': 'London'}},
        'payload': {'a.b': 3},
        'counts': [10, 20],
    }
    spec = BidiSpec({
        'profile': {
            'display_name': 'user.name',
            'city': Path('user', 'address', 'city'),
        },
        'dotted': Path('payload', 'a.b'),
        'first': Path('counts', 0),
        'second': Path('counts', 1),
    })

    output = spec.forward(target)
    assert output == {
        'profile': {'display_name': 'Ada', 'city': 'London'},
        'dotted': 3,
        'first': 10,
        'second': 20,
    }
    assert spec.inverse(output) == target
    assert glom(target, spec) == output


def test_fixed_sequence_output_roundtrip():
    target = {'left': 'L', 'right': 'R', 'name': 'pair'}
    spec = BidiSpec({
        'name': 'name',
        'values': [Path('left'), Path('right')],
    })

    output = spec.forward(target)
    assert output == {'name': 'pair', 'values': ['L', 'R']}
    assert spec.inverse(output) == target


def test_duplicate_source_is_rejected():
    with pytest.raises(DuplicateSourceError):
        BidiSpec({'one': 'a.b', 'two': 'a.b'})


def test_parent_child_source_conflict_is_rejected():
    with pytest.raises(ParentChildConflictError):
        BidiSpec({'parent': 'a', 'child': 'a.b'})


def test_unsupported_specs_are_rejected():
    with pytest.raises(NotInvertibleError):
        BidiSpec({'x': lambda target: target['a']})
    with pytest.raises(NotInvertibleError):
        BidiSpec({'x': 1})
    with pytest.raises(NotInvertibleError):
        BidiSpec({'x': Coalesce('a', default=None)})
    with pytest.raises(NotInvertibleError):
        BidiSpec({'x': 'a.*'})
    with pytest.raises(NotInvertibleError):
        BidiSpec({'x': T['a'](str)})


def test_sequence_positions_must_be_fixed_and_contiguous():
    with pytest.raises(SequenceConflictError):
        BidiSpec({'first': Path('items', 0), 'last': Path('items', 2)})
    with pytest.raises(ParentChildConflictError):
        BidiSpec({'first': Path('items', 0), 'name': Path('items', 'name')})


def test_none_is_a_value_and_missing_is_an_error():
    spec = BidiSpec({'value': 'a'})

    assert spec.forward({'a': None}) == {'value': None}
    assert spec.inverse({'value': None}) == {'a': None}

    with pytest.raises(PathAccessError):
        spec.forward({})
    with pytest.raises(PathAccessError):
        spec.inverse({})


def test_inputs_are_not_mutated():
    target = {
        'a': {'b': [{'c': 2}]},
        'd.e': None,
    }
    original = copy.deepcopy(target)
    spec = BidiSpec({
        'nested': {'leaf': Path('a', 'b', 0, 'c')},
        'dot': Path('d.e'),
    })

    output = spec.forward(target)
    assert target == original
    restored = spec.inverse(output)
    assert target == original
    assert restored == original
    assert output == {'nested': {'leaf': 2}, 'dot': None}


def test_sidecar_preserves_unmapped_fields():
    target = {
        'user': {'name': 'Ada', 'age': 36},
        'meta': {'created': None, 'trace': 'abc'},
    }
    spec = BidiSpec({'name': 'user.name'}, unmapped='sidecar')
    output, sidecar = spec.forward(target, with_sidecar=True)

    assert output == {'name': 'Ada'}
    assert sidecar.data == {
        'user': {'age': 36},
        'meta': {'created': None, 'trace': 'abc'},
    }
    assert spec.inverse(output, sidecar=sidecar) == target


def test_error_policy_rejects_unmapped_fields():
    spec = BidiSpec({'name': 'user.name'})
    with pytest.raises(UnmappedFieldError):
        spec.forward({'user': {'name': 'Ada', 'age': 1}})

    output = spec.forward({'user': {'name': 'Ada'}})
    with pytest.raises(UnmappedFieldError):
        spec.inverse({'name': 'Ada', 'extra': None})


def test_ignore_policy_drops_unmapped_fields():
    spec = BidiSpec({'name': 'user.name'}, unmapped='ignore')
    assert spec.forward({'user': {'name': 'Ada', 'age': 1}}) == {'name': 'Ada'}
    output = {'name': 'Ada', 'extra': None}
    assert spec.inverse(output) == {'user': {'name': 'Ada'}}


def test_roundtrip_is_exact_after_serialization_shape():
    target = {
        'user': {'name': 'Ada'},
        'payload': {'a.b': [1, 2]},
    }
    spec = BidiSpec({
        'name': 'user.name',
        'dotted': Path('payload', 'a.b'),
    })
    output = spec.forward(target)
    assert spec.inverse(copy.deepcopy(output)) == target
