import copy

import pytest

from glom import glom
from glom.projection import (Field, Project, ProjectionError,
                             load as project_load, dump as project_dump)


TARGET = {
    'service': {
        'host': 'example.com',
        'port': 8080,
        'tls': True,
        'tags': ['web', 'primary'],
    },
    'untouched': {'nested': [1, 2, 3]},
}


SPEC = Project({
    'host': Field('host'),
    'port_str': Field('port', get=str, put=int),
    'tls': Field('tls'),
}, base='service')


def test_load_projects_form_view():
    form = SPEC.load(copy.deepcopy(TARGET))
    assert form == {'host': 'example.com', 'port_str': '8080', 'tls': True}


def test_project_is_a_glomspec():
    assert glom(TARGET, SPEC) == SPEC.load(TARGET)


def test_read_then_write_is_identity():
    # Law 1: reading a target and writing the form straight back must
    # not change the target.
    original = copy.deepcopy(TARGET)
    form = SPEC.load(original)
    result = SPEC.dump(form, original)
    assert result == TARGET
    # original is not mutated by default
    assert original == TARGET
    assert result is not original


def test_write_then_read_matches_input_view():
    # Law 2: after writing an arbitrary valid form, reading again must
    # reproduce that form view.
    original = copy.deepcopy(TARGET)
    new_form = {'host': 'other.example', 'port_str': '443', 'tls': False}
    written = SPEC.dump(new_form, original)
    assert SPEC.load(written) == new_form
    assert written['service']['port'] == 443


def test_in_place_mutation():
    target = copy.deepcopy(TARGET)
    form = {'host': 'x', 'port_str': '80', 'tls': True}
    result = SPEC.dump(form, target, in_place=True)
    assert result is target
    assert target['service']['host'] == 'x'
    assert target['service']['port'] == 80


def test_unprojected_parts_preserved():
    # covered fields change, everything else is retained verbatim
    form = {'host': 'new', 'port_str': '443', 'tls': True}
    result = SPEC.dump(form, copy.deepcopy(TARGET))
    assert result['service']['tags'] == ['web', 'primary']
    assert result['untouched'] == {'nested': [1, 2, 3]}
    assert result['service']['host'] == 'new'


def test_missing_form_key_leaves_source_as_is():
    form = {'host': 'new'}  # port_str / tls omitted
    result = SPEC.dump(form, copy.deepcopy(TARGET))
    assert result['service']['host'] == 'new'
    assert result['service']['port'] == 8080
    assert result['service']['tls'] is True


def test_raw_path_fields_and_nested_form():
    spec = Project({
        'endpoint': {
            'host': 'service.host',
            'port': Field('service.port', get=str, put=int),
        },
    })
    form = spec.load(TARGET)
    assert form == {'endpoint': {'host': 'example.com', 'port': '8080'}}
    written = spec.dump({'endpoint': {'host': 'h', 'port': '22'}},
                        copy.deepcopy(TARGET))
    assert written['service'] == {
        'host': 'h', 'port': 22, 'tls': True,
        'tags': ['web', 'primary']}
    assert written['untouched'] == TARGET['untouched']


def test_nested_project_composes_with_base():
    inner = Project({
        'host': Field('host'),
        'port': Field('port', get=str, put=int),
    }, base='service')
    outer = Project({'connection': inner})
    form = outer.load(TARGET)
    assert form == {'connection': {'host': 'example.com', 'port': '8080'}}
    written = outer.dump(
        {'connection': {'host': 'h', 'port': '22'}}, copy.deepcopy(TARGET))
    assert written['service']['host'] == 'h'
    assert written['service']['port'] == 22
    assert written['untouched'] == TARGET['untouched']


def test_shared_source_agreeing_fields_write_once():
    # two form fields over the same source; both reverse to the same
    # value, so the write succeeds and the source is set once.
    spec = Project({
        'p1': Field('service.port', get=str, put=int),
        'p2': Field('service.port', get=str, put=int),
    })
    form = {'p1': '443', 'p2': '443'}
    written = spec.dump(form, copy.deepcopy(TARGET))
    assert written['service']['port'] == 443
    assert written['untouched'] == TARGET['untouched']


def test_shared_source_conflict_fails_atomically():
    spec = Project({
        'p1': Field('service.port', get=str, put=int),
        'p2': Field('service.port', get=str, put=int),
    })
    target = copy.deepcopy(TARGET)
    form = {'p1': '443', 'p2': '9999'}
    with pytest.raises(ProjectionError) as excinfo:
        spec.dump(form, target)
    assert 'conflicting write-back' in str(excinfo.value)
    # atomic: nothing changed, unprojected and projected parts intact
    assert target == TARGET


def test_one_way_get_cannot_write():
    spec = Project({'port_x2': Field('service.port', get=lambda v: v * 2)})
    target = copy.deepcopy(TARGET)
    with pytest.raises(ProjectionError) as excinfo:
        spec.dump({'port_x2': 10}, target)
    assert 'one-way' in str(excinfo.value)
    assert target == TARGET  # untouched on failure


def test_lossy_default_cannot_write_without_put():
    spec = Project({'missing': Field('service.absent', default='N/A')})
    assert spec.load(TARGET)['missing'] == 'N/A'
    target = copy.deepcopy(TARGET)
    with pytest.raises(ProjectionError) as excinfo:
        spec.dump({'missing': 'N/A'}, target)
    assert 'lossy default' in str(excinfo.value)
    assert target == TARGET


def test_default_with_explicit_put_is_writable():
    spec = Project({
        'missing': Field('service.absent', default=0, put=int),
    })
    written = spec.dump({'missing': 5}, copy.deepcopy(TARGET))
    assert written['service']['absent'] == 5
    assert written['service']['port'] == 8080


def test_missing_source_without_default_raises_on_read():
    spec = Project({'missing': Field('service.absent')})
    with pytest.raises(Exception):
        spec.load(TARGET)


def test_convenience_functions():
    fields = {'host': Field('host'),
              'port': Field('port', get=str, put=int)}
    form = project_load(TARGET, fields, base='service')
    assert form == {'host': 'example.com', 'port': '8080'}
    written = project_dump({'host': 'h', 'port': '22'},
                           copy.deepcopy(TARGET), fields, base='service')
    assert written['service']['host'] == 'h'
    assert written['service']['port'] == 22


def test_bad_field_definitions():
    with pytest.raises(TypeError):
        Project({'x': 123})  # non-dict/Project top-level handled below
    with pytest.raises(TypeError):
        Project({1: 'service.host'})
    with pytest.raises(TypeError):
        Field(object())
