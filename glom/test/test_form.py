import copy

import pytest

from glom import (Form, Field, FormError, FormConflictError,
                  FieldNotWritableError, BadSpec, Path, T)


def make_target():
    return {
        'user': {'name': 'Ada', 'age': 36, 'tags': ['admin', 'ops']},
        'theme': 'dark',
        'limits': {'cpu': '4', 'mem_mb': 512},
    }


def make_form():
    return Form({
        'name': Field('user.name'),
        'age': Field('user.age', forward=str, backward=int),
        'tags': Field('user.tags',
                      forward=lambda tags: ','.join(tags),
                      backward=lambda text: text.split(',')),
        'profile': Form({
            'cpu': Field('limits.cpu'),
        }),
    })


def test_read_projection():
    view = make_form().read(make_target())
    assert view == {'name': 'Ada', 'age': '36', 'tags': 'admin,ops',
                    'profile': {'cpu': '4'}}


def test_law_read_then_unchanged_write_is_identity():
    form = make_form()
    target = make_target()
    snapshot = copy.deepcopy(target)

    result = form.write(target, form.read(target))

    assert result == snapshot
    assert target == snapshot  # original object untouched


def test_law_write_then_read_matches_view():
    form = make_form()
    target = make_target()
    view = {'name': 'Grace', 'age': '41', 'tags': 'dev,ops',
            'profile': {'cpu': '8'}}

    new_target = form.write(target, view)

    assert form.read(new_target) == view
    # unprojected parts of the original are preserved
    assert new_target['theme'] == 'dark'
    assert new_target['limits']['mem_mb'] == 512
    # and the original object itself is fully intact
    assert target == make_target()


def test_shared_source_agreement_writes_once():
    form = Form({
        'age_int': Field('user.age', forward=str, backward=int),
        'age_float': Field('user.age', forward=str, backward=float),
    })
    target = make_target()

    new_target = form.write(target, {'age_int': '40', 'age_float': '40.0'})

    assert new_target['user']['age'] == 40
    assert target['user']['age'] == 36


def test_shared_source_conflict_aborts_write():
    form = Form({
        'age_int': Field('user.age', forward=str, backward=int),
        'age_float': Field('user.age', forward=str, backward=float),
        'name': Field('user.name'),
    })
    target = make_target()
    snapshot = copy.deepcopy(target)

    with pytest.raises(FormConflictError):
        form.write(target, {'age_int': '40', 'age_float': '40.5', 'name': 'Grace'})

    # the whole write failed: original object and unprojected fields intact
    assert target == snapshot


def test_one_way_field_is_not_writable():
    form = Form({
        'name': Field('user.name'),
        'age': Field('user.age', forward=str),
    })
    target = make_target()

    # unchanged round-trip is fine, even for read-only fields
    assert form.write(target, form.read(target)) == target

    with pytest.raises(FieldNotWritableError):
        form.write(target, {'name': 'Ada', 'age': '40'})


def test_lossy_default_does_not_become_reversible():
    field = Field('user.missing', forward=str, default='n/a')
    assert not field.writable
    assert field.read(make_target()) == 'n/a'

    form = Form({'missing': field, 'name': Field('user.name')})
    with pytest.raises(FieldNotWritableError):
        form.write(make_target(), {'missing': 'x', 'name': 'Ada'})

    # a plain default with no conversion is still a writable pass-through
    assert Field('user.missing', default='n/a').writable


def test_nested_form_roundtrip():
    form = Form({
        'limits': Form({
            'cpu': Field('limits.cpu'),
            'mem': Field('limits.mem_mb', forward=str, backward=int),
        }),
    })
    target = make_target()
    view = {'limits': {'cpu': '16', 'mem': '1024'}}

    new_target = form.write(target, view)

    assert form.read(new_target) == view
    assert new_target['limits'] == {'cpu': '16', 'mem_mb': 1024}
    assert target['limits'] == {'cpu': '4', 'mem_mb': 512}


def test_view_validation():
    form = make_form()
    target = make_target()
    view = form.read(target)

    del view['name']
    with pytest.raises(FormError, match='missing fields'):
        form.write(target, view)

    view = form.read(target)
    view['bogus'] = 1
    with pytest.raises(FormError, match='unexpected fields'):
        form.write(target, view)


def test_spec_validation():
    with pytest.raises(BadSpec):
        Field(123)
    with pytest.raises(BadSpec):
        Field('a.b', forward='not-callable')
    with pytest.raises(BadSpec):
        Form(['not', 'a', 'mapping'])
    with pytest.raises(BadSpec):
        Form({'x': 'a.b'})  # fields must be explicit Field/Form specs


def test_path_and_t_sources():
    form = Form({
        'name': Field(Path('user', 'name')),
        'theme': Field(T['theme']),
    })
    target = make_target()

    new_target = form.write(target, {'name': 'Grace', 'theme': 'light'})

    assert new_target['user']['name'] == 'Grace'
    assert new_target['theme'] == 'light'
    assert target == make_target()
