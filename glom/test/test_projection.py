from collections import OrderedDict

import pytest

from glom import (BadSpec,
                 Coalesce,
                 Glommer,
                 Path,
                 PathAccessError,
                 T,
                 glom)
from glom import Project
from glom.core import TargetRegistry


class TrackingDict(dict):
    reads = 0

    def __getitem__(self, key):
        type(self).reads += 1
        value = super().__getitem__(key)
        if isinstance(value, dict) and not isinstance(value, TrackingDict):
            tracked = TrackingDict(value)
            super().__setitem__(key, tracked)
            return tracked
        return value


class CountedBox:
    def __init__(self, value):
        self.value = value


class VolatileSource:
    def __init__(self):
        self.node_reads = 0

    @property
    def node(self):
        self.node_reads += 1
        return CountedBox(self.node_reads)


def test_project_shares_declared_pure_prefix_within_one_record():
    target = TrackingDict({'shared': {'a': 1, 'b': 2}})
    fields = {
        'a': T['shared']['a'],
        'b': T['shared']['b'],
    }

    TrackingDict.reads = 0
    ordinary = Project(fields)
    assert ordinary.glom(target) == {'a': 1, 'b': 2}
    assert TrackingDict.reads == 4
    TrackingDict.reads = 0

    glommer = Glommer()
    glommer.register(TrackingDict, get=dict.__getitem__, pure=True)
    compiled = Project(fields)

    assert glommer.glom(target, compiled) == {'a': 1, 'b': 2}
    assert TrackingDict.reads == 3

    TrackingDict.reads = 0
    second = TrackingDict({'shared': {'a': 3, 'b': 4}})
    glommer.glom(second, compiled)
    assert TrackingDict.reads == 3


def test_project_does_not_cache_across_records():
    glommer = Glommer()
    glommer.register(TrackingDict, get=dict.__getitem__, pure=True)
    target = TrackingDict({'shared': {'a': 1, 'b': 2}})
    compiled = Project({'a': T['shared']['a'], 'b': T['shared']['b']})

    TrackingDict.reads = 0
    glommer.glom(target, compiled)
    assert TrackingDict.reads == 3
    glommer.glom(target, compiled)
    assert TrackingDict.reads == 6


def test_undeclared_custom_attribute_keeps_ordinary_observations():
    target = VolatileSource()
    compiled = Project({'a': T.node.value, 'b': T.node.value})

    assert compiled.glom(target) == {'a': 1, 'b': 2}
    assert target.node_reads == 2
    ordinary_target = VolatileSource()
    assert glom(ordinary_target, {'a': T.node.value, 'b': T.node.value}) == {'a': 1, 'b': 2}
    assert ordinary_target.node_reads == 2


def test_declared_custom_attribute_can_share_prefix():
    glommer = Glommer()
    glommer.register(VolatileSource, pure=True)
    target = VolatileSource()
    compiled = Project({'a': T.node.value, 'b': T.node.value})

    assert glommer.glom(target, compiled) == {'a': 1, 'b': 1}
    assert target.node_reads == 1


def test_project_preserves_order_defaults_and_none_terminal():
    compiled = Project(OrderedDict([
        ('mid', (T.a.b.c, 'mid-default')),
        ('none', (T['a'], 'none-default')),
        ('last', T['last']),
    ]))
    target = {'a': None, 'last': 'done'}

    result = compiled.glom(target)
    assert list(result) == ['mid', 'none', 'last']
    assert result['mid'] == 'mid-default'
    assert result['none'] is None
    assert result['last'] == 'done'


def test_project_supports_independent_default_forms():
    fields = {
        'tuple': (T.missing, 'tuple-default'),
        'coalesce': Coalesce(T.missing, default='coalesce-default'),
        'mapping': T.missing,
    }
    compiled = Project(fields, defaults={'mapping': 'mapping-default'})
    assert compiled.glom({}) == {
        'tuple': 'tuple-default',
        'coalesce': 'coalesce-default',
        'mapping': 'mapping-default',
    }


def test_project_preserves_access_error_location():
    compiled = Project({'value': T['a']['b']['c']})

    with pytest.raises(PathAccessError) as exc_info:
        compiled.glom({'a': {}})
    assert exc_info.value.part_idx == 1
    assert exc_info.value.path == Path(T['a']['b']['c'])

    with pytest.raises(PathAccessError) as exc_info:
        compiled.glom({'a': {'b': {}}})
    assert exc_info.value.part_idx == 2


def test_project_rejects_unsupported_specs_at_compile_time():
    def reject(fields, **kwargs):
        with pytest.raises(BadSpec):
            Project(fields, **kwargs)

    reject({'x': lambda target: target})
    reject({'x': T.a()})
    reject({'x': T.a.__star__()})
    reject({'x': T[T['dynamic']]})
    reject({'x': T.a + 1})
    reject({'x': {'nested': T.a}})
    reject({'x': (T.a, lambda: 'default')})
    reject({T.dynamic: T.a})
    with pytest.raises(TypeError):
        Project({['x']: T.a})
    reject({'x': (T.a, 'default')}, defaults={'x': 'another-default'})

    with pytest.raises(BadSpec):
        Project('not-a-mapping')


def test_project_accepts_star_as_an_explicit_static_key():
    assert Project({'x': T['*']}).glom({'*': 'value'}) == {'x': 'value'}
    assert Project({'x': Path('*')}).glom({'*': 'value'}) == {'x': 'value'}


def test_project_accepts_static_callable_key():
    def key_func():
        pass

    target = {key_func: 'found'}
    assert Project({'x': T[key_func]}).glom(target) == {'x': 'found'}


def test_project_shares_explicit_path_prefix_for_declared_pure_type():
    glommer = Glommer()
    glommer.register(TrackingDict, get=lambda obj, key: obj[key], pure=True)
    target = TrackingDict({'shared': {'a': 1, 'b': 2}})
    compiled = Project({
        'a': Path('shared', 'a'),
        'b': Path('shared', 'b'),
    })

    TrackingDict.reads = 0
    assert glommer.glom(target, compiled) == {'a': 1, 'b': 2}
    assert TrackingDict.reads == 3


def test_pure_registration_only_marks_read_operations():
    glommer = Glommer()

    class IterableTrackingDict(TrackingDict):
        def __iter__(self):
            return iter(())

    glommer.register(IterableTrackingDict, iterate=iter, pure=True)
    registry = glommer.scope[TargetRegistry]
    target = IterableTrackingDict()
    assert registry.is_pure(target, 'get')
    assert registry.is_pure(target, 'item')
    assert registry.is_pure(target, 'attr')
    assert not registry.is_pure(target, 'iterate')
