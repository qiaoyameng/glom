import pytest

from glom import (glom, Glommer, Path, T, S, A, Field, BadSpec,
                  PathAccessError, compile_projection, CompiledProjection)


class CountingNode:
    """Custom object which counts item reads."""
    def __init__(self, children):
        self._children = dict(children)
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return self._children[key]


def _counting_record():
    return CountingNode({
        'a': CountingNode({
            'b': CountingNode({'x': 1, 'y': 2}),
            'c': 3,
        }),
    })


_COUNTING_FIELDS = {
    'x': T['a']['b']['x'],
    'y': T['a']['b']['y'],
    'c': T['a']['c'],
}


def test_basic_projection_matches_plain_glom():
    target = {'user': {'name': 'Ada', 'addr': {'city': 'Paris', 'zip': '75001'}},
              'tags': ['a', 'b']}
    fields = {
        'name': T['user']['name'],
        'city': T['user']['addr']['city'],
        'zip': T['user']['addr']['zip'],
        'first_tag': T['tags'][0],
    }
    proj = compile_projection(fields)
    expected = glom(target, dict(fields))
    assert proj(target) == expected
    assert list(proj(target).keys()) == list(fields.keys())


def test_field_order_preserved():
    proj = compile_projection({
        'z': T['v'],
        'a': T['v'],
        'm': T['v'],
    })
    result = proj({'v': 1})
    assert list(result.keys()) == ['z', 'a', 'm']
    assert proj.field_names == ('z', 'a', 'm')


def test_list_of_pairs_and_empty_projection():
    proj = compile_projection([('a', T['x']), ('b', T['y'])])
    assert proj({'x': 1, 'y': 2}) == {'a': 1, 'b': 2}
    assert compile_projection({})({'anything': 1}) == {}
    assert compile_projection({'self': T})('rec') == {'self': 'rec'}


def test_path_objects_accepted():
    proj = compile_projection({
        'x': Path('a', 'b'),
        'y': Path(T['a'], 'c'),
        'z': Field(Path('a', 'missing'), default='D'),
    })
    assert proj({'a': {'b': 1, 'c': 2}}) == {'x': 1, 'y': 2, 'z': 'D'}


def test_compile_time_rejections():
    bad_specs = [
        T['a']['b'](),       # function call
        T.a.b(),             # method call
        T['a'][T['b']],      # dynamic segment
        T['a'].__star__(),   # wildcard
        T['a'].__starstar__(),
        T['a'] + 1,          # arithmetic
        'a.b',               # string spec
        ('a', len),          # tuple subspec
        {'a': T['b']},       # dict subspec
        lambda t: t,         # callable
        123,                 # non-spec
        S['a'],              # not rooted at T
        A['a'],              # assignment path
    ]
    for spec in bad_specs:
        with pytest.raises(BadSpec):
            compile_projection({'x': spec})
    # Field-wrapped specs are validated, too
    with pytest.raises(BadSpec):
        compile_projection({'x': Field('a.b', default='D')})
    with pytest.raises(BadSpec):
        compile_projection(42)
    with pytest.raises(TypeError):
        compile_projection({}, pure_types=('not a type',))


def test_per_field_defaults():
    target = {'a': {'b': None}}
    proj = compile_projection({
        # final value happens to be None: a real value, not missing
        'none_val': Field(T['a']['b'], default='D'),
        # missing mid-path: cannot descend into None
        'mid_missing': Field(T['a']['b']['c'], default='D'),
        # missing at the first segment
        'top_missing': Field(T['nope']['x'], default='D'),
        # explicit None default is distinct from no default
        'none_default': Field(T['nope'], default=None),
    })
    assert proj(target) == {
        'none_val': None,
        'mid_missing': 'D',
        'top_missing': 'D',
        'none_default': None,
    }
    # without a default, the same missing paths raise
    proj = compile_projection({'x': T['a']['b']['c']})
    with pytest.raises(PathAccessError):
        proj(target)
    proj = compile_projection({'x': T['nope']})
    with pytest.raises(PathAccessError):
        proj(target)


def test_error_location_matches_plain_glom():
    target = {'a': {'b': {}}}
    spec = T['a']['b']['c']['d']
    with pytest.raises(PathAccessError) as ref_exc:
        glom(target, {'x': spec})
    proj = compile_projection({'x': spec})
    with pytest.raises(PathAccessError) as proj_exc:
        proj(target)
    assert proj_exc.value.part_idx == ref_exc.value.part_idx == 2
    assert proj_exc.value.path == ref_exc.value.path
    assert proj_exc.value.path.values() == ('a', 'b', 'c', 'd')


def test_error_field_precedence_follows_declaration_order():
    # shared missing prefix: the error reported belongs to the
    # earliest-declared field, as with a plain glom dict spec
    proj = compile_projection({
        'x': T['missing']['b'],
        'y': T['missing']['c'],
    })
    with pytest.raises(PathAccessError) as exc_info:
        proj({})
    assert exc_info.value.part_idx == 0
    assert exc_info.value.path.values() == ('missing', 'b')

    proj = compile_projection({
        'y': T['missing']['c'],
        'x': T['missing']['b'],
    })
    with pytest.raises(PathAccessError) as exc_info:
        proj({})
    assert exc_info.value.path.values() == ('missing', 'c')

    # a default on the first field defers to the next failing field
    proj = compile_projection({
        'x': Field(T['missing']['b'], default='D'),
        'y': T['missing']['c'],
    })
    with pytest.raises(PathAccessError) as exc_info:
        proj({})
    assert exc_info.value.path.values() == ('missing', 'c')


def test_pure_types_share_prefix_reads_within_record():
    record = _counting_record()
    proj = compile_projection(_COUNTING_FIELDS, pure_types=(CountingNode,))
    assert proj(record) == {'x': 1, 'y': 2, 'c': 3}
    # 'a' read once on the root, 'b' and 'c' read once on 'a'
    assert record.reads == 1
    assert record._children['a'].reads == 2
    assert record._children['a']._children['b'].reads == 2


def test_undeclared_types_keep_plain_glom_read_pattern():
    record = _counting_record()
    proj = compile_projection(_COUNTING_FIELDS)
    # compare against a plain glom dict spec on a separate record
    assert proj(record) == glom(_counting_record(), dict(_COUNTING_FIELDS))
    # exactly the reads a plain glom dict spec would perform:
    # 'a' once per field, 'b' once per field through 'a'
    assert record.reads == 3
    assert record._children['a'].reads == 3
    assert record._children['a']._children['b'].reads == 2


def test_no_caching_across_records():
    proj = compile_projection(_COUNTING_FIELDS, pure_types=(CountingNode,))
    rec1 = _counting_record()
    rec2 = _counting_record()
    assert proj(rec1) == {'x': 1, 'y': 2, 'c': 3}
    assert proj(rec2) == {'x': 1, 'y': 2, 'c': 3}
    # each record was read independently; nothing carried over
    assert rec1.reads == 1
    assert rec2.reads == 1
    # and reprocessing a mutated record sees the new values
    rec1._children['a']._children['b']._children['x'] = 99
    assert proj(rec1)['x'] == 99


class FlipFlop:
    """Property which returns a different value on every read."""
    def __init__(self):
        self.calls = 0

    @property
    def val(self):
        self.calls += 1
        return self.calls


def test_impure_custom_type_not_optimized():
    fields = {'first': T['ff'].val, 'second': T['ff'].val}
    record = {'ff': FlipFlop()}
    expected = glom(record, dict(fields))
    assert expected == {'first': 1, 'second': 2}
    proj = compile_projection(fields)
    assert proj({'ff': FlipFlop()}) == expected


def test_declared_pure_type_opts_into_sharing():
    fields = {'first': T['ff'].val, 'second': T['ff'].val}
    proj = compile_projection(fields, pure_types=(FlipFlop,))
    ff = FlipFlop()
    assert proj({'ff': ff}) == {'first': 1, 'second': 1}
    assert ff.calls == 1


def test_mixed_pure_and_impure_layers():
    # pure dict prefix shared, impure custom suffix read per field
    fields = {'a': T['deep']['obj'].val, 'b': T['deep']['obj'].val}
    proj = compile_projection(fields)
    expected = glom({'deep': {'obj': FlipFlop()}}, dict(fields))
    assert proj({'deep': {'obj': FlipFlop()}}) == expected == {'a': 1, 'b': 2}


def test_usable_as_glom_spec_and_per_record_memo():
    proj = compile_projection(_COUNTING_FIELDS, pure_types=(CountingNode,))
    rec1, rec2 = _counting_record(), _counting_record()
    result = glom({'rows': [rec1, rec2]}, ('rows', [proj]))
    assert result == [{'x': 1, 'y': 2, 'c': 3}] * 2
    # memo is per record, not per projection
    assert rec1.reads == 1
    assert rec2.reads == 1


def test_glommer_scope_and_registered_get():
    class Bag:
        def __init__(self, d):
            self.d = d
            self.reads = 0

    glommer = Glommer()

    def bag_get(bag, key):
        bag.reads += 1
        return bag.d[key]

    glommer.register(Bag, get=bag_get)
    proj = compile_projection(
        {'x': Path('a', 'x'), 'y': Path('a', 'y')},
        pure_types=(Bag,))
    inner = Bag({'x': 1, 'y': 2})
    target = Bag({'a': inner})
    assert glommer.glom(target, proj) == {'x': 1, 'y': 2}
    # registered get handler was used, and 'a' was read once
    assert target.reads == 1
    assert inner.reads == 2


def test_projection_repr_and_field_repr():
    proj = compile_projection({'x': T['a']['b'], 'y': Field(T.a, default=1)})
    assert 'x' in repr(proj) and 'y' in repr(proj)
    assert repr(Field(T.a)) == 'Field(T.a)'
    assert repr(Field(T.a, default='D')) == "Field(T.a, default='D')"
    assert isinstance(proj, CompiledProjection)
