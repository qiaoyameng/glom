import pytest

from glom import glom, Path, T, S, Spec, Glommer, PathAssignError, PathAccessError, Coalesce
from glom import assign, Assign, delete, Delete, PathDeleteError, Or
from glom import Update, update, PathUpdateError, Call
from glom import core
from glom.core import register as register_type
from glom.mutation import UPDATE_TARGET, UPDATE_ARGS
from glom.core import UnregisteredTarget


def test_assign():
    class Foo:
        pass

    assert glom({}, Assign(T['a'], 1)) == {'a': 1}
    assert glom({'a': {}}, Assign(T['a']['a'], 1)) == {'a': {'a': 1}}
    assert glom({'a': {}}, Assign('a.a', 1)) == {'a': {'a': 1}}
    assert glom(Foo(), Assign(T.a, 1)).a == 1
    assert glom({}, Assign('a', 1)) == {'a': 1}
    assert glom(Foo(), Assign('a', 1)).a == 1
    assert glom({'a': Foo()}, Assign('a.a', 1))['a'].a == 1
    def r():
        r = {}
        r['r'] = r
        return r
    assert glom(r(), Assign('r.r.r.r.r.r.r.r.r', 1)) == {'r': 1}
    assert glom(r(), Assign(T['r']['r']['r']['r'], 1)) == {'r': 1}
    assert glom(r(), Assign(Path('r', 'r', T['r']), 1)) == {'r': 1}
    assert assign(r(), Path('r', 'r', T['r']), 1) == {'r': 1}
    with pytest.raises(TypeError, match='path argument must be'):
        Assign(1, 'a')
    with pytest.raises(ValueError, match='path must have at least one element'):
        Assign(T, 1)

    assert repr(Assign(T.a, 1)) == 'Assign(T.a, 1)'
    assign_spec = Assign(T.a, 1, missing=dict)
    assert repr(assign_spec) == "Assign(T.a, 1, missing=dict)"
    assert repr(assign_spec) == repr(eval(repr(assign_spec)))


def test_assign_recursive():
    val = {}
    val[1] = [val]
    recur_out = glom({}, Assign(T['a'], val))['a']
    assert recur_out[1][0] is recur_out


def test_assign_spec_val():
    output = glom({'b': 'c'}, Assign('a', Spec('b')))
    assert output['a'] == output['b'] == 'c'
    output = glom({'b': 'c'}, Assign('a', Or('d', 'b')))
    assert output['a'] == output['b'] == 'c'


def test_unregistered_assign():
    # test with bare target registry
    glommer = Glommer(register_default_types=False)

    with pytest.raises(UnregisteredTarget, match='assign'):
        glommer.glom({}, Assign('a', 'b'))

    # test for unassignable tuple
    with pytest.raises(UnregisteredTarget, match='assign'):
        glom({'a': ()}, Assign('a.0', 'b'))


def test_bad_assign_target():
    class BadTarget:
        def __setattr__(self, name, val):
            raise Exception("and you trusted me?")

    # sanity check
    spec = Assign('a', 'b')
    ok_target = lambda: None
    glom(ok_target, spec)
    assert ok_target.a == 'b'

    with pytest.raises(PathAssignError, match='could not assign'):
        glom(BadTarget(), spec)

    with pytest.raises(PathAccessError, match='could not access'):
        assign({}, 'a.b.c', 'moot')
    return


def test_sequence_assign():
    target = {'alist': [0, 1, 2]}
    assign(target, 'alist.2', 3)
    assert target['alist'][2] == 3

    with pytest.raises(PathAssignError, match='could not assign') as exc_info:
        assign(target, 'alist.3', 4)

    # the following test is because pypy's IndexError is different than CPython's:
    # E         - PathAssignError(IndexError('list index out of range',), Path('alist'), '3')
    # E         + PathAssignError(IndexError('list assignment index out of range',), Path('alist'), '3')
    # E         ?                                  +++++++++++

    exc_repr = repr(exc_info.value)
    assert exc_repr.startswith('PathAssignError(')
    assert exc_repr.endswith("'3')")
    return


def test_invalid_assign_op_target():
    target = {'afunc': lambda x: 'hi %s' % x}
    spec = T['afunc'](x=1)

    with pytest.raises(ValueError):
        assign(target, spec, None)
    return


def test_assign_missing_signature():
    # test signature (non-callable missing hook)
    with pytest.raises(TypeError, match='callable'):
        assign({}, 'a.b.c', 'lol', missing='invalidbcnotcallable')
    return


def test_assign_missing_dict():
    target = {}
    val = object()

    from itertools import count
    counter = count()
    def debugdict():
        ret = dict()
        #ret['id'] = id(ret)
        #ret['inc'] = counter.next()
        return ret

    assign(target, 'a.b.c.d', val, missing=debugdict)

    assert target == {'a': {'b': {'c': {'d': val}}}}


def test_assign_missing_object():
    val = object()
    class Container:
        pass

    target = Container()
    target.a = extant_a = Container()
    assign(target, 'a.b.c.d', val, missing=Container)

    assert target.a.b.c.d is val
    assert target.a is extant_a  # make sure we didn't overwrite anything on the path


def test_assign_missing_with_extant_keys():
    """This test ensures that assign with missing doesn't overwrite
    perfectly fine extant keys that are along the path it needs to
    assign to. call count is also checked to make sure missing() isn't
    invoked too many times.

    """
    target = {}
    value = object()
    default_struct = {'b': {'c': {}}}

    call_count = [0]

    def _get_default_struct():
        call_count[0] += 1  # make sure this is only called once
        return default_struct

    assign(target, 'a.b.c', value, missing=_get_default_struct)

    assert target['a']['b']['c'] is value
    assert target['a']['b'] is default_struct['b']
    assert call_count == [1]


def test_assign_missing_unassignable():
    """Check that the final assignment to the target object comes last,
    ensuring that failed assignments don't leave targets in a bad
    state.

    """

    class Tarjay:
        init_count = 0
        def __init__(self):
            self.__class__.init_count += 1

        @property
        def unassignable(self):
            return

    value = object()
    target = {"preexisting": "ok"}

    with pytest.raises(PathAssignError):
        assign(target, 'tarjay.unassignable.a.b.c', value, missing=Tarjay)

    assert target == {'preexisting': 'ok'}

    # why 3? "c" gets the value of "value", while "b", "a", and
    # "tarjay" all succeed and are set to Tarjay instances. Then
    # unassignable is already present, but not possible to assign to,
    # raising the PathAssignError.
    assert Tarjay.init_count == 3


def test_s_assign():
    '''
    check that assign works when storing things into S
    '''
    assert glom({}, (Assign(S['foo'], 'bar'), S['foo'])) == 'bar'


def test_delete():
    class Foo:
        def __init__(self, d=None):
            for k, v in d.items():
                setattr(self, k, v)

    assert glom({'a': 1}, Delete(T['a'])) == {}
    assert glom({'a': {'a': 1}}, Delete(T['a']['a'])) == {'a': {}}
    assert glom({'a': {'a': 1}}, Delete('a.a')) == {'a': {}}
    assert not hasattr(glom(Foo({'a': 1}), Delete(T.a)), 'a')
    assert glom({'a': 1}, Delete('a')) == {}
    assert not hasattr(glom(Foo({'a': 1}), Delete('a')), 'a')
    assert not hasattr(glom({'a': Foo({'a': 1})}, Delete('a.a'))['a'], 'a')

    def r():
        r = {}
        r['r'] = r
        return r

    assert glom(r(), Delete('r.r.r.r.r.r.r.r.r')) == {}
    assert glom(r(), Delete(T['r']['r']['r']['r'])) == {}
    assert glom(r(), Delete(Path('r', 'r', T['r']))) == {}
    assert delete(r(), Path('r', 'r', T['r'])) == {}
    with pytest.raises(TypeError, match='path argument must be'):
        Delete(1, 'a')
    with pytest.raises(ValueError, match='path must have at least one element'):
        Delete(T, 1)

    assert repr(Delete(T.a)) == 'Delete(T.a)'

    # test delete from scope
    assert glom(1, (S(x=T), S['x'])) == 1
    with pytest.raises(PathAccessError):
        glom(1, (S(x=T), Delete(S['x']), S['x']))

    # test raising on missing parent
    with pytest.raises(PathAccessError):
        glom({}, Delete(T['a']['b']))

    # test raising on missing index
    with pytest.raises(PathDeleteError):
        glom([], Delete(T[0]))
    target = []
    assert glom(target, Delete(T[0], ignore_missing=True)) is target

    # test raising on missing attr
    with pytest.raises(PathDeleteError):
        glom(object(), Delete(T.does_not_exist))
    target = object()
    assert glom(target, Delete(T.does_not_exist, ignore_missing=True)) is target


def test_unregistered_delete():
    glommer = Glommer(register_default_types=False)

    with pytest.raises(UnregisteredTarget, match='delete'):
        glommer.glom({'a': 1}, Delete('a'))

    with pytest.raises(UnregisteredTarget, match='delete'):
        glom({'a': (1,)}, Delete('a.0'))


def test_bad_delete_target():
    class BadTarget:
        def __delattr__(self, name):
            raise Exception("and you trusted me?")

    spec = Delete('a')
    ok_target = lambda: None
    ok_target.a = 1
    glom(ok_target, spec)
    assert not hasattr(ok_target, 'a')

    with pytest.raises(PathDeleteError, match='could not delete'):
        glom(BadTarget(), spec)

    with pytest.raises(PathDeleteError, match='could not delete'):
        delete({}, 'a')
    return


def test_sequence_delete():
    target = {'alist': [0, 1, 2]}
    delete(target, 'alist.1')
    assert target['alist'] == [0, 2]

    with pytest.raises(PathDeleteError, match='could not delete') as exc_info:
        delete(target, 'alist.2')

    exc_repr = repr(exc_info.value)
    assert exc_repr.startswith('PathDeleteError(')
    assert exc_repr.endswith("'2')")
    return


def test_invalid_delete_op_target():
    target = {'afunc': lambda x: 'hi %s' % x}
    spec = T['afunc'](x=1)

    with pytest.raises(ValueError):
        delete(target, spec, None)
    return


def test_delete_ignore_missing():
    assert delete({}, 'a', ignore_missing=True) == {}
    assert delete({}, 'a.b', ignore_missing=True) == {}


def test_star_broadcast():
    val = {'a': [{'b': [{'c': 1}, {'c': 2}, {'c': 3}]}]}
    assert glom(val, (Assign('a.*.b.*.d', 'a'), 'a.*.b.*.d')) == [['a', 'a', 'a']]
    glom(val, Delete('a.*.b.*.d'))
    assert 'c' in val['a'][0]['b'][0]
    assert 'd' not in val['a'][0]['b'][0]


# ---------------------------------------------------------------------------
# Update (leaf-wise mutation)
# ---------------------------------------------------------------------------

def _identity(v):
    return v


def test_update_per_leaf_basic():
    target = {'orders': [{'price': 100}, {'price': 200}]}
    out = glom(target, Update('orders.*.price', lambda p: p * 2))
    assert out is target
    assert target == {'orders': [{'price': 200}, {'price': 400}]}

    # a single (non-wildcard) leaf is updated from its own old value
    target = {'a': {'b': 1}}
    glom(target, Update('a.b', lambda v: v + 10))
    assert target == {'a': {'b': 11}}

    # convenience function
    target = {'orders': [{'price': 100}, {'price': 200}]}
    update(target, 'orders.*.price', lambda p: p + 1)
    assert target == {'orders': [{'price': 101}, {'price': 201}]}


def test_update_leaves_get_independent_values():
    # the whole point: each leaf is computed from its own old value,
    # rather than one computed result broadcast to every leaf
    target = {'orders': [{'price': 1}, {'price': 2}, {'price': 3}]}
    seen = []
    glom(target, Update('orders.*.price',
                        lambda p: (seen.append(p), p * 10)[1]))
    assert seen == [1, 2, 3]
    assert [o['price'] for o in target['orders']] == [10, 20, 30]


def test_update_mixed_dict_list_double_star():
    data = {'groups': [{'items': [{'price': 1}, {'price': 2}]},
                       {'items': [{'price': 3}]}]}
    glom(data, Update('groups.*.items.*.price', lambda p: p + 100))
    assert data['groups'][0]['items'] == [{'price': 101}, {'price': 102}]
    assert data['groups'][1]['items'] == [{'price': 103}]


def test_update_call_args_and_target_scope():
    # call arguments reachable directly as S[...] ...
    target = {'orders': [{'price': 100}, {'price': 200}]}
    glom(target, Update('orders.*.price', T * S['factor']),
         scope={'factor': 0.9})
    assert target == {'orders': [{'price': 90.0}, {'price': 180.0}]}

    # ... and via the explicit UPDATE_ARGS / UPDATE_TARGET entries
    target = {'orders': [{'price': 100}]}
    def conv(price, root_target, discount):
        return price - len(root_target['orders']) - discount
    spec = Update('orders.*.price',
                  Call(conv, args=(T, S[UPDATE_TARGET], S[UPDATE_ARGS]['discount'])))
    glom(target, spec, scope={'discount': 5})
    assert target['orders'][0]['price'] == 94

    # update() helper forwards scope
    target = {'orders': [{'price': 100}]}
    update(target, 'orders.*.price', T * S['factor'], scope={'factor': 0.5})
    assert target == {'orders': [{'price': 50.0}]}


def test_update_shared_subobject_updated_once():
    shared = {'price': 100}
    target = {'items': {'a': shared, 'b': shared, 'c': {'price': 5}}}
    seen = []
    glom(target, Update('items.*.price',
                        lambda p: (seen.append(p), p + 1)[1]))
    # shared leaf updated a single time, at first discovery
    assert seen == [100, 5]
    assert shared['price'] == 101
    assert target['items']['c']['price'] == 6

    # shared leaf inside a list (same object repeated)
    shared = {'v': 1}
    target = [shared, shared]
    seen = []
    glom(target, Update('*.v', lambda v: (seen.append(v), v * 10)[1]))
    assert seen == [1]
    assert shared['v'] == 10


def test_update_dedup_order_is_first_discovery():
    shared = {'price': 1}
    target = {'k1': shared, 'k2': shared, 'k3': {'price': 2}}

    order = []
    def conv(p):
        order.append(p)
        return p

    glom(target, Update('*.price', conv))
    assert order == [1, 2]


def test_update_objects_and_registered_type():
    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    target = Obj(a=Obj(price=10), b=Obj(price=20))
    glom(target, Update('*.price', lambda p: p + 1))
    assert target.a.price == 11 and target.b.price == 21

    class Pair:
        def __init__(self, data=None):
            self._d = dict(data or {})

    register_type(
        Pair,
        get=lambda p, k: p._d[k],
        keys=lambda p: list(p._d),
        assign=lambda p, k, v: p._d.__setitem__(k, v))
    p = Pair({'x': Pair({'price': 1}), 'y': Pair({'price': 2})})
    glom(p, Update('*.price', lambda v: v * 3))
    assert p._d['x']._d['price'] == 3
    assert p._d['y']._d['price'] == 6


def test_update_unregistered_type():
    glommer = Glommer(register_default_types=False)
    with pytest.raises(UnregisteredTarget, match='get'):
        glommer.glom({}, Update('a', lambda v: v))


def test_update_missing_default_raises():
    with pytest.raises(PathAccessError, match='could not access'):
        glom({}, Update('a.b.c', lambda v: v + 1))

    with pytest.raises(PathAccessError, match='could not access'):
        glom({'a': {}}, Update('a.b', lambda v: v + 1))

    # wildcard: a parent that does not exist at all
    with pytest.raises(PathAccessError):
        glom({}, Update('orders.*.price', lambda p: p + 1))


def test_update_skip_missing():
    # non-wildcard missing leaf / parent leaves the target untouched
    target = {'a': {}}
    glom(target, Update('a.b', lambda v: v + 1, skip_missing=True))
    assert target == {'a': {}}

    target = {}
    glom(target, Update('a.b.c', lambda v: v, skip_missing=True))
    assert target == {}

    # wildcard: missing leaves skipped, present leaves updated
    target = {'orders': [{'price': 10}, {'other': 1}, {'price': 20}]}
    glom(target, Update('orders.*.price', lambda p: p + 1, skip_missing=True))
    assert target['orders'][0]['price'] == 11
    assert 'price' not in target['orders'][1]
    assert target['orders'][2]['price'] == 21


def test_update_missing_callable_creates():
    target = {}
    glom(target, Update('a.b.c', lambda old: 99, missing=dict))
    assert target == {'a': {'b': {'c': 99}}}

    # leaf missing: callable supplies the "old value", then it is transformed
    target = {'a': {}}
    glom(target, Update('a.b', lambda old: old * 2, missing=lambda: 10))
    assert target == {'a': {'b': 20}}

    with pytest.raises(TypeError, match='callable'):
        Update('a', lambda v: v, missing='notcallable')


def test_update_wildcard_missing_leaf_created():
    # a present wildcard container whose leaf key is absent gets created
    target = {'orders': [{'price': 10}, {'other': 1}]}
    glom(target, Update('orders.*.price', lambda old: old * 2,
                        missing=lambda: 5))
    assert target['orders'][0]['price'] == 20
    assert target['orders'][1]['price'] == 10


def test_update_double_star():
    target = {'a': {'n': {'v': 1}}, 'b': {'n': {'v': 2}}}
    glom(target, Update('**.n.v', lambda x: x * 5))
    assert target['a']['n']['v'] == 5
    assert target['b']['n']['v'] == 10


def test_update_s_destination():
    out = glom({'x': 1},
               (S(rows=[1, 2, 3]),
                Update(S['rows'], lambda r: [v * 2 for v in r]),
                S['rows']))
    assert out == [2, 4, 6]


def test_update_scope_entries_are_scoped_per_call():
    # outside the Update, the explicit entries are not leaked into scope
    target = {'orders': [{'price': 1}]}
    out = glom(target,
               (Update('orders.*.price', lambda p: p + 1),
                {'leaked': Coalesce(S[UPDATE_TARGET], default=None)}))
    assert out == {'leaked': None}


def test_update_conversion_failure_retains_prior_updates():
    target = {'orders': [{'price': 10}, {'price': 20}, {'price': 30}]}

    def conv(p):
        if p == 20:
            raise ValueError('bad price data')
        return p + 1

    with pytest.raises(PathUpdateError) as exc_info:
        glom(target, Update('orders.*.price', conv))

    err = exc_info.value
    assert err.updated_count == 1
    message = err.get_message()
    assert 'could not update' in message
    assert '1 leaf target(s) were updated' in message
    assert 'retained' in message

    # earlier leaf kept, failed leaf and later leaves untouched
    assert target['orders'][0]['price'] == 11
    assert target['orders'][1]['price'] == 20
    assert target['orders'][2]['price'] == 30

    # full string retains the target-spec path diagnostics
    assert 'Target-spec trace' in str(err)
    assert 'orders' in str(err)


def test_update_first_failure_no_updates():
    target = {'orders': [{'price': 10}, {'price': 20}]}
    with pytest.raises(PathUpdateError) as exc_info:
        glom(target, Update('orders.*.price', T * S['does_not_exist']))
    err = exc_info.value
    assert err.updated_count == 0
    assert 'no prior modifications' in err.get_message()
    assert target['orders'][0]['price'] == 10
    assert target['orders'][1]['price'] == 20


def test_update_write_failure_retains_prior_updates():
    class ReadOnly:
        def __init__(self):
            self._d = {'ro': 5}

    register_type(ReadOnly,
                  get=lambda o, k: o._d[k],
                  keys=lambda o: list(o._d),
                  assign=lambda o, k, v: (_ for _ in ()).throw(
                      RuntimeError('read-only!')))

    target = {'leaves': [{'ro': 5}, ReadOnly()]}
    with pytest.raises(PathUpdateError, match='read-only') as exc_info:
        glom(target, Update('leaves.*.ro', lambda v: v + 1))
    assert exc_info.value.updated_count == 1
    # the earlier successful write is retained
    assert target['leaves'][0] == {'ro': 6}


def test_update_validation_and_repr():
    with pytest.raises(TypeError, match='path argument must be'):
        Update(1, _identity)
    with pytest.raises(ValueError, match='path must have at least one element'):
        Update(T, _identity)
    with pytest.raises(ValueError, match='must be setattr or setitem'):
        Update(T['a']('x'), _identity)

    r = repr(Update('a.b', _identity))
    assert r.startswith(f"Update(Path('a', 'b'), <function {_identity.__name__}")
    r = repr(Update('a', _identity, skip_missing=True))
    assert 'skip_missing=True' in r
    r = repr(Update('a', _identity, missing=dict))
    assert r.endswith("missing=dict)")


def test_update_does_not_change_assign_semantics():
    # Assign still computes one value from the whole target and
    # broadcasts it to wildcard matches
    target = {'orders': [{'price': 10}, {'price': 20}]}
    glom(target, Assign('orders.*.price', 999))
    assert [o['price'] for o in target['orders']] == [999, 999]
