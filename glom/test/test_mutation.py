import pytest

from glom import glom, Path, T, S, Spec, Glommer, PathAssignError, PathAccessError
from glom import assign, Assign, delete, Delete, update, Update, PathDeleteError, Or
from glom import core
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


def test_update_basic():
    # each leaf is computed from its own old value, not one shared result
    target = {'orders': [{'price': 100}, {'price': 40}, {'price': 7}]}
    glom(target, Update('orders.*.price', T * 0.5))
    assert target == {'orders': [{'price': 50.0}, {'price': 20.0}, {'price': 3.5}]}

    # mixed dict/list structure, T path, and attribute access
    class Order:
        def __init__(self, price):
            self.price = price

    target = {'orders': [Order(10), Order(20)]}
    glom(target, Update(T['orders'].__star__().price, T + 1))
    assert [o.price for o in target['orders']] == [11, 21]

    # plain callable receives the old value
    target = {'a': {'b': 3}}
    glom(target, Update('a.b', lambda v: v * 10))
    assert target == {'a': {'b': 30}}

    # update() convenience function
    target = {'a': [1, 2]}
    update(target, 'a.1', lambda v: v + 100)
    assert target == {'a': [1, 102]}

    # return value is the target
    target = {'a': 1}
    assert glom(target, Update('a', T + 1)) is target


def test_update_scope_access():
    # call parameters are available through S, original target through S['target']
    target = {'tax_rate': 0.2, 'orders': [{'price': 100}, {'price': 50}]}
    spec = Update('orders.*.price', T * (S['target']['tax_rate'] + 1) * S['discount'])
    glom(target, spec, scope={'discount': 0.5})
    assert target == {'tax_rate': 0.2, 'orders': [{'price': 60.0}, {'price': 30.0}]}

    # S['target'] sees the target Update was evaluated on, even when nested
    target = {'factor': 3, 'items': [{'v': 1}, {'v': 2}]}
    glom(target, (Update('items.*.v', T * S['target']['factor']),))
    assert target == {'factor': 3, 'items': [{'v': 3}, {'v': 6}]}


def test_update_shared_subobject():
    # two paths to the same physical location update only once
    shared = {'price': 10}
    target = {'a': shared, 'b': shared}
    glom(target, Update('*.price', T + 1))
    assert shared == {'price': 11}

    # first-discovery order: the update is applied where first found
    calls = []
    shared = {'v': 0}
    target = {'first': shared, 'second': shared}
    glom(target, Update('*.v', (T, lambda v: calls.append(v) or v + 1)))
    assert calls == [0]
    assert shared == {'v': 1}

    # distinct but equal objects are still updated separately
    target = {'a': {'price': 10}, 'b': {'price': 10}}
    glom(target, Update('*.price', T + 1))
    assert target == {'a': {'price': 11}, 'b': {'price': 11}}


def test_update_missing():
    # missing paths raise by default
    with pytest.raises(PathAccessError):
        glom({}, Update('a.b', T))

    # missing leaf on an existing parent also raises by default
    with pytest.raises(PathAccessError):
        glom({'a': {}}, Update('a.b', T))

    # caller explicitly opts in to creation with missing=
    target = {}
    glom(target, Update('a.b.c', (T, lambda v: (v or 0) + 1), missing=dict))
    assert target == {'a': {'b': {'c': 1}}}

    # absent leaves get None as their old value
    target = {'a': {}}
    glom(target, Update('a.b', (T, lambda v: (v or 0) + 1), missing=dict))
    assert target == {'a': {'b': 1}}

    # existing values still update from their old value when missing is set
    target = {'a': {'b': 3}}
    glom(target, Update('a.b', T * 10, missing=dict))
    assert target == {'a': {'b': 30}}

    # the original target stays reachable through S['target'] during backfill
    target = {'x': 5}
    glom(target, Update('a.b', S['target']['x'] * 2, missing=dict))
    assert target == {'x': 5, 'a': {'b': 10}}

    with pytest.raises(TypeError, match='expected missing to be callable'):
        Update('a', T, missing='dict')


def test_update_registered_type():
    class Box:
        def __init__(self):
            self._data = {}

    glommer = Glommer(register_default_types=False)
    glommer.register(Box,
                     get=lambda box, key: box._data[key],
                     assign=lambda box, key, val: box._data.__setitem__(key, val))

    box = Box()
    box._data['v'] = 4
    glommer.glom(box, Update('v', T * 2))
    assert box._data['v'] == 8

    # unregistered types raise on read, same as other ops
    class Unregistered:
        pass

    with pytest.raises(UnregisteredTarget, match='get'):
        glommer.glom(Unregistered(), Update('v', T))


def test_update_errors():
    # constructor validation, mirroring Assign
    with pytest.raises(TypeError, match='path argument must be'):
        Update(1, 'a')
    with pytest.raises(ValueError, match='path must have at least one element'):
        Update(T, 'a')

    # write failures keep path diagnostics
    class BadTarget:
        a = 1  # readable, but not writable

        def __setattr__(self, name, val):
            raise Exception('nope')

    with pytest.raises(PathAssignError, match='could not assign'):
        glom(BadTarget(), Update('a', T))

    # read failures raise PathAccessError with the full path
    with pytest.raises(PathAccessError, match='could not access'):
        glom({'a': {}}, Update('a.b', T))

    # transform failures propagate, and earlier updates are retained
    def boom(v):
        raise ValueError('bad leaf')

    target = {'items': [{'v': 1}, {'v': 2}]}
    spec = Update('items.*.v', (T, lambda v: v + 1 if v < 2 else boom(v)))
    with pytest.raises(ValueError, match='bad leaf'):
        glom(target, spec)
    assert target == {'items': [{'v': 2}, {'v': 2}]}  # first update kept

    # write failure midway also retains earlier updates
    target = [[1], (2,)]
    with pytest.raises(UnregisteredTarget, match='assign'):
        glom(target, Update('*.0', T + 1))
    assert target[0] == [2]


def test_update_repr():
    assert repr(Update(T.a, T)) == 'Update(T.a, T)'
    assert repr(Update('a.b', 1)) == "Update(Path('a', 'b'), 1)"
    update_spec = Update(T.a, T, missing=dict)
    assert repr(update_spec) == 'Update(T.a, T, missing=dict)'
    assert repr(update_spec) == repr(eval(repr(update_spec)))
