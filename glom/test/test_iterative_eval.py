"""Tests for stack-safe evaluation of declarative specs.

Declarative specs (Ref, dict, list, tuple, str, TType, Val, Path, and
Pipe) are evaluated without growing the Python call stack, so data
structures nested thousands of levels deep no longer hit
RecursionError. These tests prove stack usage is independent of
nesting depth *without* relying on timing thresholds: depths are
multiples of the interpreter recursion limit, and one test evaluates
a structure far deeper than the (temporarily lowered) limit itself.

"""
import sys

import pytest

from glom import glom, Ref, Val, Path, Pipe, T, S, Spec, SKIP, STOP
from glom import GlomError, PathAccessError


def _nested_list(depth):
    """[[[...[]...]]] with *depth* levels, built without recursion."""
    ret = []
    for _ in range(depth):
        ret = [ret]
    return ret


def _list_depth(val):
    depth = 0
    while val:
        assert type(val) is list and len(val) == 1
        val = val[0]
        depth += 1
    return depth


def _chain_target(depth, leaf_marker=True):
    """{'next': {'next': ... {'leaf': ...}}} chain of dicts."""
    ret = cur = {}
    for _ in range(depth):
        cur['next'] = {}
        cur = cur['next']
    if leaf_marker:
        cur['leaf'] = 'bottom'
    return ret


def _nested_tuple_spec(depth, leaf_spec):
    """('next', ('next', ... leaf_spec)) pipe chain."""
    spec = leaf_spec
    for _ in range(depth):
        spec = ('next', spec)
    return spec


def test_deep_ref_recursion():
    # the canonical recursive-Ref spec; depth is a multiple of the
    # recursion limit, so any per-level Python stack usage would fail
    depth = sys.getrecursionlimit() * 3
    result = glom(_nested_list(depth), Ref('item', [Ref('item')]))
    assert _list_depth(result) == depth


def test_deep_eval_does_not_consume_python_stack():
    # direct proof that evaluation does not grow the Python stack with
    # data depth: with the recursion limit lowered far below the data
    # depth, evaluation still succeeds
    depth = 2000
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(500)
    try:
        result = glom(_nested_list(depth), Ref('item', [Ref('item')]))
    finally:
        sys.setrecursionlimit(old_limit)
    assert _list_depth(result) == depth


def test_deep_tuple_pipe():
    depth = sys.getrecursionlimit() * 2
    target = _chain_target(depth)
    assert glom(target, _nested_tuple_spec(depth, 'leaf')) == 'bottom'


def test_deep_pipe_spec():
    depth = 2000
    spec = 'leaf'
    for _ in range(depth):
        spec = Pipe('next', spec)
    assert glom(_chain_target(depth), spec) == 'bottom'


def test_deep_dict_spec():
    depth = 2000
    spec = {'leaf': 'leaf'}
    for _ in range(depth):
        spec = {'result': ('next', spec)}
    result = glom(_chain_target(depth), spec)
    cur = result
    for _ in range(depth):
        cur = cur['result']
    assert cur == {'leaf': 'bottom'}


def test_deep_mixed_declarative_spec():
    # Ref + dict + list + tuple + Val + Path in a single recursive spec
    depth = 2000
    target = {'v': 0, 'kids': []}
    for i in range(1, depth):
        target = {'v': i, 'kids': [target]}
    spec = Ref('node', {'v': 'v',
                        'tag': Val('n'),
                        'p': Path('v'),
                        'kids': ('kids', [Ref('node')])})
    result = glom(target, spec)
    cur = result
    for i in range(depth - 1, -1, -1):
        assert cur['v'] == i
        assert cur['tag'] == 'n'
        assert cur['p'] == i
        if i:
            assert len(cur['kids']) == 1
            cur = cur['kids'][0]
        else:
            assert cur['kids'] == []


def test_deep_error_is_readable_and_bounded():
    # a missing field thousands of levels down should produce a
    # readable, finite PathAccessError, not a RecursionError
    depth = 2000
    target = _chain_target(depth, leaf_marker=False)
    spec = _nested_tuple_spec(depth, 'missing')
    with pytest.raises(PathAccessError) as exc_info:
        glom(target, spec)
    msg = str(exc_info.value)
    assert "could not access 'missing'" in msg
    assert 'frames elided' in msg
    assert len(msg.splitlines()) < 200


def test_ref_lexical_scoping():
    # an inner Ref definition shadows the outer one within its own
    # subtree (if the outer def leaked in, the inner Ref('x') would
    # evaluate the outer dict spec against lists and fail)
    target = {'tag': 'root', 'sub': [[[]]]}
    spec = Ref('x', {'tag': 'tag', 'nested': ('sub', Ref('x', [Ref('x')]))})
    assert glom(target, spec) == {'tag': 'root', 'nested': [[[]]]}


def test_ref_definition_does_not_leak():
    # a Ref definition is visible within its own subtree, not to
    # siblings (if defs leaked globally, 'item' below would resolve to
    # Val('shadow') instead of recursing)
    spec = Ref('x', [{'sh': Ref('x', Val('shadow')), 'item': Ref('x')}])
    assert glom([[[]]], spec) == [{'sh': 'shadow', 'item': [{'sh': 'shadow', 'item': []}]}]
    # sibling branches cannot see each other's definitions at all
    with pytest.raises(GlomError):
        glom({'a': 1, 'b': 2},
             {'first': ('a', Ref('x', Val('defined'))), 'second': ('b', Ref('x'))})


def test_ref_definition_visible_down_pipe():
    # tuple steps chain scopes, so a Ref defined in an earlier step is
    # visible to later steps of the same pipe
    assert glom({'v': 1}, (Ref('x', Val('v')), Ref('x'))) == 'v'


def test_declarative_equivalence():
    # same spec, same result as the recursive evaluator, at everyday
    # depths (values recorded from the original executor)
    assert glom({'a': 1, 'b': 2}, {'x': 'a', 'y': 'b'}) == {'x': 1, 'y': 2}
    assert glom({'a': {'b': {'c': 5}}}, {'x': {'y': 'a.b'}}) == {'x': {'y': {'c': 5}}}
    assert glom([1, 2, 3], [lambda t: t * 2]) == [2, 4, 6]
    assert glom([1, 2, 3, 4], [lambda t: SKIP if t % 2 else t]) == [2, 4]
    assert glom(range(10), [lambda x: x if x < 5 else STOP]) == [0, 1, 2, 3, 4]
    assert glom({'a': {'b': -5}}, ('a', 'b', abs)) == 5
    assert glom(1, (lambda t: SKIP, lambda t: 99)) == 99
    assert glom(1, (lambda t: STOP, lambda t: 99)) == 1
    assert glom({'a': {'b': -5}}, Pipe('a', 'b', abs)) == 5
    assert glom(7, Pipe()) == 7
    assert glom(7, ()) == 7
    assert glom({}, {'v': Val({'k': [1, 2]})}) == {'v': {'k': [1, 2]}}
    assert glom({'a': {'b': 'c'}}, Path('a', 'b')) == 'c'
    assert glom({'a': {'b': 'c'}}, 'a.b') == 'c'
    assert glom([[[]]], Ref('item', [Ref('item')])) == [[[]]]
    assert glom({'a': 'k', 'k': 1}, {Spec('a'): 'k'}) == {'k': 1}
    assert glom({'a': 'k', 'k': 1}, {T['a']: 'k'}) == {'k': 1}
    assert glom({'a': 1}, {'a': lambda t: SKIP, 'b': Val(2)}) == {'b': 2}
    assert glom({'a': {'b': 'c', 'd': 'e'}, 'h': [0, 1, 2]},
                {'a': 'a.b', 'd': 'a.d', 'h': ('h', [lambda x: x * 2])}) \
        == {'a': 'c', 'd': 'e', 'h': [0, 2, 4]}
    assert glom('tgt', (Val(5), lambda t: t + 1)) == 6
    assert glom({'a': {'b': 1}}, {'p': Path('a', 'b'), 'v': Val('x')}) == {'p': 1, 'v': 'x'}
    assert glom({'a': 3}, T['a'] + 1) == 4
    assert glom({'a': 'x'}, T['a'].upper()) == 'X'
    assert glom({'a': {'x': 1}, 'b': {'x': 2}}, T.values().__star__()['x']) == [1, 2]
    assert glom({'data': {'val': 9}}, (S(value=T['data']['val']), {'val': S['value']})) == {'val': 9}
    # arbitrary user callables are still called (their own recursion
    # remains their own responsibility)
    assert glom(5, lambda t: t + 1) == 6
    # etree-style Ref recursion over dicts
    tree = {'tag': 'html', 'kids': [{'tag': 'head', 'kids': []},
                                    {'tag': 'body', 'kids': [{'tag': 'p', 'kids': []}]}]}
    assert glom(tree, Ref('el', {'tag': 'tag', 'kids': ('kids', [Ref('el')])})) == tree
    # error behavior is preserved, too
    with pytest.raises(GlomError):
        glom([1], [])
    with pytest.raises(PathAccessError):
        glom({'a': {'b': {}}}, ('a', 'b', 'c'))
