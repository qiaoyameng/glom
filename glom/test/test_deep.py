"""Regression tests for deep declarative evaluation.

Declarative specs (Ref, dict/list/tuple containers, Path, Val, and
sequential pipes) are evaluated iteratively, so recursive specs over
deeply nested data no longer consume Python call stack proportional to
the data depth. Arbitrary user callables are still invoked as before;
glom makes no promise about recursion inside user functions themselves.
"""
import sys

import pytest

from glom import glom, Ref, Val, Pipe, Path, SKIP, STOP, GlomError, PathAccessError


def _make_chain(depth):
    """Build a linked-list-shaped target of the given depth:

    {'val': 0, 'next': [{'val': 1, 'next': [...]}]}
    """
    root = {'val': 0, 'next': []}
    cur = root
    for i in range(1, depth + 1):
        nxt = {'val': i, 'next': []}
        cur['next'] = [nxt]
        cur = nxt
    return root


def _walk_vals(node):
    """Iteratively collect the 'val' keys of a chain-shaped structure."""
    vals = []
    while True:
        vals.append(node['val'])
        if not node['next']:
            return vals
        node = node['next'][0]


def _frame_depth():
    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


# a recursive spec mixing Ref, dict/list/tuple containers, a string
# path, a Val, and a sequential pipe
CHAIN_SPEC = Ref('node', {
    'val': 'val',
    'const': Val('constant'),
    'next': ('next', [Ref('node')]),
})


def test_deep_chain_matches_shallow():
    # the same recursive spec produces corresponding results at any depth
    for depth in (0, 1, 5, 50):
        result = glom(_make_chain(depth), CHAIN_SPEC)
        vals = []
        node = result
        while True:
            vals.append(node['val'])
            assert node['const'] == 'constant'
            if not node['next']:
                break
            node = node['next'][0]
        assert vals == list(range(depth + 1))


def test_deep_eval_does_not_grow_python_stack():
    # Prove evaluation is iterative: with the recursion limit lowered
    # to just above the current call depth, an evaluator which consumed
    # Python stack proportional to the data depth would fail, while an
    # iterative one handles thousands of levels. This is a structural
    # assertion; no timing thresholds are involved.
    depth = 3000
    target = _make_chain(depth)
    assert _walk_vals(glom(target, CHAIN_SPEC)) == list(range(depth + 1))

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(_frame_depth() + 200)
    try:
        result = glom(target, CHAIN_SPEC)
    finally:
        sys.setrecursionlimit(old_limit)
    assert _walk_vals(result) == list(range(depth + 1))


def test_nested_ref_shadowing():
    # an inner Ref definition shadows same-named outer definitions
    # within its own subtree, and only there
    tree = {'name': 'root', 'children': [
        {'name': 'a', 'children': []},
        {'name': 'b', 'children': [{'name': 'c', 'children': []}]},
    ]}
    spec = Ref('n', {
        'kid_names': ('children', [Ref('n', 'name')]),  # inner 'n' := 'name'
        'name': 'name',
        'kids': ('children', [Ref('n')]),  # must still see the outer 'n'
    })
    assert glom(tree, spec) == {
        'kid_names': ['a', 'b'],
        'name': 'root',
        'kids': [
            {'kid_names': [], 'name': 'a', 'kids': []},
            {'kid_names': ['c'], 'name': 'b', 'kids': [
                {'kid_names': [], 'name': 'c', 'kids': []}]},
        ],
    }


def test_deep_nested_ref_shadowing():
    depth = 3000
    target = {'name': 'n0', 'children': []}
    cur = target
    for i in range(1, depth):
        nxt = {'name': 'n%d' % i, 'children': []}
        cur['children'] = [nxt]
        cur = nxt
    spec = Ref('n', {
        'kid_names': ('children', [Ref('n', 'name')]),
        'name': 'name',
        'kids': ('children', [Ref('n')]),
    })
    node = glom(target, spec)
    i = 0
    while node['kids']:
        assert node['name'] == 'n%d' % i
        # if the inner definition leaked, these would be dicts, not names
        assert node['kid_names'] == ['n%d' % (i + 1)]
        node = node['kids'][0]
        i += 1
    assert node['name'] == 'n%d' % (depth - 1)
    assert node['kid_names'] == []


def test_deep_skip_and_stop():
    depth = 3000
    target = _make_chain(depth)
    spec = Ref('node', {
        'val': lambda t: SKIP if t['val'] == depth else t['val'],
        'next': ('next', [Ref('node')]),
    })
    node = glom(target, spec)
    count = 0
    while node['next']:
        assert 'val' in node
        count += 1
        node = node['next'][0]
    assert 'val' not in node  # deepest level's val was SKIPped
    assert count == depth

    # STOP still halts list iteration and tuple pipes
    assert glom([1, 2, 3, 4], [lambda x: STOP if x > 2 else x]) == [1, 2]
    assert glom(range(10), (list, [lambda x: x if x < 5 else STOP])) == [0, 1, 2, 3, 4]


def test_deep_pipe_and_path():
    depth = 3000
    target = _make_chain(depth)
    # Pipe composed with Ref recursion
    spec = Ref('node', Pipe('next', [Ref('node')]))
    result = glom(target, spec)
    nest_depth = 0
    while result:
        nest_depth += 1
        result = result[0]
    assert nest_depth == depth
    # an inner Ref definition shadows the outer one inside a Pipe, too
    spec = Ref('node', Pipe('next', [Ref('node', Pipe('val'))]))
    assert glom(target, spec) == [1]
    # explicit Path objects at depth
    spec = Ref('node', {'v': Path('val'), 'next': ('next', [Ref('node')])})
    result = glom(_make_chain(3), spec)
    assert result['next'][0]['next'][0]['next'][0]['v'] == 3
    # a long flat pipe of steps
    steps = []
    for _ in range(depth):
        steps.extend(['next', '0'])
    steps.append('val')
    assert glom(target, tuple(steps)) == depth
    assert glom(target, Pipe(*steps)) == depth


def test_deep_error_is_bounded_and_readable():
    depth = 3000
    target = _make_chain(depth)
    node = target
    while node['next']:
        node = node['next'][0]
    del node['val']  # missing field at the deepest level

    with pytest.raises(PathAccessError) as exc_info:
        glom(target, CHAIN_SPEC)
    message = str(exc_info.value)
    # the trace is elided in the middle instead of listing every frame
    assert len(message.splitlines()) < 200
    assert 'elided' in message
    # ...while the root cause stays front and center
    assert "could not access 'val'" in message


def test_user_callable_still_called_at_depth():
    depth = 3000
    target = _make_chain(depth)
    calls = []

    def record(val):
        calls.append(val)
        return val * 2

    spec = Ref('node', {'doubled': ('val', record), 'next': ('next', [Ref('node')])})
    result = glom(target, spec)
    assert calls == list(range(depth + 1))
    node = result
    expected = 0
    while True:
        assert node['doubled'] == expected * 2
        if not node['next']:
            break
        expected += 1
        node = node['next'][0]


def test_ref_definitions_do_not_leak_between_evaluations():
    assert glom({'a': 1}, Ref('x', 'a')) == 1
    # a fresh evaluation has no 'x' defined
    with pytest.raises(GlomError):
        glom({'a': 1}, Ref('x'))
    # a failed evaluation cleans up its definitions, too
    with pytest.raises(GlomError):
        glom({}, Ref('x', ('missing', Ref('x', 'a'))))
    assert glom({'a': 5}, Ref('x', 'a')) == 5
