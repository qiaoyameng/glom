"""spill-to-disk grouping mode.

:class:`SpillGroup` mirrors :class:`~glom.grouping.Group`, but bounds
the number of *active* (in-memory) groups to a shared budget. Once the
budget is exceeded, the least-recently-used group's accumulator state
is encoded through an explicit protocol and staged in a local
:mod:`sqlite3` database. When that group shows up again its state is
decoded and swapped back into memory.

Only aggregators which implement the spill encoding protocol
(:class:`SpillableAgg`) are supported; arbitrary objects are never
pickled. The first version covers :class:`~glom.reduction.Count`,
:class:`~glom.reduction.Sum`, :class:`~glom.grouping.Min` and
:class:`~glom.grouping.Max`, plus their nested dict/list combinations.
"""

import json
import os
import sqlite3
import tempfile
from collections import OrderedDict

from boltons.typeutils import make_sentinel

from .core import glom, T, SKIP, STOP, MODE, MIN_MODE, AUTO, BadSpec
from .grouping import ACC_TREE, CUR_AGG, target_iter


SPILL_RUNNER = make_sentinel('SPILL_RUNNER')
SPILL_RUNNER.__doc__ = """
the active :class:`_SpillRunner`, stashed in the Scope so the SPILL_MODE
dispatcher can reach its per-record recursion.
"""

SPILL_PATH = make_sentinel('SPILL_PATH')
SPILL_PATH.__doc__ = """
the bucket path currently being processed by the SPILL_MODE dispatcher.
"""


def SPILL(target, spec, scope):
    """SPILL_MODE dispatcher; delegates to the active runner."""
    runner = scope[SPILL_RUNNER]
    return runner.dispatch(target, spec, scope,
                           scope[SPILL_PATH], scope[ACC_TREE])


SPILL_MODE = SPILL  # callable mode dispatcher, like GROUP/AUTO
SPILL_MODE.__doc__ = """
mode placed in the Scope under MODE while a SpillGroup is running;
spec dispatch delegates every nested dict/list/aggregator back to the
active :class:`_SpillRunner` so spill bookkeeping stays consistent.
"""

# runtime tree bookkeeping slots
_KEYSPEC_MAP = '__keyspecs__'   # id(keyspec) -> (keyspec, valspec)
_KEY_INDEX = '__key_index__'    # key -> id(keyspec)
_STOPPED = '__stopped__'        # set of id(keyspec) which returned STOP
_LIST_SUBS = '__list_subs__'    # list node: per-element persistent state trees
_EMITTED = '__emitted__'        # dict node: keys that produced non-SKIP output

_SCALAR_TYPES = (str, int, float, bool, type(None))


class SpillError(BadSpec):
    """Raised when a spec cannot run under spill mode -- e.g. an
    aggregator without an explicit encoding protocol, or an
    unsupported group key."""


class SpillableAgg:
    """Base class for aggregators that know how to spill.

    A spillable aggregator carries four pieces:

    * ``agg(target, tree)`` -- same contract as a normal Group
      aggregator, mutating the live accumulator held in *tree*;
    * :meth:`encode_state` -- turn the live accumulator into a
      JSON-friendly protocol payload;
    * :meth:`restore_state` -- rebuild the live accumulator from such a
      payload and place it back in *tree*;
    * :meth:`finalize` -- derive the final output value from the live
      accumulator (defaults to returning it unchanged).

    The encoded payload is the only thing written to disk, and it must
    be JSON-friendly. Arbitrary objects are never pickled. Subclasses
    must be hashable (they key their accumulator by themselves in the
    accumulator tree, like the built-in aggs).
    """

    state_type = None

    def agg(self, target, tree):  # pragma: no cover - interface
        raise NotImplementedError

    def encode_state(self, tree):  # pragma: no cover - interface
        raise NotImplementedError

    def restore_state(self, payload, tree):  # pragma: no cover - interface
        raise NotImplementedError

    def finalize(self, tree):
        """derive the final result from the live accumulator."""
        return tree.get(self)


class SpillCount(SpillableAgg):
    """spillable counterpart of :class:`~glom.reduction.Count`

    >>> glom([1, 2, 3], SpillGroup(SpillCount()))
    3
    """

    state_type = 'count'

    def agg(self, target, tree):
        tree[self] = tree.get(self, 0) + 1
        return tree[self]

    def encode_state(self, tree):
        return tree.get(self, 0)

    def restore_state(self, payload, tree):
        tree[self] = payload

    def __repr__(self):
        return '%s()' % self.__class__.__name__


class SpillSum(SpillableAgg):
    """spillable counterpart of :class:`~glom.reduction.Sum`

    >>> glom([1, 2, 3], SpillGroup(SpillSum()))
    6
    """

    state_type = 'sum'

    def agg(self, target, tree):
        tree[self] = tree.get(self, 0) + target
        return tree[self]

    def encode_state(self, tree):
        return tree.get(self, 0)

    def restore_state(self, payload, tree):
        tree[self] = payload

    def __repr__(self):
        return '%s()' % self.__class__.__name__


class SpillMin(SpillableAgg):
    """spillable counterpart of :class:`~glom.grouping.Min`

    >>> glom([3, 1, 2], SpillGroup(SpillMin()))
    1
    """

    state_type = 'min'

    def agg(self, target, tree):
        if self not in tree or target < tree[self]:
            tree[self] = target
        return tree[self]

    def encode_state(self, tree):
        return tree[self] if self in tree else None

    def restore_state(self, payload, tree):
        if payload is not None:
            tree[self] = payload

    def __repr__(self):
        return '%s()' % self.__class__.__name__


class SpillMax(SpillableAgg):
    """spillable counterpart of :class:`~glom.grouping.Max`

    >>> glom([3, 1, 2], SpillGroup(SpillMax()))
    3
    """

    state_type = 'max'

    def agg(self, target, tree):
        if self not in tree or target > tree[self]:
            tree[self] = target
        return tree[self]

    def encode_state(self, tree):
        return tree[self] if self in tree else None

    def restore_state(self, payload, tree):
        if payload is not None:
            tree[self] = payload

    def __repr__(self):
        return '%s()' % self.__class__.__name__


# explicit state encoding protocol -------------------------------------
#
# A group's live accumulator tree is encoded into JSON made only of
# tagged arrays, so the on-disk representation is self-describing and
# never relies on pickle or object identity:
#
#   leaf agg : ["agg", type_tag, payload]        (payload may be null)
#   dict node: ["dict", [[key, keyspec_id, child], ...], [stopped_ids]]
#   list node: ["list", [child, ...]]
#
# keyspec_id is id(keyspec), stable within one SpillGroup run because
# spec objects are not rebuilt per record.

def _is_unsupported_agg(spec):
    """a Group-style aggregator (defines ``agg``) that is not a
    :class:`SpillableAgg` and so has no encoding protocol.

    Callable built-in specs (e.g. T-expressions) also carry an ``agg``
    hook but are ordinary value specs, not aggregators, so they are
    excluded and handled as raw values.
    """
    return (not isinstance(spec, SpillableAgg)
            and not callable(spec)
            and callable(getattr(spec, 'agg', None)))


def encode_state(spec, tree):
    """encode the live accumulator *tree* of *spec* into protocol form."""
    if isinstance(spec, SpillableAgg):
        payload = spec.encode_state(tree) if spec in tree else None
        return ['agg', spec.state_type, payload]
    if _is_unsupported_agg(spec):
        raise SpillError(
            'aggregator %r has no spill encoding protocol; subclass'
            ' SpillableAgg and implement encode_state/restore_state'
            % (spec,))
    if callable(spec):
        return ['raw', tree.get(id(spec))]
    spec_type = type(spec)
    if spec_type is dict:
        entries = []
        for key, sub_tree in tree[id(spec)].items():
            keyspec_id = tree[_KEY_INDEX][key]
            keyspec, valspec = tree[_KEYSPEC_MAP][keyspec_id]
            entries.append([key, keyspec_id, encode_state(valspec, sub_tree)])
        stopped_ids = sorted(tree.get(_STOPPED, set()))
        emitted_keys = [key for key in tree[id(spec)]
                        if key in tree.get(_EMITTED, set())]
        return ['dict', entries, stopped_ids, emitted_keys]
    if spec_type is list:
        # a list node accumulates per-record values, plus persistent
        # per-element aggregator state in parallel sub-trees
        subs = []
        for valspec, sub_tree in zip(spec, tree[_LIST_SUBS]):
            if isinstance(valspec, SpillableAgg) or callable(valspec):
                subs.append(encode_state(valspec, sub_tree))
            else:
                subs.append(None)
        return ['list', list(tree[id(spec)]), subs]
    raise SpillError('unsupported spec in spill mode: %r' % (spec,))


def restore_state(spec, state, tree):
    """rebuild the live accumulator *tree* of *spec* from protocol form."""
    tag = state[0]
    if tag == 'raw':
        if not callable(spec) or isinstance(spec, SpillableAgg):
            raise SpillError('spill raw state does not match spec %r' % (spec,))
        _, value = state
        if value is not None:
            tree[id(spec)] = value
        return
    if tag == 'agg':
        _, type_tag, payload = state
        if not isinstance(spec, SpillableAgg) or spec.state_type != type_tag:
            raise SpillError('spill state %r does not match spec %r'
                             % (type_tag, spec))
        if payload is not None:
            spec.restore_state(payload, tree)
        return
    spec_type = type(spec)
    if tag == 'dict':
        if spec_type is not dict:
            raise SpillError('spill dict state does not match spec %r' % (spec,))
        _, entries, stopped_ids, emitted_keys = state
        acc = tree.setdefault(id(spec), OrderedDict())
        key_index = tree.setdefault(_KEY_INDEX, {})
        keyspec_map = tree.setdefault(_KEYSPEC_MAP, {})
        for keyspec, valspec in spec.items():
            keyspec_map.setdefault(id(keyspec), (keyspec, valspec))
        tree.setdefault(_STOPPED, set()).update(stopped_ids)
        tree.setdefault(_EMITTED, set()).update(emitted_keys)
        for key, keyspec_id, child_state in entries:
            keyspec, valspec = keyspec_map[keyspec_id]
            sub_tree = {}
            restore_state(valspec, child_state, sub_tree)
            acc[key] = sub_tree
            key_index[key] = keyspec_id
        return
    if tag == 'list':
        if spec_type is not list:
            raise SpillError('spill list state does not match spec %r' % (spec,))
        _, values, sub_states = state
        tree[id(spec)] = list(values)
        sub_trees = []
        for valspec, sub_state in zip(spec, sub_states):
            sub_tree = {}
            if sub_state is not None:
                restore_state(valspec, sub_state, sub_tree)
            sub_trees.append(sub_tree)
        tree[_LIST_SUBS] = sub_trees
        return
    raise SpillError('unknown spill state tag: %r' % (tag,))


def dumps_state(spec, tree):
    return json.dumps(encode_state(spec, tree), separators=(',', ':'),
                      ensure_ascii=False)


def loads_state(spec, text):
    tree = {}
    restore_state(spec, json.loads(text), tree)
    return tree


class _SpillStore:
    """per-run staging area backed by a local SQLite database.

    The database lives in a private temp file (WAL journal) and is
    removed on :meth:`close`. No extra database process is involved --
    this is just the stdlib embedded SQLite engine.
    """

    def __init__(self, path=None):
        if path is None:
            fd, path = tempfile.mkstemp(prefix='glom-spill-', suffix='.db')
            os.close(fd)
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS spill ('
            '  bucket_id TEXT PRIMARY KEY,'
            '  state TEXT NOT NULL'
            ')')
        self._closed = False

    def save(self, bucket_id, state_text):
        self._conn.execute(
            'INSERT OR REPLACE INTO spill (bucket_id, state) VALUES (?, ?)',
            (bucket_id, state_text))
        self._conn.commit()

    def load(self, bucket_id):
        cur = self._conn.execute(
            'SELECT state FROM spill WHERE bucket_id = ?', (bucket_id,))
        row = cur.fetchone()
        return row[0] if row is not None else None

    def delete(self, bucket_id):
        self._conn.execute(
            'DELETE FROM spill WHERE bucket_id = ?', (bucket_id,))
        self._conn.commit()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        finally:
            for suffix in ('', '-wal', '-shm'):
                try:
                    os.remove(self.path + suffix)
                except OSError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


_SPILLED = make_sentinel('_SPILLED')
_SPILLED.__doc__ = """
placeholder left in the reference tree where a group's live state has
been staged to the spill store; replaced by the restored tree on load.
"""


def _check_key(key):
    """validate a group key; keys must be scalars or tuples of scalars."""
    if isinstance(key, _SCALAR_TYPES):
        return
    if isinstance(key, tuple):
        for item in key:
            if not isinstance(item, _SCALAR_TYPES):
                raise SpillError('unsupported group key component: %r' % (item,))
        return
    raise SpillError('unsupported group key: %r' % (key,))


def _key_text(key):
    return json.dumps(key, separators=(',', ':'), ensure_ascii=False)


class _SpillRunner:
    """per-run engine; owns the accumulator tree, LRU budget and store.

    A single live accumulator tree doubles as the reference tree: every
    dict bucket maps a key to either its live sub-tree or to the
    :data:`_SPILLED` placeholder. All mutable state is per-run, so
    exceptions, early STOP and an explicit close() always clean up this
    run's staging files.
    """

    def __init__(self, spec, budget, store):
        self.spec = spec
        self.budget = budget
        self.store = store
        self.tree = {}
        # LRU order of resident groups: bucket_id -> None
        self.lru = OrderedDict()
        self._current_path = ()
        self.peak_active = 0
        self.spill_count = 0
        self.load_count = 0
        self._validate(spec)

    # -- validation ----------------------------------------------------

    def _validate(self, spec):
        if isinstance(spec, SpillableAgg):
            return
        if _is_unsupported_agg(spec):
            raise SpillError(
                'aggregator %r has no spill encoding protocol; subclass'
                ' SpillableAgg and implement encode_state/restore_state'
                % (spec,))
        if callable(spec):
            return
        spec_type = type(spec)
        if spec_type is dict:
            for valspec in spec.values():
                self._validate(valspec)
            return
        if spec_type is list:
            for valspec in spec:
                if type(valspec) is dict:
                    raise BadSpec('dicts within lists are not allowed'
                                  ' while in Group mode: %r' % spec)
                self._validate(valspec)
            return
        raise BadSpec('Group mode expected dict, list, callable, or'
                      ' aggregator, not: %r' % (spec,))

    # -- node helpers ---------------------------------------------------

    def _valspec_for(self, dict_spec, keyspec_id):
        for keyspec, valspec in dict_spec.items():
            if id(keyspec) == keyspec_id:
                return valspec
        raise SpillError('keyspec not found in spec: %r' % (keyspec_id,))

    def _parent_of(self, path):
        """return ``(parent_spec, parent_node)`` for the bucket at *path*.

        *parent_node* is the live tree of the dict node holding the
        bucket. Ancestor groups are guaranteed resident.
        """
        spec = self.spec
        node = self.tree
        for key in path[:-1]:
            keyspec_id = node[_KEY_INDEX][key]
            valspec = self._valspec_for(spec, keyspec_id)
            bucket = node[id(spec)][key]
            if bucket is _SPILLED:
                raise SpillError('internal: ancestor group not resident')
            spec, node = valspec, bucket
        return spec, node

    # -- LRU / spill ---------------------------------------------------

    @property
    def active_group_count(self):
        return len(self.lru)

    def _touch(self, path, key):
        entry = tuple(path) + (key,)
        self._current_path = entry
        if entry in self.lru:
            self.lru.move_to_end(entry)
            return
        # make room before admitting the new group so the resident
        # count never exceeds the budget (except for nested groups on
        # the record's current path, which cannot be evicted)
        self._enforce_budget(admitting=entry)
        self.lru[entry] = None
        if len(self.lru) > self.peak_active:
            self.peak_active = len(self.lru)

    @staticmethod
    def _bucket_id(path):
        # SQLite row key: stable text form of the bucket path; tuple keys
        # round-trip only here, never back into Python object identity
        return _key_text(list(path))

    def _is_protected(self, path):
        """buckets on the record's current path must stay resident."""
        cur = self._current_path
        return len(path) <= len(cur) and cur[:len(path)] == path

    @staticmethod
    def _is_ancestor(ancestor_path, path):
        return (len(ancestor_path) < len(path)
                and path[:len(ancestor_path)] == ancestor_path)

    def _enforce_budget(self, admitting=None):
        # while admitting, free a slot once the LRU is already full;
        # otherwise trim any entries that exceed the budget
        limit = self.budget - 1 if admitting is not None else self.budget
        while len(self.lru) > limit:
            resident_paths = list(self.lru)
            victim = None
            for path in resident_paths:
                if self._is_protected(path):
                    continue
                # only pick a group whose resident subtree is closed:
                # no descendant (and no ancestor) is also resident
                nested = any(p != path and (
                    self._is_ancestor(path, p) or self._is_ancestor(p, path))
                    for p in resident_paths)
                if not nested:
                    victim = path
                    break
            if victim is None:
                return  # every resident group is needed by this record
            del self.lru[victim]
            parent_path, key = victim[:-1], victim[-1]
            parent_spec, parent = self._parent_of(parent_path)
            buckets = parent[id(parent_spec)]
            live_tree = buckets[key]
            if live_tree is _SPILLED:
                continue  # stale entry; bucket already spilled
            keyspec_id = parent[_KEY_INDEX][key]
            bucket_spec = self._valspec_for(parent_spec, keyspec_id)
            self.store.save(self._bucket_id(victim),
                            dumps_state(bucket_spec, live_tree))
            buckets[key] = _SPILLED
            self.spill_count += 1

    def _load(self, full_path, key):
        # *full_path* already includes *key* as its last element
        parent_spec, parent = self._parent_of(full_path)
        keyspec_id = parent[_KEY_INDEX][key]
        bucket_spec = self._valspec_for(parent_spec, keyspec_id)
        bucket_id = self._bucket_id(full_path)
        state_text = self.store.load(bucket_id)
        if state_text is None:
            raise SpillError('spill state missing for key %r' % (key,))
        live_tree = loads_state(bucket_spec, state_text)
        self.store.delete(bucket_id)
        parent[id(parent_spec)][key] = live_tree
        self.load_count += 1
        return live_tree

    # -- per-record processing ----------------------------------------

    def process(self, target, scope):
        scope[SPILL_RUNNER] = self
        scope[SPILL_PATH] = ()
        scope[ACC_TREE] = self.tree
        return self.dispatch(target, self.spec, scope, (), self.tree)

    @staticmethod
    def _is_structural(spec):
        """specs that participate in the accumulator tree (and so must go
        through SPILL dispatch) rather than being plain value specs."""
        return isinstance(spec, SpillableAgg) or type(spec) in (dict, list)

    @staticmethod
    def _eval_auto(target, spec, scope):
        """evaluate a key/value spec outside SPILL dispatch.

        Temporarily switches to AUTO mode so T-expressions, plain
        callables and other non-aggregating specs resolve exactly as
        they would outside Group mode, against the current *target*.
        """
        scope[MIN_MODE] = AUTO
        try:
            return scope[glom](target, spec, scope)
        finally:
            scope[MIN_MODE] = None

    def dispatch(self, target, spec, scope, path, tree):
        """mirror glom GROUP dispatch, with explicit spill handling."""
        if isinstance(spec, SpillableAgg):
            return spec.agg(target, tree)
        if _is_unsupported_agg(spec):
            raise SpillError(
                'aggregator %r has no spill encoding protocol; subclass'
                ' SpillableAgg and implement encode_state/restore_state'
                % (spec,))
        if callable(spec):
            value = self._eval_auto(target, spec, scope)
            if value is not SKIP:
                tree[id(spec)] = value  # last value wins, like ordinary Group
            return value
        spec_type = type(spec)
        if spec_type not in (dict, list):
            raise BadSpec('Group mode expected dict, list, callable, or'
                          ' aggregator, not: %r' % (spec,))

        spec_id = id(spec)
        if spec_type is dict:
            keyspec_map = tree.setdefault(_KEYSPEC_MAP, {})
            for keyspec, valspec in spec.items():
                keyspec_map.setdefault(id(keyspec), (keyspec, valspec))
            key_index = tree.setdefault(_KEY_INDEX, {})
            stopped = tree.setdefault(_STOPPED, set())
            emitted = tree.setdefault(_EMITTED, set())
            acc = tree.setdefault(spec_id, OrderedDict())
            done = True
            for keyspec, valspec in spec.items():
                if id(keyspec) in stopped:
                    continue
                key = self._eval_auto(target, keyspec, scope)
                if key is SKIP:
                    done = False
                    continue
                if key is STOP:
                    stopped.add(id(keyspec))
                    continue
                _check_key(key)
                child_path = path + (key,)
                if key not in acc:
                    acc[key] = {}
                    key_index[key] = id(keyspec)
                if acc[key] is _SPILLED:
                    acc[key] = self._load(child_path, key)
                sub_tree = acc[key]
                self._touch(path, key)
                scope[ACC_TREE] = sub_tree
                scope[SPILL_PATH] = child_path
                if self._is_structural(valspec):
                    result = scope[glom](target, valspec, scope)
                else:
                    result = self._eval_auto(target, valspec, scope)
                    if result is not SKIP:
                        # last value wins, like ordinary Group
                        sub_tree[id(valspec)] = result
                if result is STOP:
                    stopped.add(id(keyspec))
                    continue
                done = False
                if result is not SKIP:
                    emitted.add(key)
            if done:
                return STOP
            return acc

        # list node: each element's per-record result is appended, mirroring
        # ordinary Group mode; SKIP skips that element, STOP halts the node.
        # aggregator leaves keep persistent state in a parallel sub-tree.
        acc_list = tree.get(spec_id)
        if acc_list is None:
            acc_list = tree[spec_id] = []
            tree[_LIST_SUBS] = [{} for _ in spec]
        sub_trees = tree[_LIST_SUBS]
        for idx, valspec in enumerate(spec):
            if type(valspec) is dict:
                raise BadSpec('dicts within lists are not allowed'
                              ' while in Group mode: %r' % spec)
            scope[ACC_TREE] = sub_trees[idx]
            scope[SPILL_PATH] = path
            if self._is_structural(valspec):
                result = scope[glom](target, valspec, scope)
            else:
                result = self._eval_auto(target, valspec, scope)
            if result is STOP:
                return STOP
            if result is not SKIP:
                acc_list.append(result)
        return acc_list

    # -- result assembly ------------------------------------------------

    def result(self):
        self._restore_all()
        return self._materialize(self.spec, self.tree)

    def _restore_all(self):
        """swap every still-spilled bucket back before materialization."""
        self._restore_node(self.spec, self.tree, ())

    def _restore_node(self, spec, tree, path):
        spec_type = type(spec)
        if spec_type is dict:
            if id(spec) not in tree:
                return  # no record ever reached this node
            buckets = tree[id(spec)]
            for key in list(buckets.keys()):
                child_path = path + (key,)
                if buckets[key] is _SPILLED:
                    buckets[key] = self._load(child_path, key)
                valspec = self._valspec_for(spec, tree[_KEY_INDEX][key])
                self._restore_node(valspec, buckets[key], child_path)

    def _materialize(self, spec, tree):
        if isinstance(spec, SpillableAgg):
            return spec.finalize(tree)
        if callable(spec):
            return tree.get(id(spec))
        spec_type = type(spec)
        if spec_type is dict:
            if id(spec) not in tree:
                return OrderedDict()  # no record ever reached this node
            key_index = tree[_KEY_INDEX]
            emitted = tree.get(_EMITTED, set())
            ret = OrderedDict()
            for key, sub_tree in tree[id(spec)].items():
                if key not in emitted:
                    continue  # bucket exists but produced no non-SKIP output
                valspec = self._valspec_for(spec, key_index[key])
                ret[key] = self._materialize(valspec, sub_tree)
            return ret
        if spec_type is list:
            return list(tree.get(id(spec), []))
        raise SpillError('unsupported spec in spill mode: %r' % (spec,))


class SpillGroup:
    """Group mode with an explicit memory budget and disk spill.

    Behaves like :class:`~glom.grouping.Group`, but keeps at most
    *budget* groups resident in memory. When a new group appears and
    the budget is full, the least-recently-used group's accumulator
    state is encoded through the spill protocol and staged in a local
    SQLite database; it is transparently swapped back in when the group
    is seen again.

    >>> glom([1, 2, 3, 1, 2], SpillGroup({T: SpillCount()}, budget=1))
    OrderedDict({1: 2, 2: 2, 3: 1})

    Group keys must be scalars (str/int/float/bool/None) or tuples of
    scalars, and final groups are delivered in first-seen order. The
    staging database is per-run and removed on completion, error,
    early STOP, or explicit :meth:`close`.
    """

    def __init__(self, spec, budget=1024, spill_path=None):
        if budget < 1:
            raise ValueError('budget must be >= 1, not %r' % (budget,))
        self.spec = spec
        self.budget = budget
        self.spill_path = spill_path
        self._runner = None
        self._store = None
        # snapshots kept after close() so stats remain readable post-run
        self._peak_active_groups = 0
        self._spill_count = 0
        self._load_count = 0

    # -- stats ---------------------------------------------------------

    @property
    def active_group_count(self):
        return self._runner.active_group_count if self._runner else 0

    @property
    def peak_active_groups(self):
        if self._runner:
            return self._runner.peak_active
        return self._peak_active_groups

    @property
    def spill_count(self):
        if self._runner:
            return self._runner.spill_count
        return self._spill_count

    @property
    def load_count(self):
        if self._runner:
            return self._runner.load_count
        return self._load_count

    # -- lifecycle -----------------------------------------------------

    def _open(self):
        self._store = _SpillStore(self.spill_path)
        self._runner = _SpillRunner(self.spec, self.budget, self._store)

    def close(self):
        """drop this run's staging database, if any."""
        if self._store is not None:
            if self._runner is not None:
                self._peak_active_groups = self._runner.peak_active
                self._spill_count = self._runner.spill_count
                self._load_count = self._runner.load_count
            self._store.close()
            self._store = None
            self._runner = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    # -- spec protocol ---------------------------------------------------

    def glomit(self, target, scope):
        scope[MODE] = SPILL_MODE
        scope[CUR_AGG] = None
        scope[ACC_TREE] = {}
        self._open()
        try:
            for t in target_iter(target, scope):
                if self._runner.process(t, scope) is STOP:
                    break
            # materialize on both normal completion and early STOP, so no
            # spilled placeholder can leak into the returned structure
            return self._runner.result()
        finally:
            self.close()

    def __repr__(self):
        return '%s(%r, budget=%r)' % (
            self.__class__.__name__, self.spec, self.budget)
