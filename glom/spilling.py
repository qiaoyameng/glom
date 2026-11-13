"""
Spill mode -- bounded-memory grouping with SQLite-backed spillover.

:class:`SpillGroup` behaves like a restricted :class:`~glom.grouping.Group`
whose per-group accumulator states share a fixed memory budget. When the
number of in-memory (active) group states exceeds the budget, the
least-recently-used states are encoded and written to a local SQLite
database, then swapped back in when their group is seen again.

Only aggregators with an explicit state-encoding protocol are supported.
``Count``, ``Sum``, ``Min``, and ``Max`` work out of the box; custom
aggregators must define ``encode_state(state)`` (returning a
JSON-serializable value) and ``decode_state(data)``. Arbitrary objects are
never pickled.
"""

import os
import json
import sqlite3
import tempfile
from collections import OrderedDict

from .core import glom, MODE, SKIP, STOP, BadSpec, T
from .grouping import GROUP, target_iter, ACC_TREE, CUR_AGG, Min, Max
from .reduction import Count, Sum


_SCALAR_ENCODERS = {
    type(None): lambda v: ['none'],
    bool: lambda v: ['bool', v],
    int: lambda v: ['int', v],
    float: lambda v: ['float', v],
    str: lambda v: ['str', v],
}

_SCALAR_DECODERS = {
    'none': lambda v: None,
    'bool': bool,
    'int': int,
    'float': float,
    'str': str,
}


def _encode_scalar(value):
    try:
        return _SCALAR_ENCODERS[type(value)](value)
    except KeyError:
        raise BadSpec('SpillGroup only supports scalar keys/states of type'
                      ' None, bool, int, float, and str, not: %r' % (value,))


def _decode_scalar(node):
    tag = node[0]
    try:
        decoder = _SCALAR_DECODERS[tag]
    except KeyError:
        raise BadSpec('SpillGroup could not decode scalar of tag %r' % (tag,))
    return decoder(node[1]) if len(node) > 1 else decoder()


def _encode_key(key):
    if type(key) is tuple:
        return ['tuple', [_encode_key(part) for part in key]]
    return _encode_scalar(key)


def _encode_path(ordinal, path):
    return json.dumps([ordinal, [_encode_key(key) for key in path]])


def _is_agg(spec):
    return callable(getattr(spec, 'agg', None)) or callable(getattr(spec, 'glomit', None))


class _Slot:
    """a leaf aggregator position within a SpillGroup spec, plus its codec"""
    __slots__ = ('ordinal', 'agg', 'encode', 'decode', 'finalize', 'guarded')

    def __init__(self, ordinal, agg):
        self.ordinal = ordinal
        self.agg = agg
        encode = getattr(agg, 'encode_state', None)
        decode = getattr(agg, 'decode_state', None)
        if callable(encode) and callable(decode):
            self.encode = encode
            self.decode = decode
            # custom states may be mutated in place, so guard failures
            # with a pre-aggregation snapshot
            self.guarded = True
        elif isinstance(agg, (Count, Sum, Min, Max)):
            self.encode = _encode_scalar
            self.decode = _decode_scalar
            self.guarded = False
        else:
            raise BadSpec('SpillGroup aggregates must be Count, Sum, Min, Max,'
                          ' or define encode_state()/decode_state() methods,'
                          ' not: %r' % (agg,))
        finalize = getattr(agg, 'finalize_state', None)
        self.finalize = finalize if callable(finalize) else (lambda state: state)


def _walk_spec(spec, leaves):
    if isinstance(spec, dict):
        if not spec:
            raise BadSpec('SpillGroup does not support empty dict specs')
        for keyspec, valspec in spec.items():
            _walk_spec(valspec, leaves)
        return
    if not _is_agg(spec):
        raise BadSpec('SpillGroup specs must be dicts with aggregator leaves'
                      ' (Count, Sum, Min, Max, or custom aggregators defining'
                      ' encode_state()/decode_state()), not: %r' % (spec,))
    leaves.append(spec)


class SpillGroup:
    """A bounded-memory variant of :class:`~glom.grouping.Group` which
    spills inactive group states to a local SQLite database.

    Args:
       spec: a (possibly nested) dict of keyspecs whose leaves are
          aggregators. Built-in support covers ``Count``, ``Sum``,
          ``Min``, and ``Max``. Custom aggregators must define
          ``encode_state(state)`` / ``decode_state(data)`` and may define
          ``finalize_state(state)``.
       max_active (int): shared budget of group states kept in memory.
          Least-recently-used states beyond this budget are spilled.
       spill_path (str): path of the SQLite spill file. Defaults to a
          temporary file. The file is removed on close in either case.

    Group keys must be scalars (``None``, ``bool``, ``int``, ``float``,
    ``str``) or tuples thereof. Final groups are delivered as a dict in
    order of first appearance, matching :class:`~glom.grouping.Group`.

    >>> glom([1, 2, 3, 4], SpillGroup({T % 2: Count()}, max_active=1))
    {1: 2, 0: 2}

    The spill file is cleaned up when the aggregation finishes, stops
    early, raises, or when :meth:`close` is called explicitly. No separate
    database process is required.
    """
    def __init__(self, spec, max_active=1024, spill_path=None):
        if isinstance(spec, dict):
            self.spec = spec
        elif _is_agg(spec):
            self.spec = spec
        else:
            raise BadSpec('SpillGroup expected a dict spec or aggregator,'
                          ' not: %r' % (spec,))
        if not isinstance(max_active, int) or max_active < 1:
            raise ValueError('expected max_active >= 1, not: %r' % (max_active,))
        self.max_active = max_active
        self.spill_path = spill_path
        leaves = []
        _walk_spec(spec, leaves)
        self._slots = [_Slot(ordinal, agg) for ordinal, agg in enumerate(leaves)]
        self._slots_by_id = {id(slot.agg): slot for slot in self._slots}
        self._leaf_spec = spec if not isinstance(spec, dict) else None
        self._reset_runtime_state()

    def _reset_runtime_state(self):
        self._db = None
        self._tmp_path = None
        self._active = OrderedDict()  # (ordinal, path) -> state, LRU-ordered
        self._key_order = []  # (ordinal, path) in first-appearance order
        self._seen = set()
        self._stopped = set()  # id()s of keyspecs which returned STOP
        self.spill_count = 0  # states written out to SQLite
        self.load_count = 0  # states swapped back in from SQLite
        self.max_active_seen = 0  # high-water mark of in-memory group states

    @property
    def active_count(self):
        """number of group states currently held in memory"""
        return len(self._active)

    def _begin(self):
        self._reset_runtime_state()
        if self.spill_path is None:
            fd, self._tmp_path = tempfile.mkstemp(prefix='glom-spill-', suffix='.sqlite')
            os.close(fd)
            db_path = self._tmp_path
        else:
            db_path = self.spill_path
        self._db = sqlite3.connect(db_path, isolation_level=None)  # autocommit
        self._db.execute('DROP TABLE IF EXISTS spill')  # no stale state from prior runs
        self._db.execute('CREATE TABLE IF NOT EXISTS spill (k TEXT PRIMARY KEY, v TEXT)')

    def close(self):
        """release the spill database and remove this run's spill file;
        safe to call multiple times"""
        db, self._db = self._db, None
        if db is not None:
            db.close()
        for path in (self._tmp_path, self.spill_path):
            if not path:
                continue
            for suffix in ('', '-journal', '-wal', '-shm'):
                try:
                    os.unlink(path + suffix)
                except OSError:
                    pass
        self._tmp_path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def glomit(self, target, scope):
        scope[MODE] = GROUP
        scope[CUR_AGG] = None
        scope[ACC_TREE] = {}
        self._begin()
        try:
            for t in target_iter(target, scope):
                if self._process_record(t, scope):
                    break  # every top-level keyspec returned STOP
            return self._finalize()
        finally:
            self.close()

    def _process_record(self, t, scope):
        if self._leaf_spec is not None:
            self._agg_leaf(self._slots[0], t, (), scope)
            return False
        return self._rec_dict(self.spec, t, (), scope)

    def _rec_dict(self, spec, t, path, scope):
        all_done = True
        for keyspec, valspec in spec.items():
            if id(keyspec) in self._stopped:
                continue
            key = scope[glom](t, keyspec, scope)
            if key is SKIP:
                all_done = False  # SKIP means we still want more vals
                continue
            if key is STOP:
                self._stopped.add(id(keyspec))
                continue
            _encode_key(key)  # validate before touching any state
            if isinstance(valspec, dict):
                done = self._rec_dict(valspec, t, path + (key,), scope)
            else:
                slot = self._slots_by_id[id(valspec)]
                result = self._agg_leaf(slot, t, path + (key,), scope)
                done = result is STOP
            if done:
                self._stopped.add(id(keyspec))
            else:
                all_done = False
        return all_done

    def _agg_leaf(self, slot, t, path, scope):
        agg_key = (slot.ordinal, path)
        if agg_key in self._active:
            self._active.move_to_end(agg_key)
            present = True
        else:
            present = self._swap_in(slot, agg_key)
        state = self._active[agg_key] if present else None
        tree = {slot.agg: state} if present else {}
        scope[ACC_TREE] = tree
        scope[CUR_AGG] = None
        snapshot = slot.encode(state) if present and slot.guarded else None
        try:
            result = scope[glom](t, slot.agg, scope)
        except Exception:
            # a failed aggregation must not corrupt the group's counts;
            # restore the pre-aggregation state and leave persistence alone
            if snapshot is not None:
                self._active[agg_key] = slot.decode(snapshot)
            raise
        if result is STOP or result is SKIP:
            return result
        self._store(slot, agg_key, tree[slot.agg])
        return None

    def _swap_in(self, slot, agg_key):
        row = self._db.execute('SELECT v FROM spill WHERE k = ?',
                               (_encode_path(*agg_key),)).fetchone()
        if row is None:
            return False
        self._db.execute('DELETE FROM spill WHERE k = ?', (_encode_path(*agg_key),))
        self.load_count += 1
        self._store(slot, agg_key, slot.decode(json.loads(row[0])))
        return True

    def _store(self, slot, agg_key, state):
        if agg_key not in self._active and len(self._active) >= self.max_active:
            self._evict()
        self._active[agg_key] = state
        self._active.move_to_end(agg_key)
        self.max_active_seen = max(self.max_active_seen, len(self._active))
        if agg_key not in self._seen:
            self._seen.add(agg_key)
            self._key_order.append(agg_key)

    def _evict(self):
        (ordinal, path), state = self._active.popitem(last=False)
        slot = self._slots[ordinal]
        self._db.execute('INSERT OR REPLACE INTO spill (k, v) VALUES (?, ?)',
                         (_encode_path(ordinal, path), json.dumps(slot.encode(state))))
        self.spill_count += 1

    def _finalize(self):
        if self._leaf_spec is not None:
            slot = self._slots[0]
            state, present = self._get_state(slot, (slot.ordinal, ()))
            return slot.finalize(state) if present else None
        result = {}
        for ordinal, path in self._key_order:
            slot = self._slots[ordinal]
            state, present = self._get_state(slot, (ordinal, path))
            if not present:
                continue
            node = result
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = slot.finalize(state)
        return result

    def _get_state(self, slot, agg_key):
        if agg_key in self._active:
            return self._active[agg_key], True
        row = self._db.execute('SELECT v FROM spill WHERE k = ?',
                               (_encode_path(*agg_key),)).fetchone()
        if row is None:
            return None, False
        return slot.decode(json.loads(row[0])), True

    def __repr__(self):
        cn = self.__class__.__name__
        return f'{cn}({self.spec!r}, max_active={self.max_active!r})'
