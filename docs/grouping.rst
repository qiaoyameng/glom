Reduction & Grouping
====================

This document contains glom techniques for transforming a collection
of data to a smaller set, otherwise known as "grouping" or
"reduction".

Combining iterables with Flatten and Merge
------------------------------------------

.. versionadded:: 19.1.0

Got lists of lists? Sets of tuples? A sequence of dicts (but only want
one)? Do you find yourself reaching for Python's builtin :func:`sum`
and :func:`reduce`? To handle these situations and more, glom has five
specifier types and two convenience functions:

.. autofunction:: glom.flatten

.. autoclass:: glom.Flatten

.. autofunction:: glom.merge

.. autoclass:: glom.Merge

.. autoclass:: glom.Sum

.. autoclass:: glom.Fold

Bounded-memory grouping with SpillGroup
---------------------------------------

When grouping a stream of many users/keys, the accumulator dict of a
plain ``Group`` grows with the number of distinct keys.
:class:`~glom.SpillGroup` offers the same nested grouping/aggregation
model under a shared memory budget: once the number of in-memory group
states exceeds ``max_active``, the least-recently-used states are
encoded and spilled to a local SQLite file (no separate database
process required), and swapped back in when their key reappears.

The first version supports the :class:`~glom.Count`,
:class:`~glom.Sum`, ``Min``, and ``Max`` aggregators, including nested
combinations. Custom aggregators must define
``encode_state(state)``/``decode_state(data)`` methods (and optionally
``finalize_state(state)``); arbitrary objects are never pickled. Group
keys are limited to scalars (``None``, ``bool``, ``int``, ``float``,
``str``) or tuples thereof, and final groups are delivered in order of
first appearance. ``SKIP``, ``STOP``, and aggregation failures never
leave partial counts behind, and the spill file is cleaned up on
completion, early stop, error, or explicit ``close()``.

.. autoclass:: glom.SpillGroup
   :members: close, active_count

Exceptions
----------

.. autoclass:: glom.FoldError
