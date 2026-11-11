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

Spill-to-disk grouping with SpillGroup
--------------------------------------

When a grouping operation would accumulate too many groups to fit in
memory, :class:`~glom.SpillGroup` keeps the number of *active* (in-memory)
groups bounded by a shared ``budget``. Once the budget is exceeded, the
least-recently-used group's accumulator state is encoded through an
explicit, JSON-based protocol and staged in a local :mod:`sqlite3`
database. When that group appears again its state is decoded and swapped
back into memory -- no extra database process is required.

Only aggregators that implement the spill encoding protocol
(:class:`~glom.SpillableAgg`) are supported, so arbitrary objects are
never pickled. The built-in spill aggregators cover count, sum, min and
max, including their nested dict/list combinations. Final groups are
delivered in first-seen order.

.. autoclass:: glom.SpillGroup

.. autoclass:: glom.SpillableAgg

.. autoclass:: glom.SpillCount

.. autoclass:: glom.SpillSum

.. autoclass:: glom.SpillMin

.. autoclass:: glom.SpillMax

Exceptions
----------

.. autoclass:: glom.FoldError

.. autoclass:: glom.SpillError
