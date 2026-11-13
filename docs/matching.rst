Matching & Validation
=====================

.. automodule:: glom.matching

.. contents:: Contents
   :local:

Validation with Match
~~~~~~~~~~~~~~~~~~~~~

For matching whole data structures, use a :class:`~glom.Match` spec.

.. autoclass:: glom.Match
   :members:

Optional and required ``dict`` key matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that our four :class:`~glom.Match` rules above imply that
:class:`object` is a match-anything pattern.  Because
``isinstance(val, object)`` is true for all values in Python,
``object`` is a useful stopping case. For instance, if we wanted to
extend an example above to allow additional keys and values in the
user dict above we could add :class:`object` as a generic pass through::

  >>> target = [{'id': 1, 'email': 'alice@example.com', 'extra': 'val'}]
  >>> spec = Match([{'id': int, 'email': str, object: object}])
  >>> assert glom(target, spec) == \\
      ... [{'id': 1, 'email': 'alice@example.com', 'extra': 'val'}]
  True

The fact that ``{object: object}`` will match any dictionary exposes
the subtlety in :class:`~glom.Match` dictionary evaluation.

By default, value match keys are required, and other keys are
optional.  For example, ``'id'`` and ``'email'`` above are required
because they are matched via ``==``.  If either was not present, it
would raise :class:`~glom.MatchError`.  :class:`object` however is matched
with :func:`isinstance`. Since it is not a value-match comparison,
it is not required.

This default behavior can be modified with :class:`~glom.Required`
and :class:`~glom.Optional`.

.. autoclass:: glom.Optional

.. autoclass:: glom.Required

Named captures and backreferences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~glom.Capture` binds the current target to a name.  Reusing the
same name later in the pattern requires the new value to compare equal,
which makes it possible to express relationships such as a nested
record's ``parent_id`` matching the outer record's ``id``::

  >>> from glom import Match, Capture, CaptureRef
  >>> target = {'id': 1, 'child': {'parent_id': 1}}
  >>> spec = Match({
  ...     'id': Capture('record_id'),
  ...     'child': {'parent_id': CaptureRef('record_id')},
  ... })
  >>> result, bindings = spec.verify_with_captures(target)
  >>> bindings
  {'record_id': 1}

Bindings are local to one top-level :func:`~glom.glom` call.  They are
never stored on a capture rule or on the input data, so specs can be
reused across calls and threads.  Use :class:`~glom.Captures` as a spec
to get a snapshot of the bindings accumulated so far.

Bindings from failed alternatives are discarded.  :class:`~glom.And`
shares the bindings made by its successful children, :class:`~glom.Or`
commits only the branch that succeeds, and :class:`~glom.Not` never
publishes bindings made while testing its child.  Dictionary, list,
tuple, :class:`~glom.Optional`, and :class:`~glom.Switch` matching follow
the same transactional rules.

.. autoclass:: glom.Capture

.. autoclass:: glom.CaptureRef

.. autoclass:: glom.Captures

.. autoclass:: glom.MatchResult

.. autoclass:: glom.UnboundCaptureError

``M`` Expressions
~~~~~~~~~~~~~~~~~

The most concise way to express validation and guards.

.. autodata:: glom.M

Boolean operators and matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While ``M`` is an easy way to construct expressions, sometimes a more
object-oriented approach can be more suitable.

.. autoclass:: glom.Or

.. autoclass:: glom.And

.. autoclass:: glom.Not

String matching
~~~~~~~~~~~~~~~

.. autoclass:: glom.Regex

Control flow with ``Switch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Match becomes even more powerful when combined with the ability to
branch spec execution.

.. autoclass:: glom.Switch

Exceptions
~~~~~~~~~~

.. autoclass:: glom.MatchError

.. autoclass:: glom.TypeMatchError

Validation with Check
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: glom.Check

.. autoclass:: glom.CheckError
