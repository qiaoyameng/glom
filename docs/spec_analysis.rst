Static Spec Analysis
====================

Before shipping a newly extracted glom spec, it helps to know whether
it walks fields that the target data does not actually have.
:func:`~glom.analyze_spec` answers that question *statically*: give it
a declared target type and a spec, and it reports the inferred output
type, paths that can fail at runtime, and paths that are definitely
wrong, all without constructing a business object.

The analysis is fully read-only:

* no instance of the target type is constructed;
* glom's handler registry is never modified;
* functions, methods, and getters referenced by the spec are never
  called -- only their declared type annotations are read.

Supported target types include :class:`~typing.TypedDict`,
:func:`~dataclasses.dataclass`, and anything expressible with
:mod:`typing` (``Optional[...]``, ``list[...]``, nested dicts and
tuples, and recursive/self-referential types). Supported spec shapes
include paths and :data:`~glom.T` expressions, dict/list/tuple
assembly, :class:`~glom.Coalesce`, :class:`~glom.Fill`,
:class:`~glom.Val`, :class:`~glom.Call`/:class:`~glom.Invoke`, and
plain annotated callables.

Three kinds of findings
-----------------------

* **definite errors** (``definitely_missing``) access a field the
  declared type does not have. Such a path cannot succeed on any
  valid instance.
* **maybe missing** (``maybe_missing``) paths are legal specs that
  traverse an optional or nullable field. They run correctly on some
  data and raise on other data.
* **unknown** (``unknown``) fragments cannot be verified from the
  declared types alone -- for example, an unannotated function or a
  method call. They are never "tried out" at runtime.

A result is only :attr:`~glom.SpecAnalysis.fully_verified` when there
are no definite errors *and* no unknown branches. An unknown branch
never silently counts as verified.

.. code-block:: python

   from typing import NotRequired, TypedDict
   from glom import Coalesce, analyze_spec

   class Contact(TypedDict):
       email: str
       phone: NotRequired[str]

   class Person(TypedDict):
       person_id: int
       nickname: NotRequired[str]
       contact: NotRequired[Contact]

   analyze_spec(Person, "person_id")
   # SpecAnalysis(output_type=int, definite_errors=0, ...)

   analyze_spec(Person, "nickname")
   # maybe_missing at ('nickname',): legal, but can fail on sparse data

   analyze_spec(Person, "ssn")
   # definitely_missing at ('ssn',): Person declares no such field

   analyze_spec(Person, "contact.fax_number")
   # maybe_missing ('contact',) + definitely_missing ('contact', 'fax_number')

   analyze_spec(Person, Coalesce("nickname", "ghost", default="anon"))
   # the bad 'ghost' branch is contained by Coalesce; the spec can still run

A runnable demo that cross-checks the findings against real objects
lives at ``examples/spec_analysis_demo.py``.

API
---

.. autofunction:: glom.analyze_spec

.. autoclass:: glom.SpecAnalysis
   :members:

.. autoclass:: glom.SpecIssue
   :members:

.. autodata:: glom.MAYBE_MISSING
.. autodata:: glom.DEFINITELY_MISSING
.. autodata:: glom.UNKNOWN
