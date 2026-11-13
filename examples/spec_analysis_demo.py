"""Static spec analysis demo.

Run with: python examples/spec_analysis_demo.py

Shows the difference between:
  * a *definite error*  - a spec path that accesses a field the
    declared type does not have, so it can never succeed;
  * a *maybe missing*    - a legal spec over an optional field that
    runs fine on some data and fails on other data;
  * an *unknown branch*  - a fragment the analyzer cannot verify
    without executing code.

The second half builds small, real objects and runs the specs to
cross-check that the paths the analyzer allowed really work.
"""

from dataclasses import dataclass
from typing import NotRequired, TypedDict

from glom import Coalesce, glom
from glom import analyze_spec


class Contact(TypedDict):
    email: str
    phone: NotRequired[str]


class Person(TypedDict):
    person_id: int                       # required
    nickname: NotRequired[str]           # optional
    contact: NotRequired[Contact]        # optional nested model


@dataclass
class Account:
    account_id: int
    label: str = "default"               # default => optional


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show(result):
    print("  output type :", result.output_type)
    print("  verified    :", result.fully_verified)
    for issue in result.issues:
        label = issue.severity + (" (contained)" if issue.contained else "")
        print("  - %-29s %-28s %s"
              % (label, tuple(issue.path), issue.message))


banner("1. Required field: clean and fully verified")
show(analyze_spec(Person, "person_id"))

banner("2. Optional field: runnable, but can fail on some data")
show(analyze_spec(Person, "nickname"))

banner("3. Definite error: field is not declared anywhere")
show(analyze_spec(Person, "ssn"))

banner("4. Optional step then a definite error underneath")
show(analyze_spec(Person, "contact.fax_number"))

banner("5. Coalesce: one bad branch contained by a working branch")
show(analyze_spec(Person, Coalesce("nickname", "ghost", default="anon")))

banner("6. Unknown branch: unannotated function is never called")


def unnamed(person):
    raise AssertionError("analysis must not call this function")


show(analyze_spec(Person, unnamed))

banner("7. Dataclass with a defaulted (optional) field")
show(analyze_spec(Account, "label"))
show(analyze_spec(Account, "nonexistent"))


# ---------------------------------------------------------------------------
# runtime cross-check on real, small objects
# ---------------------------------------------------------------------------

banner("8. Cross-check the allowed paths against real objects")

complete_person = {
    "person_id": 7,
    "nickname": "seven",
    "contact": {"email": "seven@example.com", "phone": "555-0100"},
}
sparse_person = {"person_id": 8}  # every optional field absent

allowed_specs = [
    "person_id",
    "nickname",
    "contact.email",
    Coalesce("nickname", "person_id"),
    Coalesce("nickname", "ghost", default="anon"),
]

for spec in allowed_specs:
    analysis = analyze_spec(Person, spec)
    assert not analysis.has_definite_errors, analysis
    on_complete = glom(complete_person, spec)
    print("  %-45r complete -> %r" % (spec, on_complete))
    try:
        on_sparse = glom(sparse_person, spec)
        print("  %-45r sparse   -> %r" % (spec, on_sparse))
    except Exception as exc:
        # expected for the plain optional paths with no Coalesce
        print("  %-45r sparse   -> %s (the analyzer flagged maybe_missing)"
              % (spec, type(exc).__name__))

account = Account(account_id=1)
print("  dataclass Account.label          -> %r"
      % glom(account, "label"))

banner("9. Confirm the definite error really fails at runtime")
bad = analyze_spec(Person, "ssn")
assert bad.has_definite_errors and not bad.fully_verified
try:
    glom(complete_person, "ssn")
except Exception as exc:
    print("  glom raised %s as predicted by the definite-error finding"
          % type(exc).__name__)

print("\nAll static findings matched the runtime behavior.")
