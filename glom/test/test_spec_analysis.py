"""Tests for the read-only static spec analyzer (glom.spec_analysis)."""

from dataclasses import dataclass, field
from typing import (
    List,
    NotRequired,
    Optional,
    Union,
    TypedDict,
)

import pytest

from glom import Call, Coalesce, Fill, Invoke, Path, Pipe, T, Val
from glom.core import TargetRegistry, _DEFAULT_SCOPE
from glom.spec_analysis import (
    DEFINITELY_MISSING,
    UNKNOWN,
    SpecAnalysis,
    analyze_spec,
)


# ---------------------------------------------------------------------------
# mixed required/optional models
# ---------------------------------------------------------------------------

class Contact(TypedDict):
    email: str
    phone: NotRequired[Optional[str]]


class Person(TypedDict):
    person_id: int
    nickname: NotRequired[str]
    contact: NotRequired[Contact]


@dataclass
class Address:
    city: str
    zipcode: str = "00000"


@dataclass
class Account:
    account_id: int
    address: Optional[Address] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class Tree:
    value: int
    children: List["Tree"] = field(default_factory=list)


@dataclass
class Chain:
    value: int
    next: Optional["Chain"] = None


def test_required_field_is_fully_verified():
    result = analyze_spec(Person, "person_id")
    assert result.output_type is int
    assert result.fully_verified
    assert not result.issues


def test_unknown_field_is_definite_error():
    result = analyze_spec(Person, "does_not_exist")
    assert result.has_definite_errors
    assert not result.fully_verified
    error = result.definite_errors[0]
    assert error.severity == DEFINITELY_MISSING
    assert error.path == ("does_not_exist",)
    assert result.output_type is None


def test_optional_field_is_maybe_missing_but_runnable():
    result = analyze_spec(Person, "nickname")
    assert not result.definite_errors
    assert len(result.maybe_missing) == 1
    assert result.maybe_missing[0].path == ("nickname",)
    # optional traversal is legal: still counts as fully verified
    assert result.fully_verified


def test_nested_definite_error_keeps_full_path():
    result = analyze_spec(Person, "contact.bogus")
    paths = {issue.path for issue in result.definite_errors}
    assert ("contact", "bogus") in paths
    maybe = {issue.path for issue in result.maybe_missing}
    assert ("contact",) in maybe
    assert not result.fully_verified


def test_optional_then_required_only_maybe_missing():
    result = analyze_spec(Person, "contact.email")
    assert not result.definite_errors
    assert [i.path for i in result.maybe_missing] == [("contact",)]
    assert result.output_type is str


def test_dataclass_required_vs_default_fields():
    required = analyze_spec(Account, "account_id")
    assert required.fully_verified and required.output_type is int

    optional_city = analyze_spec(Account, "address.city")
    assert optional_city.output_type is str
    assert not optional_city.definite_errors
    maybe_paths = [i.path for i in optional_city.maybe_missing]
    assert ("address",) in maybe_paths

    bad = analyze_spec(Account, "address.country")
    assert any(i.path == ("address", "country") for i in bad.definite_errors)


def test_dataclass_default_value_is_optional():
    result = analyze_spec(Address, "zipcode")
    assert not result.definite_errors
    assert result.maybe_missing[0].path == ("zipcode",)


# ---------------------------------------------------------------------------
# collections, tuples and recursion
# ---------------------------------------------------------------------------

def test_list_spec_iterates_element_type():
    result = analyze_spec(List[Account], ["account_id"])
    assert result.output_type == List[int]
    assert result.fully_verified


def test_list_spec_inside_nested_path():
    result = analyze_spec(List[Person], [{"id": "person_id"}])
    # List[Dict[..., int]]
    assert result.issues == ()


def test_fixed_tuple_indexing():
    @dataclass
    class Holder:
        pair: tuple[int, str]

    assert analyze_spec(Holder, "pair.0").output_type is int
    assert analyze_spec(Holder, "pair.1").output_type is str
    bad = analyze_spec(Holder, "pair.5")
    assert bad.definite_errors[0].path == ("pair", 5)


def test_numeric_string_path_segment_is_index():
    @dataclass
    class Holder:
        items: tuple[int, int]

    result = analyze_spec(Holder, Path("items", 0))
    assert result.output_type is int


def test_dict_assembly_output_type():
    result = analyze_spec(Person, {"id": "person_id", "name": "nickname"})
    assert result.definite_errors == ()
    assert result.maybe_missing  # nickname optional


def test_fill_tuple_assembly():
    result = analyze_spec(Person, Fill((T.person_id, T.nickname)))
    assert not result.definite_errors
    assert any(i.path == (1, "nickname") for i in result.maybe_missing)
    assert result.output_type.__args__ == (int, str)


def test_recursive_dataclass_terminates():
    result = analyze_spec(Tree, ("children", ["value"]))
    assert result.output_type == List[int]
    # recursion terminates instead of raising or looping forever
    assert isinstance(result, SpecAnalysis)


def test_optional_chain_records_each_none_branch():
    result = analyze_spec(Chain, "next.next.value")
    paths = [i.path for i in result.maybe_missing]
    assert ("next",) in paths
    assert ("next", "next") in paths
    assert ("next", "next", "value") in paths
    assert result.output_type is int


def test_list_index_maybe_missing_but_typed():
    result = analyze_spec(Account, T.tags[0])
    assert result.output_type is str
    assert not result.definite_errors
    assert any(i.path == ("tags", 0) for i in result.maybe_missing)


def test_star_traversal_preserves_element_type():
    result = analyze_spec(List[Person], T["*"]["person_id"])
    assert result.output_type == List[int]
    assert result.fully_verified


def test_star_over_non_iterable_keeps_unknown_branch():
    result = analyze_spec(Person, T["*"])
    assert result.unknown
    assert not result.fully_verified


# ---------------------------------------------------------------------------
# Coalesce
# ---------------------------------------------------------------------------

def test_coalesce_all_valid_branches():
    result = analyze_spec(Person, Coalesce("nickname", "person_id"))
    assert not result.definite_errors
    assert result.fully_verified


def test_coalesce_bad_branch_contained_by_good_branch():
    result = analyze_spec(Person, Coalesce("nickname", "ghost_field"))
    # the bad branch cannot fail the whole Coalesce
    assert not result.definite_errors
    assert result.contained_errors
    assert result.contained_errors[0].path == ("coalesce_1", "ghost_field")
    assert result.fully_verified


def test_coalesce_all_bad_with_default_is_contained():
    result = analyze_spec(Person, Coalesce("a", "b", default="fallback"))
    assert not result.definite_errors
    assert len(result.contained_errors) == 2
    assert result.output_type is str


def test_coalesce_all_bad_without_default_is_definite():
    result = analyze_spec(Person, Coalesce("a", "b"))
    assert result.has_definite_errors
    assert not result.fully_verified


# ---------------------------------------------------------------------------
# functions and the no-execution / no-registry-change guarantees
# ---------------------------------------------------------------------------

def test_annotated_function_uses_return_type():
    def label(person: Person) -> str:
        raise AssertionError("must never be called")

    result = analyze_spec(Person, label)
    assert result.output_type is str
    assert result.fully_verified


def test_unannotated_function_is_unknown_not_invoked():
    def mystery(person):
        raise AssertionError("must never be called")

    result = analyze_spec(Person, mystery)
    assert result.unknown
    assert result.unknown[0].severity == UNKNOWN
    assert not result.fully_verified


def test_call_and_invoke_never_execute():
    def annotated(account: Account) -> int:
        raise AssertionError("must never be called")

    call_result = analyze_spec(Account, Call(annotated))
    invoke_result = analyze_spec(Account, Invoke(annotated).specs(T))
    assert call_result.output_type is int
    assert invoke_result.output_type is int


def test_t_method_call_is_unknown_without_invocation():
    class Exploding:
        def __getattribute__(self, name):
            raise AssertionError("getter must not run")

    @dataclass
    class Model:
        number: int

    result = analyze_spec(Model, T.number.bit_length())
    assert result.unknown
    assert not result.fully_verified


def test_analysis_does_not_modify_registry():
    registry = _DEFAULT_SCOPE[TargetRegistry]
    before = {key: dict(value) for key, value in registry._op_type_map.items()}
    analyze_spec(Person, Coalesce("nickname", "ghost", default=1))
    after = {key: dict(value) for key, value in registry._op_type_map.items()}
    assert before == after


def test_analysis_does_not_instantiate_target():
    @dataclass
    class Strict:
        number: int

        def __post_init__(self):
            raise AssertionError("target must not be constructed")

    result = analyze_spec(Strict, "number")
    assert result.output_type is int


def test_unknown_branch_blocks_full_verification_claim():
    @dataclass
    class Model:
        number: int

    def dynamic(value):
        return value

    good = analyze_spec(Model, "number")
    mixed = analyze_spec(Model, ("number", dynamic))
    assert good.fully_verified
    assert not mixed.fully_verified
    assert mixed.unknown


def test_unknown_step_does_not_emit_cascading_noise():
    def dynamic(value):
        return value

    result = analyze_spec(Person, (dynamic, "a", "b"))
    assert len(result.unknown) == 1
    assert not result.definite_errors
    assert not result.fully_verified


def test_pipe_matches_tuple_chain():
    pipe_result = analyze_spec(Person, Pipe("contact", "email"))
    tuple_result = analyze_spec(Person, ("contact", "email"))
    assert pipe_result.output_type == tuple_result.output_type is str


def test_val_uses_literal_type():
    assert analyze_spec(Person, Val(7)).output_type is int
    assert analyze_spec(Person, Val("x")).output_type is str


def test_issue_glom_path_roundtrip():
    result = analyze_spec(Person, "contact.bogus")
    error = result.definite_errors[0]
    assert error.glom_path == Path("contact", "bogus")


# ---------------------------------------------------------------------------
# union types
# ---------------------------------------------------------------------------

class Cat(TypedDict):
    meow: NotRequired[str]


class Dog(TypedDict):
    bark: NotRequired[str]


class Pet(TypedDict):
    pet: Union[Cat, Dog]


def test_union_partial_member_access_is_maybe_missing():
    result = analyze_spec(Pet, "pet.meow")
    assert not result.definite_errors
    assert result.contained_errors
    assert any(i.path == ("pet", "meow") for i in result.maybe_missing)
    assert result.fully_verified


def test_union_no_member_support_is_definite():
    result = analyze_spec(Pet, "pet.quack")
    assert result.has_definite_errors
    assert not result.fully_verified
