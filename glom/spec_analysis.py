"""Static, read-only analysis of glom specs against declared data types.

The main entrypoint is :func:`analyze_spec`. Given a target type (a
:class:`~typing.TypedDict`, :func:`~dataclasses.dataclass`, or any
other type expressible with the :mod:`typing` module) and a glom
spec, it walks the spec *without* constructing a target instance or
calling any function or getter, and reports:

* the inferred output type of the spec;
* spec paths that can fail at runtime because they traverse an
  optional field (``maybe_missing``);
* spec paths that are definitely wrong because they access a field
  that the declared type does not have (``definite_errors``);
* spec fragments that could not be verified because the static
  information runs out (``unknown``).

A result only counts as :attr:`SpecAnalysis.fully_verified` when
there are no definite errors *and* no unknown branches. Analysis
never mutates glom's handler registry and never executes the
functions, methods, or getters embedded in the spec.
"""

import dataclasses as _dataclasses
import inspect as _inspect
import sys
import types as _types
import typing as _typing

from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from glom.core import (
    Call,
    Coalesce,
    Fill,
    Invoke,
    Path,
    Pipe,
    Ref,
    Spec,
    T,
    TType,
    Val,
)
from glom.core import _MISSING
from glom.mutation import Assign, Delete


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

# severity levels
MAYBE_MISSING = 'maybe_missing'
DEFINITELY_MISSING = 'definitely_missing'
UNKNOWN = 'unknown'


class SpecIssue:
    """A single finding recorded during analysis.

    Attributes:
       severity: one of :data:`MAYBE_MISSING`,
          :data:`DEFINITELY_MISSING`, or :data:`UNKNOWN`.
       path: a tuple of path segments (strings, ints, etc.) locating
          the fragment relative to the root target.
       message: a human-readable explanation.
       spec: the spec fragment the finding is about.
    """

    __slots__ = ('severity', 'path', 'message', 'spec', 'contained')

    def __init__(self, severity, path, message, spec=None, contained=False):
        self.severity = severity
        self.path = tuple(path)
        self.message = message
        self.spec = spec
        self.contained = contained

    @property
    def glom_path(self):
        """:class:`~glom.Path` equivalent of :attr:`path`."""
        return Path(*self.path) if self.path else Path()

    def __repr__(self):
        return 'SpecIssue(%r, %r, %r)' % (self.severity, self.path, self.message)

    def __eq__(self, other):
        if not isinstance(other, SpecIssue):
            return NotImplemented
        return (self.severity == other.severity and self.path == other.path
                and self.message == other.message
                and self.contained == other.contained)

    def __hash__(self):
        return hash((self.severity, self.path, self.message, self.contained))


class SpecAnalysis:
    """Result of :func:`analyze_spec`.

    Attributes:
       target_type: the type the analysis started from.
       output_type: the inferred static output type, or :data:`None`
          when the output type cannot be determined.
       issues: tuple of every :class:`SpecIssue` found.
    """

    def __init__(self, target_type, output_type, issues):
        self.target_type = target_type
        self.output_type = output_type
        self.issues = tuple(issues)

    def _with_severity(self, severity):
        return tuple(i for i in self.issues if i.severity == severity)

    @property
    def definite_errors(self):
        """Issues that will fail the spec at runtime.

        Definite errors contained by a :class:`~glom.Coalesce` branch
        are excluded because Coalesce catches them; see
        :attr:`contained_errors`.
        """
        return tuple(i for i in self.issues
                     if i.severity == DEFINITELY_MISSING and not i.contained)

    @property
    def contained_errors(self):
        """Definite errors contained by a surrounding branch.

        These are caught either by a :class:`~glom.Coalesce` branch or
        made non-fatal because another union member supports the same
        access, so they cannot fail the whole spec on their own.
        """
        return tuple(i for i in self.issues
                     if i.severity == DEFINITELY_MISSING and i.contained)

    @property
    def maybe_missing(self):
        """Issues for paths that traverse optional/nullable fields."""
        return self._with_severity(MAYBE_MISSING)

    @property
    def unknown(self):
        """Issues for fragments that could not be statically verified."""
        return self._with_severity(UNKNOWN)

    @property
    def has_definite_errors(self):
        return bool(self.definite_errors)

    @property
    def fully_verified(self):
        """``True`` only when nothing is erroneous *or* unverified.

        Optional-field traversals (``maybe_missing``) do not prevent
        full verification: they are legal specs that may simply fail
        at runtime on particular data. Any unknown branch means the
        spec cannot be claimed fully verified.
        """
        return all(issue.severity == MAYBE_MISSING or issue.contained
                   for issue in self.issues)

    def __repr__(self):
        return ('SpecAnalysis(output_type=%r, definite_errors=%d, '
                'maybe_missing=%d, unknown=%d, contained=%d)' % (
                    self.output_type, len(self.definite_errors),
                    len(self.maybe_missing), len(self.unknown),
                    len(self.contained_errors)))


# ---------------------------------------------------------------------------
# static type model
# ---------------------------------------------------------------------------

class _TypeNode:
    """Common base for the analyzer's lightweight type representation."""

    kind = 'unknown'

    def render(self):
        return _render_type(self)


class _UnknownType(_TypeNode):
    kind = 'unknown'

    def __init__(self, reason=None):
        self.reason = reason

    def __repr__(self):
        return 'Unknown'


class _AnyType(_TypeNode):
    kind = 'any'

    def __repr__(self):
        return 'Any'


class _NoneType(_TypeNode):
    kind = 'none'

    def __repr__(self):
        return 'None'


class _ClassType(_TypeNode):
    """A plain class accessed via attributes."""

    kind = 'class'

    def __init__(self, cls):
        self.cls = cls

    def __repr__(self):
        return getattr(self.cls, '__name__', str(self.cls))


class _DataclassType(_TypeNode):
    kind = 'dataclass'

    def __init__(self, cls, fields):
        self.cls = cls
        # name -> (type node, required bool)
        self.fields = fields

    def __repr__(self):
        return 'Dataclass(%s)' % getattr(self.cls, '__name__', str(self.cls))


class _TypedDictType(_TypeNode):
    kind = 'typeddict'

    def __init__(self, cls, fields):
        self.cls = cls
        # name -> (type node, required bool)
        self.fields = fields

    def __repr__(self):
        return 'TypedDict(%s)' % getattr(self.cls, '__name__', str(self.cls))


class _DictType(_TypeNode):
    kind = 'dict'

    def __init__(self, key_type, value_type):
        self.key_type = key_type
        self.value_type = value_type

    def __repr__(self):
        return 'Dict[%r, %r]' % (self.key_type, self.value_type)


class _ListType(_TypeNode):
    kind = 'list'

    def __init__(self, item_type):
        self.item_type = item_type

    def __repr__(self):
        return 'List[%r]' % (self.item_type,)


class _TupleType(_TypeNode):
    kind = 'tuple'

    def __init__(self, item_types, variadic=False):
        # variadic tuple: item_types has one element repeated
        self.item_types = tuple(item_types)
        self.variadic = variadic

    def __repr__(self):
        if self.variadic:
            return 'Tuple[%r, ...]' % (self.item_types[0],)
        return 'Tuple[%s]' % ', '.join(repr(t) for t in self.item_types)


class _SetType(_TypeNode):
    kind = 'set'

    def __init__(self, item_type):
        self.item_type = item_type

    def __repr__(self):
        return 'Set[%r]' % (self.item_type,)


class _StarType(_TypeNode):
    """Marker for a ``'*'``/``'**'`` traversal: further path steps apply
    to each element and remain wrapped in a list."""

    kind = 'star'

    def __init__(self, item_type):
        self.item_type = item_type

    def __repr__(self):
        return 'Star[%r]' % (self.item_type,)


class _UnionType(_TypeNode):
    kind = 'union'

    def __init__(self, options):
        self.options = tuple(options)

    def __repr__(self):
        return 'Union[%s]' % ', '.join(repr(o) for o in self.options)


_UNKNOWN = _UnknownType()
_ANY = _AnyType()
_NONE = _NoneType()

# PEP 604 union type only exists on 3.10+
_UNION_TYPES = (Union, getattr(_types, 'UnionType', None))


def _flatten_union(options):
    out = []
    for option in options:
        if isinstance(option, _UnionType):
            out.extend(option.options)
        else:
            out.append(option)
    return tuple(out)


def _make_union(options):
    options = _flatten_union(options)
    deduped = []
    seen = set()
    for option in options:
        key = id(option)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    if len(deduped) == 1:
        return deduped[0]
    return _UnionType(tuple(deduped))


def _split_optional(node):
    """Return (non_none_options, has_none) for a possibly-nullable node."""
    if isinstance(node, _UnionType):
        non_none = [o for o in node.options if not isinstance(o, _NoneType)]
        has_none = len(non_none) != len(node.options)
        if not non_none:
            return (_NONE, True)
        if len(non_none) == 1:
            return (non_none[0], has_none)
        return (_UnionType(tuple(non_none)), has_none)
    return (node, False)


# ---------------------------------------------------------------------------
# Python typing -> internal node conversion
# ---------------------------------------------------------------------------

class _TypeNormalizer:
    """Converts Python type expressions into internal :class:`_TypeNode`s.

    Recursive (self-referential) dataclasses and TypedDicts terminate
    via an ``id``-keyed memo: the second encounter reuses the node
    already under construction instead of recursing forever.
    """

    def __init__(self):
        self._memo = {}

    def normalize(self, tp):
        if tp is None or tp is type(None):
            return _NONE
        if tp is Any:
            return _ANY
        if isinstance(tp, _TypeNode):
            return tp

        key = id(tp)
        if key in self._memo:
            return self._memo[key]

        node = self._build(tp)
        self._memo[key] = node
        return node

    def _hints(self, cls):
        try:
            return get_type_hints(cls, include_extras=True)
        except Exception:
            return self._resolve_local_hints(cls)

    def _resolve_local_hints(self, cls):
        """Fallback for locally-defined classes whose ForwardRefs cannot be
        resolved by :func:`get_type_hints` due to missing module globals."""
        raw = dict(getattr(cls, '__annotations__', {}))
        localns = dict(vars(sys.modules.get(cls.__module__, {})))
        localns[cls.__name__] = cls
        resolved = {}
        for name, hint in raw.items():
            resolved[name] = self._resolve_forward_ref(hint, localns)
        return resolved

    def _resolve_forward_ref(self, hint, localns):
        origin = get_origin(hint)
        if origin in _UNION_TYPES:
            return _typing.Union[tuple(
                self._resolve_forward_ref(arg, localns) for arg in get_args(hint))]
        if origin in (list, tuple, dict, set, frozenset):
            args = tuple(self._resolve_forward_ref(arg, localns)
                         for arg in get_args(hint))
            try:
                return origin[args] if len(args) > 1 else origin[args[0]]
            except TypeError:
                return hint
        if isinstance(hint, _typing.ForwardRef):
            try:
                return hint._evaluate(localns, None, frozenset())
            except Exception:
                return _UNKNOWN
        return hint

    def _build(self, tp):
        origin = get_origin(tp)

        if origin is None:
            return self._build_simple(tp)

        if origin in _UNION_TYPES:
            return _make_union([self.normalize(a) for a in get_args(tp)])

        args = get_args(tp)

        if origin is list:
            item = self.normalize(args[0]) if args else _UNKNOWN
            return _ListType(item)
        if origin is set or origin is frozenset:
            item = self.normalize(args[0]) if args else _UNKNOWN
            return _SetType(item)
        if origin is tuple:
            if not args:
                return _TupleType((), variadic=False)
            if len(args) == 2 and args[1] is Ellipsis:
                return _TupleType((self.normalize(args[0]),), variadic=True)
            return _TupleType([self.normalize(a) for a in args])
        if origin is dict:
            if not args:
                return _DictType(_UNKNOWN, _UNKNOWN)
            return _DictType(self.normalize(args[0]), self.normalize(args[1]))

        # generic alias of a concrete user class (e.g. MyList[int]):
        # the unsubscripted class is the best statically known answer
        return self._build_simple(origin)

    def _build_simple(self, tp):
        if tp is Any:
            return _ANY
        if isinstance(tp, _TypeNode):
            return tp
        if not isinstance(tp, type):
            # unresolvable ForwardRef, alias expression, etc.
            return _UnknownType('unresolvable type %r' % (tp,))

        if is_typeddict(tp):
            return self._build_typeddict(tp)
        if _dataclasses.is_dataclass(tp):
            return self._build_dataclass(tp)
        return _ClassType(tp)

    def _build_typeddict(self, cls):
        hints = self._hints(cls)
        required = set(getattr(cls, '__required_keys__', set()))
        node = self._memo.get(id(cls))
        if isinstance(node, _TypedDictType) and node.fields:
            return node
        fields = {}
        if not isinstance(node, _TypedDictType):
            node = _TypedDictType(cls, fields)
            self._memo[id(cls)] = node
        for name, raw_hint in hints.items():
            field_node, field_required = self._field_node(raw_hint, name in required)
            fields[name] = (field_node, field_required)
        node.fields = fields
        return node

    def _build_dataclass(self, cls):
        hints = self._hints(cls)
        node = self._memo.get(id(cls))
        if isinstance(node, _DataclassType) and node.fields:
            return node
        fields = {}
        if not isinstance(node, _DataclassType):
            node = _DataclassType(cls, fields)
            self._memo[id(cls)] = node
        for field in _dataclasses.fields(cls):
            raw_hint = hints.get(field.name, field.type)
            field_node = self.normalize(self._strip_markers(raw_hint))
            has_default = (field.default is not _dataclasses.MISSING
                           or field.default_factory is not _dataclasses.MISSING)
            fields[field.name] = (field_node, not has_default)
        node.fields = fields
        return node

    def _field_node(self, raw_hint, required):
        stripped = raw_hint
        meta_required = None
        origin = get_origin(stripped)
        if origin is _typing.Required:
            meta_required = True
            stripped = get_args(stripped)[0]
        elif origin is _typing.NotRequired:
            meta_required = False
            stripped = get_args(stripped)[0]
        is_required = meta_required if meta_required is not None else required
        return self.normalize(stripped), is_required

    @staticmethod
    def _strip_markers(raw_hint):
        origin = get_origin(raw_hint)
        if origin in (_typing.Required, _typing.NotRequired):
            return get_args(raw_hint)[0]
        return raw_hint


def _render_type(node):
    """Best-effort conversion of an internal node back to a readable type.

    Structural nodes render to generic equivalents; dataclass and
    TypedDict nodes render to the originating class. :data:`None`
    means the type is statically unknown.
    """
    if isinstance(node, _UnknownType):
        return None
    if isinstance(node, _AnyType):
        return Any
    if isinstance(node, _NoneType):
        return type(None)
    if isinstance(node, _ClassType):
        return node.cls
    if isinstance(node, (_DataclassType, _TypedDictType)):
        return node.cls
    if isinstance(node, _ListType):
        inner = _render_type(node.item_type)
        return List[inner if inner is not None else Any]
    if isinstance(node, _StarType):
        inner = _render_type(node.item_type)
        return List[inner if inner is not None else Any]
    if isinstance(node, _SetType):
        inner = _render_type(node.item_type)
        return FrozenSet[inner if inner is not None else Any]
    if isinstance(node, _TupleType):
        if node.variadic:
            inner = _render_type(node.item_types[0])
            return Tuple[inner if inner is not None else Any, ...]
        rendered = []
        for item in node.item_types:
            rendered_item = _render_type(item)
            rendered.append(rendered_item if rendered_item is not None else Any)
        return Tuple[tuple(rendered)]
    if isinstance(node, _DictType):
        key = _render_type(node.key_type)
        val = _render_type(node.value_type)
        return Dict[key if key is not None else Any,
                    val if val is not None else Any]
    if isinstance(node, _UnionType):
        rendered = [_render_type(option) for option in node.options]
        rendered = [r if r is not None else Any for r in rendered]
        if len(rendered) == 1:
            return rendered[0]
        return Union[tuple(rendered)]
    return None


# ---------------------------------------------------------------------------
# analyzer
# ---------------------------------------------------------------------------

_DICT_ATTR_FIELDS = frozenset((
    'items', 'keys', 'values', 'get', '__contains__', 'copy', 'pop',
    'setdefault', 'update', 'clear', 'fromkeys',
))


class _SpecAnalyzer:
    """Walks a spec against a static type, collecting :class:`SpecIssue`s."""

    def __init__(self, root_type):
        self.normalizer = _TypeNormalizer()
        self.root_node = self.normalizer.normalize(root_type)
        self.issues = []
        self._ref_stack = set()

    # -- issue helpers -----------------------------------------------------

    def _record(self, severity, path, message, spec=None):
        for existing in self.issues:
            if (existing.severity == severity
                    and existing.path == tuple(path)
                    and existing.message == message):
                if message not in existing.message:
                    existing.message = '%s; %s' % (existing.message, message)
                return existing
        issue = SpecIssue(severity, path, message, spec)
        self.issues.append(issue)
        return issue

    def _unknown(self, path, message, spec=None):
        return self._record(UNKNOWN, path, message, spec)

    # -- entrypoint --------------------------------------------------------

    def analyze(self, spec):
        return self._analyze(spec, self.root_node, ())

    def _analyze(self, spec, node, path):
        return self._dispatch(spec, node, path)

    def _dispatch(self, spec, node, path):
        if isinstance(spec, str):
            return self._analyze_path(Path.from_text(spec), node, path)
        if isinstance(spec, Path):
            return self._analyze_path(spec, node, path)
        if isinstance(spec, TType):
            return self._analyze_t(spec, node, path)
        if isinstance(spec, dict):
            return self._analyze_dict(spec, node, path)
        if isinstance(spec, list):
            return self._analyze_list(spec, node, path)
        if isinstance(spec, tuple):
            return self._analyze_chain(spec, node, path)
        if isinstance(spec, Pipe):
            return self._analyze_chain(spec.steps, node, path)
        if isinstance(spec, Coalesce):
            return self._analyze_coalesce(spec, node, path)
        if isinstance(spec, Val):
            return self._analyze_val(spec)
        if isinstance(spec, Spec):
            return self._analyze(spec.spec, node, path)
        if isinstance(spec, Fill):
            return self._analyze_fill(spec, node, path)
        if isinstance(spec, (Call, Invoke)):
            return self._analyze_call(spec, node, path)
        if isinstance(spec, Ref):
            return self._analyze_ref(spec, node, path)
        if isinstance(spec, (Assign, Delete)):
            self._unknown(path,
                          'mutation spec %r is outside read-only analysis'
                          % (spec,), spec)
            return _UNKNOWN
        if callable(spec):
            return self._analyze_callable(spec, node, path)

        self._unknown(path, 'unsupported spec fragment %r' % (spec,), spec)
        return _UNKNOWN

    # -- paths / T expressions ---------------------------------------------

    def _analyze_path(self, path_spec, node, path):
        current = node
        accumulated = list(path)
        for op, arg in path_spec.items():
            if op in ('x', 'X'):
                accumulated.append('*' if op == 'x' else '**')
                current = self._star(current, tuple(accumulated), path_spec)
                if isinstance(current, _UnknownType):
                    break
                continue
            step_arg = self._coerce_path_segment(arg)
            accumulated.append(step_arg)
            current = self._access(current, op, step_arg,
                                   tuple(accumulated), path_spec)
            if isinstance(current, _UnknownType):
                break
        return current

    @staticmethod
    def _coerce_path_segment(arg):
        # 'a.0.b' text paths keep numeric segments as strings, but glom
        # resolves them against sequence indices first
        if isinstance(arg, str) and arg.lstrip('-').isdigit():
            return int(arg)
        return arg

    def _analyze_t(self, t_spec, node, path):
        ops = t_spec.__ops__
        root = ops[0]
        if root is not T:
            self._unknown(path, 'scope-based T/S/A expression %r cannot be '
                          'verified without runtime scope' % (t_spec,), t_spec)
            return _UNKNOWN
        current = node
        segments = []
        index = 1
        while index < len(ops):
            op, arg = ops[index], ops[index + 1]
            index += 2
            if op in ('x', 'X'):
                segments.append('*' if op == 'x' else '**')
                current = self._star(current, path + tuple(segments), t_spec)
                continue
            if op == '(':
                self._unknown(path + tuple(segments),
                              'method call in T expression %r is not invoked '
                              'during analysis' % (t_spec,), t_spec)
                current = _UNKNOWN
                break
            if op not in ('.', '[', 'P'):
                self._unknown(path + tuple(segments),
                              'arithmetic T operation %r cannot be statically '
                              'verified' % (t_spec,), t_spec)
                current = _UNKNOWN
                break
            if op == '[' and arg in ('*', '**'):
                segments.append(arg)
                current = self._star(current, path + tuple(segments), t_spec)
                if isinstance(current, _UnknownType):
                    break
                continue
            segments.append(arg)
            current = self._access(current, op, arg, path + tuple(segments), t_spec)
            if isinstance(current, _UnknownType):
                break
        return current

    def _access(self, node, op, arg, full_path, spec):
        """Apply one glom access step; return the resulting type node."""
        if isinstance(node, _StarType):
            inner = self._access(node.item_type, op, arg, full_path, spec)
            if isinstance(inner, _StarType):
                return inner
            return _StarType(inner)
        if isinstance(node, _UnionType):
            return self._access_union(node, op, arg, full_path, spec)
        if isinstance(node, _UnknownType) or isinstance(node, _AnyType):
            return _UNKNOWN
        if isinstance(node, _NoneType):
            self._record(DEFINITELY_MISSING, full_path,
                         'cannot access %r on None' % (arg,), spec)
            return _UNKNOWN

        if op == '.':
            return self._access_attr(node, arg, full_path, spec)
        if op == '[':
            return self._access_item(node, arg, full_path, spec)
        return self._access_fuzzy(node, arg, full_path, spec)

    def _access_fuzzy(self, node, arg, full_path, spec):
        """Model glom's registry-based ``'P'`` path access.

        Runtime order is getitem-style handling for dict/list/tuple,
        then getattr for object-style targets. Static types that
        declare item/field access resolve directly; only plain classes
        fall through to attribute access.
        """
        if isinstance(node, (_TypedDictType, _DataclassType, _DictType,
                             _ListType, _TupleType, _SetType)):
            return self._access_item(node, arg, full_path, spec)
        if isinstance(node, _ClassType):
            return self._access_attr(node, arg, full_path, spec)
        if isinstance(node, _UnionType):
            return self._access_union(node, 'P', arg, full_path, spec)
        return self._access(node, '[', arg, full_path, spec)

    def _access_union(self, node, op, arg, full_path, spec):
        inner, has_none = _split_optional(node)
        if has_none:
            self._record(MAYBE_MISSING, full_path,
                         'target may be None when accessing %r' % (arg,), spec)
        options = inner.options if isinstance(inner, _UnionType) else (inner,)
        results = []
        branch_errors = []
        viable = 0
        for option in options:
            before = len(self.issues)
            result = self._access(option, op, arg, full_path, spec)
            introduced = self.issues[before:]
            errors = [i for i in introduced
                      if i.severity == DEFINITELY_MISSING]
            if errors:
                branch_errors.extend(errors)
            else:
                viable += 1
                results.append(result)
        if viable == 0:
            for issue in branch_errors:
                if issue in self.issues:
                    self.issues.remove(issue)
            self._record(DEFINITELY_MISSING, full_path,
                         'no union member supports access to %r' % (arg,), spec)
            return _UNKNOWN
        if viable != len(options):
            # some members resolve, others do not: the access is legal
            # on part of the union, so it is a runtime risk, not a
            # definite failure of the whole spec
            for issue in branch_errors:
                issue.contained = True
            self._record(MAYBE_MISSING, full_path,
                         'only %d of %d union members support access to %r'
                         % (viable, len(options), arg), spec)
        return _make_union(results) if results else _UNKNOWN

    def _access_attr(self, node, name, full_path, spec):
        if isinstance(node, (_TypedDictType, _DataclassType)):
            if name in node.fields:
                field_node, required = node.fields[name]
                if not required:
                    self._record(MAYBE_MISSING, full_path,
                                 'field %r is optional' % (name,), spec)
                return field_node
            self._record(DEFINITELY_MISSING, full_path,
                         '%s has no declared field %r'
                         % (node, name), spec)
            return _UNKNOWN
        if isinstance(node, _DictType):
            if name in _DICT_ATTR_FIELDS:
                # dict method access (e.g. T.items()) ends in a call,
                # which is treated as unknown downstream
                return _UNKNOWN
            self._record(DEFINITELY_MISSING, full_path,
                         'dict value type has no attribute %r' % (name,), spec)
            return _UNKNOWN
        if isinstance(node, (_ListType, _TupleType, _SetType)):
            self._record(DEFINITELY_MISSING, full_path,
                         'collection has no attribute %r' % (name,), spec)
            return _UNKNOWN
        if isinstance(node, _ClassType):
            # plain runtime class: attribute existence cannot be known
            # from annotations without executing descriptors/getters
            self._unknown(full_path,
                          'attribute %r of %s cannot be verified from '
                          'declared types' % (name, node.cls), spec)
            return _UNKNOWN
        self._unknown(full_path,
                      'attribute %r cannot be verified on unknown type'
                      % (name,), spec)
        return _UNKNOWN

    def _access_item(self, node, key, full_path, spec):
        if isinstance(node, (_TypedDictType, _DataclassType)):
            if key in node.fields:
                field_node, required = node.fields[key]
                if not required:
                    self._record(MAYBE_MISSING, full_path,
                                 'field %r is optional' % (key,), spec)
                return field_node
            self._record(DEFINITELY_MISSING, full_path,
                         '%s has no declared field %r' % (node, key), spec)
            return _UNKNOWN
        if isinstance(node, _DictType):
            # keys are not enumerated statically: any key may be missing
            self._record(MAYBE_MISSING, full_path,
                         'dict key %r may be absent at runtime' % (key,), spec)
            return node.value_type
        if isinstance(node, _ListType):
            if isinstance(key, int):
                self._record(MAYBE_MISSING, full_path,
                             'list index %r may be out of range' % (key,), spec)
                return node.item_type
            self._record(DEFINITELY_MISSING, full_path,
                         'list cannot be indexed by %r' % (key,), spec)
            return _UNKNOWN
        if isinstance(node, _TupleType):
            return self._access_tuple(node, key, full_path, spec)
        if isinstance(node, _SetType):
            self._record(DEFINITELY_MISSING, full_path,
                         'set does not support indexing' , spec)
            return _UNKNOWN
        if isinstance(node, _ClassType):
            self._unknown(full_path,
                          'item access on %s cannot be verified from declared '
                          'types' % (node.cls,), spec)
            return _UNKNOWN
        self._unknown(full_path,
                      'item access %r cannot be verified on unknown type'
                      % (key,), spec)
        return _UNKNOWN

    def _access_tuple(self, node, key, full_path, spec):
        if not isinstance(key, int):
            self._record(DEFINITELY_MISSING, full_path,
                         'tuple cannot be indexed by %r' % (key,), spec)
            return _UNKNOWN
        if node.variadic:
            if key < 0:
                self._record(MAYBE_MISSING, full_path,
                             'tuple index %r may be out of range' % (key,), spec)
            return node.item_types[0]
        adjusted = key + len(node.item_types) if key < 0 else key
        if 0 <= adjusted < len(node.item_types):
            return node.item_types[adjusted]
        self._record(DEFINITELY_MISSING, full_path,
                     'tuple index %r out of range for tuple of length %d'
                     % (key, len(node.item_types)), spec)
        return _UNKNOWN

    def _star(self, node, full_path, spec):
        # '*'/'**' flatten traversal over a collection; preserve the
        # element type and mark the traversal depth as unknown
        if isinstance(node, _ListType):
            return _StarType(node.item_type)
        if isinstance(node, _TupleType):
            if node.variadic:
                return _StarType(node.item_types[0])
            return _StarType(_make_union(node.item_types))
        if isinstance(node, _SetType):
            return _StarType(node.item_type)
        if isinstance(node, _DictType):
            return _StarType(node.value_type)
        if isinstance(node, _StarType):
            return node
        if isinstance(node, (_TypedDictType, _DataclassType, _ClassType)):
            # non-collection traversal target: terminate and keep the
            # branch unknown rather than claiming a verified result
            self._unknown(full_path,
                          'star traversal over non-iterable type %r is '
                          'unverified' % (node,), spec)
            return _StarType(_UNKNOWN)
        self._unknown(full_path,
                      'star traversal over non-collection type %r is unverified'
                      % (node,), spec)
        return _UNKNOWN

    # -- assembly specs ----------------------------------------------------

    def _analyze_dict(self, spec, node, path):
        fields = {}
        for index, (key, subspec) in enumerate(spec.items()):
            key_name = key if isinstance(key, str) else 'field_%d' % index
            fields[key_name] = self._analyze(subspec, node, path + (key_name,))
        if not fields:
            return _DictType(_UNKNOWN, _UNKNOWN)
        return _DictType(_ANY, _make_union(list(fields.values())))

    def _analyze_list(self, spec, node, path):
        if not spec:
            return _ListType(_UNKNOWN)
        # glom lists iterate the target collection; a list spec over a
        # non-iterable target is a definite runtime failure
        element_target = self._iterate_target(node, path, spec)
        element_type = self._analyze(spec[0], element_target, path + (0,))
        if isinstance(element_type, _StarType):
            return _ListType(element_type.item_type)
        return _ListType(element_type)

    def _iterate_target(self, node, path, spec):
        if isinstance(node, _ListType):
            return node.item_type
        if isinstance(node, _TupleType):
            if node.variadic:
                return node.item_types[0]
            return _make_union(node.item_types)
        if isinstance(node, _SetType):
            return node.item_type
        if isinstance(node, _DictType):
            return node
        if isinstance(node, _UnionType):
            inner, has_none = _split_optional(node)
            if has_none:
                self._record(MAYBE_MISSING, path,
                             'iteration target may be None', spec)
            options = inner.options if isinstance(inner, _UnionType) else (inner,)
            return _make_union([self._iterate_target(option, path, spec)
                                for option in options])
        if isinstance(node, (_TypedDictType, _DataclassType, _ClassType)):
            self._record(DEFINITELY_MISSING, path,
                         'type %r is not iterable for a list spec' % (node,), spec)
            return _UNKNOWN
        self._unknown(path, 'iteration over unknown type cannot be verified', spec)
        return _UNKNOWN

    def _analyze_chain(self, steps, node, path):
        current = node
        for index, step in enumerate(steps):
            current = self._analyze(step, current, path + (index,))
            if isinstance(current, _UnknownType):
                # later steps have no static target to check
                for offset, remaining in enumerate(steps[index + 1:], start=1):
                    self._analyze(remaining, _UNKNOWN, path + (index + offset,))
                return _UNKNOWN
        return current

    def _analyze_coalesce(self, spec, node, path):
        if not spec.subspecs:
            has_fallback = (spec.default is not _MISSING
                            or spec.default_factory is not _MISSING)
            if has_fallback:
                return self._analyze_default(spec, path)
            self._unknown(path, 'Coalesce has no subspecs and no default', spec)
            return _UNKNOWN
        candidates = []
        branch_error_indices = []
        for index, subspec in enumerate(spec.subspecs):
            branch_path = path + ('coalesce_%d' % index,)
            before = len(self.issues)
            result = self._analyze(subspec, node, branch_path)
            branch_errors = [i for i in self.issues[before:]
                             if i.severity == DEFINITELY_MISSING]
            if branch_errors:
                branch_error_indices.append((index, branch_errors))
            else:
                candidates.append(result)

        has_fallback = (spec.default is not _MISSING
                        or spec.default_factory is not _MISSING)

        if not candidates:
            if has_fallback:
                self._contain_branch_errors(branch_error_indices)
                self._record(MAYBE_MISSING, path,
                             'all %d Coalesce branches miss on the declared '
                             'type; default value/factory is used at runtime'
                             % len(spec.subspecs), spec)
                return self._analyze_default(spec, path)
            # hard failure: branch errors already recorded stay definite
            return _UNKNOWN

        if branch_error_indices:
            # at least one branch resolves, so the spec can run; the
            # definite branch errors are contained by Coalesce and do
            # not make the whole spec definitely wrong
            self._contain_branch_errors(branch_error_indices)
            self._record(MAYBE_MISSING, path,
                         '%d of %d Coalesce branches cannot resolve on the '
                         'declared type but other branches or a default apply'
                         % (len(branch_error_indices), len(spec.subspecs)),
                         spec)

        if has_fallback:
            candidates.append(self._analyze_default(spec, path))
        return _make_union(candidates)

    def _contain_branch_errors(self, branch_error_indices):
        """Branch-level definite errors stay visible but are contained.

        Coalesce catches their runtime exceptions, so they cannot fail
        the whole spec; mark them contained instead of deleting them.
        """
        for _index, branch_errors in branch_error_indices:
            for issue in branch_errors:
                issue.contained = True

    def _analyze_default(self, spec, path):
        if spec.default_factory is not _MISSING:
            self._unknown(path,
                          'Coalesce default_factory is not called during '
                          'analysis', spec)
            return _UNKNOWN
        # a literal default value carries its runtime type
        default = spec.default
        if default is None:
            return _NONE
        return self.normalizer.normalize(type(default))

    # -- values, calls and wrappers ----------------------------------------

    def _analyze_val(self, spec):
        value = spec.value
        if value is None:
            return _NONE
        if isinstance(value, (str, int, float, bool, bytes)):
            return self.normalizer.normalize(type(value))
        if isinstance(value, (list, tuple, dict, set, frozenset)):
            self._unknown((), 'Val wraps a mutable literal of unverified type', spec)
            return _UNKNOWN
        return _UNKNOWN

    def _analyze_fill(self, spec, node, path):
        # Fill mode only interprets explicit specifier types; plain
        # tuples/lists/dicts are constructed literally
        return self._analyze_literal(spec.spec, node, path)

    def _analyze_literal(self, spec, node, path):
        if isinstance(spec, (Path, TType, Spec, Coalesce, Call, Invoke)):
            return self._analyze(spec, node, path)
        if isinstance(spec, dict):
            value_types = [self._analyze_literal(v, node, path + (k,))
                           for k, v in spec.items()]
            return _DictType(_ANY, _make_union(value_types) if value_types else _UNKNOWN)
        if isinstance(spec, list):
            item_types = [self._analyze_literal(v, node, path + (i,))
                          for i, v in enumerate(spec)]
            return _ListType(_make_union(item_types) if item_types else _UNKNOWN)
        if isinstance(spec, tuple):
            return _TupleType([self._analyze_literal(v, node, path + (i,))
                               for i, v in enumerate(spec)])
        if isinstance(spec, set):
            item_types = [self._analyze_literal(v, node, path) for v in spec]
            return _SetType(_make_union(item_types) if item_types else _UNKNOWN)
        if isinstance(spec, Val):
            return self._analyze_val(spec)
        if spec is None:
            return _NONE
        if isinstance(spec, (str, int, float, bool, bytes)):
            return self.normalizer.normalize(type(spec))
        self._unknown(path, 'Fill literal %r cannot be typed' % (spec,), spec)
        return _UNKNOWN

    def _analyze_callable(self, func, node, path):
        # never call the function; rely solely on declared annotations
        annotation = self._return_annotation(func)
        if annotation is _inspect.Parameter.empty:
            self._unknown(path,
                          'callable %r has no declared return type'
                          % (getattr(func, '__name__', func),), func)
            return _UNKNOWN
        return self.normalizer.normalize(annotation)

    @staticmethod
    def _return_annotation(func):
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = getattr(func, '__annotations__', {})
        return hints.get('return', _inspect.Parameter.empty)

    def _analyze_call(self, spec, node, path):
        if isinstance(spec, Call):
            func = spec.func
        else:
            func = spec.func
        if isinstance(func, (Spec, TType)):
            self._unknown(path,
                          'dynamic call target %r cannot be resolved without '
                          'execution' % (func,), spec)
            return _UNKNOWN
        if not callable(func):
            self._unknown(path, 'non-callable target %r' % (func,), spec)
            return _UNKNOWN
        # args/kwargs may themselves contain specs; analyze them for
        # field errors against the current target, but never invoke
        self._analyze_call_args(spec, node, path)
        return self._analyze_callable(func, node, path)

    def _analyze_call_args(self, spec, node, path):
        if isinstance(spec, Call):
            arg_groups = ((spec.args, False), (spec.kwargs, True))
            for args, is_kwargs in arg_groups:
                self._scan_arg_container(args, node, path, is_kwargs)
            return
        # Invoke stores templated ('C'/'S'/'*', args, kwargs) triples
        flat = getattr(spec, '_args', ())
        for index in range(len(flat) // 3):
            kind, args, kwargs = flat[index * 3: index * 3 + 3]
            if kind == 'C':
                continue  # constants need no target analysis
            if kind == '*':
                if args is not None:
                    self._analyze_if_spec(args, node, path + (index,))
                if kwargs is not None:
                    self._analyze_if_spec(kwargs, node, path + (index,))
                continue
            self._scan_arg_container(args, node, path, False)
            self._scan_arg_container(kwargs, node, path, True)

    def _scan_arg_container(self, container, node, path, is_kwargs):
        if not container:
            return
        if is_kwargs and isinstance(container, dict):
            for key, value in container.items():
                self._analyze_if_spec(value, node, path + (key,))
        elif isinstance(container, (tuple, list)):
            for index, value in enumerate(container):
                self._analyze_if_spec(value, node, path + (index,))

    def _analyze_if_spec(self, value, node, path):
        if self._is_analyzable(value):
            self._analyze(value, node, path)

    @staticmethod
    def _is_analyzable(value):
        return isinstance(value, (str, Path, TType, dict, list, tuple,
                                  Coalesce, Spec, Pipe))

    def _analyze_ref(self, spec, node, path):
        if spec.subspec is _MISSING:
            self._unknown(path,
                          'Ref %r references a runtime-registered spec'
                          % (spec.name,), spec)
            return _UNKNOWN
        if id(spec.subspec) in self._ref_stack:
            self._unknown(path,
                          'recursive Ref %r terminates with an unresolved '
                          'branch' % (spec.name,), spec)
            return _UNKNOWN
        self._ref_stack.add(id(spec.subspec))
        try:
            return self._analyze(spec.subspec, node, path)
        finally:
            self._ref_stack.discard(id(spec.subspec))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def analyze_spec(target_type, spec):
    """Statically analyze *spec* against the declared *target_type*.

    The analysis is read-only: no instance of *target_type* is
    constructed, glom's registry is not modified, and none of the
    functions or getters referenced by *spec* are executed. Only
    declared type information drives the result.

    Args:
       target_type: a :class:`~typing.TypedDict` subclass,
          :func:`~dataclasses.dataclass`, or any type built with
          :mod:`typing` (``Optional[...]``, ``list[...]``, nested
          dicts, recursive aliases, ...).
       spec: the glom spec to analyze.

    Returns:
       SpecAnalysis: the inferred output type plus all definite
       errors, possibly-missing paths, and unknown branches.
    """
    analyzer = _SpecAnalyzer(target_type)
    output_node = analyzer.analyze(spec)
    return SpecAnalysis(target_type, _render_type(output_node), analyzer.issues)
