"""Bidirectional, restricted glom specifications.

glom specs are intentionally expressive: callables, defaults, wildcards,
and iteration can all change the shape and cardinality of data.  Those
features cannot generally be inverted.  :class:`BidiSpec` provides a
small, explicit spec language which *can* be inverted:

* field paths (strings, :class:`~glom.Path`, or ``T`` access paths),
* object assembly with dictionaries, and
* fixed-length list/tuple assembly.

The object validates the mapping when it is built, so unsupported or
ambiguous specs fail early rather than producing a lossy inverse.
"""


import copy

from boltons.typeutils import make_sentinel

from glom.core import (Auto,
                       BadSpec,
                       Fill,
                       Path,
                       Spec,
                       T,
                       TType,
                       glom,
                       _MISSING)


_MAPPED = make_sentinel('_MAPPED')
_UNFILLED = make_sentinel('_UNFILLED')
_NO_OUTPUT = make_sentinel('_NO_OUTPUT')

__all__ = [
    'BidiSpec',
    'BidiSidecar',
    'Bidi',
    'BidiMap',
    'BidiMapping',
    'BiMap',
    'Bijection',
    'BijectionSpec',
    'Bidirectional',
    'BidirectionalSpec',
    'BidirectionalMapping',
    'InvertibleSpec',
    'InvertibleMapping',
    'ReversibleSpec',
    'ReversibleMapping',
    'bidi_spec',
    'bidi',
    'bijection',
    'bidi_map',
    'bidirectional_spec',
    'invertible_spec',
    'reversible_spec',
    'make_bidi',
    'inverse_spec',
    'reverse_spec',
    'BidiSpecError',
    'BidiError',
    'BijectionError',
    'NotInvertibleError',
    'NotReversibleError',
    'DuplicateSourceError',
    'ParentChildConflictError',
    'SequenceConflictError',
    'UnmappedFieldError',
    'UnmappedSourceError',
    'UnmappedOutputError',
]


class BidiSpecError(BadSpec):
    """Base error for invalid or non-invertible bidirectional specs."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def get_message(self):
        return self.message


class NotInvertibleError(BidiSpecError):
    """The spec contains a construct whose inverse cannot be inferred."""


class DuplicateSourceError(BidiSpecError):
    """Two output values refer to the same source path."""


class ParentChildConflictError(BidiSpecError):
    """A source path is both a value and the parent of another value."""


class SequenceConflictError(BidiSpecError):
    """Sequence positions cannot be reconstructed unambiguously."""


class UnmappedFieldError(BidiSpecError):
    """Fields outside the mapping were found under the current policy."""


class BidiSidecar:
    """Container for fields not represented by a :class:`BidiSpec`.

    ``data`` holds a nested tree of unmapped values.  A partially mapped
    sequence is represented by a dictionary of the remaining integer
    indices, leaving mapped positions out rather than conflating them
    with ``None``.
    """

    __slots__ = ('_raw_data',)

    def __init__(self, data):
        self._raw_data = data

    @property
    def data(self):
        if self._raw_data is _MAPPED:
            return {}
        return self._raw_data

    @property
    def raw(self):
        return self._raw_data

    @property
    def empty(self):
        return self._raw_data is _MAPPED

    def _data(self):
        return self.data

    def __getitem__(self, key):
        return self._data()[key]

    def __iter__(self):
        return iter(self._data())

    def __len__(self):
        return len(self._data())

    def __bool__(self):
        return not self.empty

    def __contains__(self, key):
        return key in self._data()

    def get(self, key, default=None):
        return self._data().get(key, default)

    def keys(self):
        return self._data().keys()

    def values(self):
        return self._data().values()

    def items(self):
        return self._data().items()

    def as_dict(self):
        return self.data

    def __eq__(self, other):
        if isinstance(other, BidiSidecar):
            other = other.raw
            if self.raw is _MAPPED:
                return other is _MAPPED
            if other is _MAPPED:
                return False
            return self.raw == other
        if self.raw is _MAPPED:
            return other == {}
        return self.data == other

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if self.empty:
            return 'BidiSidecar(<empty>)'
        return f'BidiSidecar({self.data!r})'


class _PathLeaf:
    __slots__ = ('source_path', 'source_parts', 'output_parts')

    def __init__(self, source_path, source_parts, output_parts):
        self.source_path = source_path
        self.source_parts = source_parts
        self.output_parts = output_parts


class _DictNode:
    __slots__ = ('children',)

    def __init__(self, children):
        self.children = children


class _SeqNode:
    __slots__ = ('children', 'seq_type')

    def __init__(self, children, seq_type):
        self.children = children
        self.seq_type = seq_type


class _BuildNode:
    __slots__ = ('children', 'leaf')

    def __init__(self):
        self.children = {}
        self.leaf = _MISSING


class _Policy:
    ERROR = 'error'
    SIDECAR = 'sidecar'
    IGNORE = 'ignore'

    _ALIASES = {
        'error': ERROR,
        'reject': ERROR,
        'strict': ERROR,
        'raise': ERROR,
        'sidecar': SIDECAR,
        'preserve': SIDECAR,
        'keep': SIDECAR,
        'ignore': IGNORE,
        'drop': IGNORE,
    }

    @classmethod
    def normalize(cls, value):
        if value is None:
            return cls.IGNORE
        try:
            return cls._ALIASES[value]
        except (KeyError, TypeError):
            raise BidiSpecError(
                "unmapped must be one of 'error', 'sidecar', or 'ignore',"
                f" not {value!r}")


def _is_spec_wrapper(value):
    return isinstance(value, (Spec, Auto, Fill))


def _unwrap_spec(value):
    fill_mode = False
    while _is_spec_wrapper(value):
        if isinstance(value, Spec) and value.scope:
            raise NotInvertibleError(
                'Spec objects with a custom scope cannot be inverted')
        if isinstance(value, Fill):
            fill_mode = True
        value = value.spec
    return value, fill_mode


def _extract_path(value):
    if isinstance(value, str):
        return Path.from_text(value)
    if isinstance(value, Path):
        return value
    if isinstance(value, TType):
        try:
            return Path(value)
        except ValueError as exc:
            raise NotInvertibleError(
                f'path must start from T, not {value!r}') from exc
    return None


def _path_parts(path):
    path_t = path.path_t
    ops = path_t.__ops__
    if ops[0] is not T:
        raise NotInvertibleError(
            f'path must start from T, not {ops[0]!r}: {path!r}')

    parts = []
    i = 1
    while i < len(ops):
        op, arg = ops[i], ops[i + 1]
        if op not in ('P', '['):
            raise NotInvertibleError(
                'only explicit field/item paths can be inverted;'
                f' found operation {op!r} in {path!r}')
        if op == '[' and isinstance(arg, slice):
            raise NotInvertibleError(
                f'slices cannot be inverted deterministically: {path!r}')
        try:
            hash(arg)
        except TypeError as exc:
            raise NotInvertibleError(
                f'path component is not hashable: {arg!r}') from exc
        parts.append(arg)
        i += 2
    return tuple(parts)


def _check_hashable(value, label):
    try:
        hash(value)
    except TypeError as exc:
        raise NotInvertibleError(
            f'{label} must be hashable, got {value!r}') from exc


def _compile_output(spec, output_parts, leaves, fill_mode=False):
    spec, nested_fill_mode = _unwrap_spec(spec)
    fill_mode = fill_mode or nested_fill_mode
    path = _extract_path(spec)
    if path is not None:
        if fill_mode and isinstance(spec, str):
            raise NotInvertibleError(
                'strings inside Fill are literal values, not source paths;'
                f' use Path(...) for {spec!r}')
        source_parts = _path_parts(path)
        leaves.append(_PathLeaf(path, source_parts, output_parts))
        return _PathLeaf(path, source_parts, output_parts)

    if isinstance(spec, dict):
        children = {}
        for key, value in spec.items():
            if isinstance(key, (Spec, TType)):
                raise NotInvertibleError(
                    f'dynamic dictionary key is not invertible: {key!r}')
            _check_hashable(key, 'dictionary key')
            children[key] = _compile_output(
                value, output_parts + (key,), leaves, fill_mode=fill_mode)
        return _DictNode(children)

    if isinstance(spec, (list, tuple)):
        children = []
        for index, value in enumerate(spec):
            children.append(_compile_output(
                value, output_parts + (index,), leaves, fill_mode=fill_mode))
        # Keep the container type for forward assembly.  On the way back,
        # the source structure is the authority for container type.
        seq_type = type(spec)
        return _SeqNode(tuple(children), seq_type)

    raise NotInvertibleError(
        'cannot infer an inverse for spec value'
        f' {spec!r}; use a path, dict, or fixed-length list/tuple')


def _forward_spec(node):
    if isinstance(node, _PathLeaf):
        return node.source_path
    if isinstance(node, _DictNode):
        return {key: _forward_spec(value)
                for key, value in node.children.items()}
    if isinstance(node, _SeqNode):
        return Fill(node.seq_type(
            [_forward_spec(value) for value in node.children]))
    raise TypeError(f'unknown compiled node: {node!r}')  # pragma: no cover


def _insert_source(root, parts, leaf):
    node = root
    for part_index, part in enumerate(parts):
        if node.leaf is not _MISSING:
            raise ParentChildConflictError(
                f'source path {leaf.source_path!r} is nested under'
                f' mapped path {Path(*parts[:part_index])!r}')
        try:
            node = node.children[part]
        except KeyError:
            child = _BuildNode()
            node.children[part] = child
            node = child

    if node.leaf is not _MISSING:
        raise DuplicateSourceError(
            f'multiple outputs refer to source path {leaf.source_path!r}')
    if node.children:
        raise ParentChildConflictError(
            f'source path {leaf.source_path!r} is also the parent'
            ' of another mapped path')
    node.leaf = leaf


def _validate_source_tree(node, prefix=()):
    if node.leaf is not _MISSING:
        return
    args = tuple(node.children)
    int_args = [arg for arg in args if type(arg) is int]
    if int_args:
        if len(int_args) != len(args):
            raise ParentChildConflictError(
                f'mixed sequence indices and keys below path {Path(*prefix)!r}')
        if any(arg < 0 for arg in int_args):
            raise SequenceConflictError(
                f'negative sequence index below path {Path(*prefix)!r}')
        expected = list(range(len(int_args)))
        if sorted(int_args) != expected:
            raise SequenceConflictError(
                'fixed sequence positions under'
                f' {Path(*prefix)!r} must cover 0..{len(int_args) - 1},'
                f' got {sorted(int_args)!r}')
    for arg, child in node.children.items():
        _validate_source_tree(child, prefix + (arg,))


def _build_output_tree(leaves):
    root = _BuildNode()
    for leaf in leaves:
        node = root
        for part in leaf.output_parts:
            try:
                node = node.children[part]
            except KeyError:
                child = _BuildNode()
                node.children[part] = child
                node = child
        if node.children:
            raise ParentChildConflictError(
                f'output path {Path(*leaf.output_parts)!r} is also a parent')
        node.leaf = leaf
    return root


def _materialize(node):
    if node.leaf is not _MISSING:
        return node.leaf

    args = tuple(node.children)
    if args and all(type(arg) is int for arg in args):
        result = [_UNFILLED] * len(args)
        for index, child in node.children.items():
            result[index] = _materialize(child)
        if any(value is _UNFILLED for value in result):
            raise SequenceConflictError(  # pragma: no cover
                'internal sequence hole during reconstruction')
        return result

    result = {}
    for key, child in node.children.items():
        result[key] = _materialize(child)
    return result


def _copy_value(value):
    try:
        return copy.deepcopy(value)
    except Exception:
        # Inputs are never mutated.  If an exotic value cannot be
        # deep-copied, retaining the reference is still safe for the
        # read-only sidecar use-case.
        return value


def _strip_unmapped(value, node):
    if node.leaf is not _MISSING:
        return _MAPPED
    if not node.children:
        return _copy_value(value)

    if isinstance(value, dict):
        residual = {}
        for key, child_value in value.items():
            child_node = node.children.get(key)
            if child_node is None:
                residual[key] = _copy_value(child_value)
                continue
            stripped = _strip_unmapped(child_value, child_node)
            if stripped is not _MAPPED:
                residual[key] = stripped
        return residual if residual else _MAPPED

    if isinstance(value, (list, tuple)):
        residual = {}
        for index, child_value in enumerate(value):
            child_node = node.children.get(index)
            if child_node is None:
                residual[index] = _copy_value(child_value)
                continue
            stripped = _strip_unmapped(child_value, child_node)
            if stripped is not _MAPPED:
                residual[index] = stripped
        return residual if residual else _MAPPED

    return _copy_value(value)


def _merge_sidecar(reconstructed, residual, path=()):
    if residual is _MAPPED:
        return reconstructed

    if reconstructed is _MISSING:
        return _copy_value(residual)

    if isinstance(residual, dict):
        if isinstance(reconstructed, list):
            result = list(reconstructed)
            for index, value in residual.items():
                if type(index) is not int or index < 0:
                    raise UnmappedFieldError(
                        f'invalid sequence sidecar index at {Path(*path)!r}:'
                        f' {index!r}')
                while len(result) < index:
                    result.append(_UNFILLED)
                if index < len(result):
                    if result[index] is _UNFILLED:
                        result[index] = _copy_value(value)
                    else:
                        result[index] = _merge_sidecar(
                            result[index], value, path + (index,))
                else:
                    result.append(_copy_value(value))
            if any(value is _UNFILLED for value in result):
                raise UnmappedFieldError(
                    f'missing sequence positions in sidecar at {Path(*path)!r}')
            return result

        if isinstance(reconstructed, dict):
            result = reconstructed
            for key, value in residual.items():
                if key in result:
                    result[key] = _merge_sidecar(
                        result[key], value, path + (key,))
                else:
                    result[key] = _copy_value(value)
            return result

        if reconstructed is not _MISSING:
            raise UnmappedFieldError(
                f'sidecar type conflicts with mapped value at {Path(*path)!r}')

    raise UnmappedFieldError(
        f'sidecar conflicts with mapped value at {Path(*path)!r}')


class BidiSpec:
    """A restricted, independently invertible glom specification.

    The supported forward spec is deliberately small:

    * paths (``'a.b'``, :class:`~glom.Path`, or ``T['a']['b']``),
    * dictionaries, which assemble a new object, and
    * fixed-length lists/tuples, whose elements assemble by position.

    Each output leaf must refer to exactly one source path.  A source
    path may not be reused, and a path may not be both a value and the
    parent of another mapped value.  Sequence parents must cover their
    positions contiguously from zero.

    Args:
        spec: The restricted glom spec to make bidirectional.
        unmapped: Policy for source fields not represented by *spec*.
            ``'error'`` (the default) rejects them, ``'sidecar'`` allows
            them to be extracted with :meth:`extract_sidecar`, and
            ``'ignore'`` drops them from the reconstructed source.

    Both :meth:`forward` and :meth:`inverse` use glom's normal path
    access and error wrapping.  Neither direction mutates its input.
    """

    def __init__(self, spec, unmapped='error', **kwargs):
        on_unmapped = kwargs.pop('on_unmapped', None)
        unmapped_fields = kwargs.pop('unmapped_fields', None)
        keep_unmapped = kwargs.pop('keep_unmapped', None)
        reject_unmapped = kwargs.pop('reject_unmapped', None)
        sidecar = kwargs.pop('sidecar', None)
        if kwargs:
            unexpected = ', '.join(sorted(kwargs))
            raise TypeError(f'unexpected keyword arguments: {unexpected}')
        choices = [value for value in
                   (on_unmapped, unmapped_fields, sidecar) if value is not None]
        if keep_unmapped:
            choices.append(_Policy.SIDECAR)
        if reject_unmapped:
            choices.append(_Policy.ERROR)
        if choices:
            normalized = {_Policy.normalize(value) for value in choices}
            if len(normalized) > 1:
                raise BidiSpecError('conflicting unmapped-field policies')
            unmapped = normalized.pop()

        self.spec = spec
        self.unmapped = _Policy.normalize(unmapped)
        self.unmapped_policy = self.unmapped
        self.policy = self.unmapped

        self._leaves = []
        self._root = _compile_output(spec, (), self._leaves)
        if not self._leaves:
            raise NotInvertibleError(
                'a bidirectional spec must contain at least one source path')

        self._source_tree = _BuildNode()
        for leaf in self._leaves:
            _insert_source(self._source_tree, leaf.source_parts, leaf)
        _validate_source_tree(self._source_tree)

        self._output_tree = _build_output_tree(self._leaves)
        self._forward_spec = _forward_spec(self._root)

    @property
    def forward_spec(self):
        """The ordinary glom spec used for the forward direction."""
        return self._forward_spec

    @property
    def source_paths(self):
        """Tuple of source :class:`~glom.Path` objects, in output order."""
        return tuple(leaf.source_path for leaf in self._leaves)

    @property
    def output_paths(self):
        """Tuple of output :class:`~glom.Path` objects, in output order."""
        return tuple(Path(*leaf.output_parts) for leaf in self._leaves)

    def _check_source(self, target):
        if self.unmapped != _Policy.ERROR:
            return
        residual = _strip_unmapped(target, self._source_tree)
        if residual is not _MAPPED:
            raise UnmappedFieldError(
                'source contains fields outside the bidirectional mapping;'
                " use unmapped='sidecar' to preserve them")

    def _check_output(self, output):
        if self.unmapped != _Policy.ERROR:
            return
        residual = _strip_unmapped(output, self._output_tree)
        if residual is not _MAPPED:
            raise UnmappedFieldError(
                'output contains fields outside the bidirectional mapping;'
                " use unmapped='sidecar' or unmapped='ignore'")

    def forward(self, target, with_sidecar=False, return_sidecar=None,
                sidecar=False, **kw):
        """Map *target* from the original shape to the output shape."""
        if return_sidecar is not None:
            with_sidecar = return_sidecar
        if sidecar:
            with_sidecar = True
        if 'default' in kw or 'skip_exc' in kw:
            raise NotInvertibleError(
                'default/skip_exc can hide missing values and are not'
                ' invertible')
        ret = glom(target, self._forward_spec, **kw)
        if not with_sidecar:
            self._check_source(target)
        if with_sidecar:
            return ret, self.extract_sidecar(target)
        return ret

    def inverse(self, output=_NO_OUTPUT, sidecar=None, with_sidecar=False,
                return_sidecar=None, **kw):
        """Map *output* back to the original shape.

        *sidecar* may be a :class:`BidiSidecar`, its ``data``, or an
        equivalent nested structure produced by :meth:`extract_sidecar`.

        Calling ``inverse()`` without *output* returns a lightweight
        reverse view, whose ``forward()`` is this object's
        :meth:`inverse` and whose ``inverse()`` is this object's
        :meth:`forward`.
        """
        if output is _NO_OUTPUT:
            return _BidiReverse(self)
        if return_sidecar is not None:
            with_sidecar = return_sidecar
        if 'default' in kw or 'skip_exc' in kw:
            raise NotInvertibleError(
                'default/skip_exc can hide missing values and are not'
                ' invertible')
        if not with_sidecar:
            self._check_output(output)

        root = _BuildNode()
        for leaf in self._leaves:
            value = glom(output, Path(*leaf.output_parts), **kw)
            node = root
            for part in leaf.source_parts:
                try:
                    node = node.children[part]
                except KeyError:
                    child = _BuildNode()
                    node.children[part] = child
                    node = child
            node.leaf = value

        reconstructed = _materialize(root)

        sidecar_data = _MAPPED
        if sidecar is not None:
            if isinstance(sidecar, BidiSidecar):
                sidecar_data = sidecar.raw
            else:
                sidecar_data = sidecar
        reconstructed = _merge_sidecar(reconstructed, sidecar_data)
        if reconstructed is _MISSING:
            reconstructed = {}

        if with_sidecar:
            return reconstructed, self.extract_output_sidecar(output)
        return reconstructed

    def reverse(self, output=_NO_OUTPUT, **kw):
        """Alias for :meth:`inverse`."""
        return self.inverse(output, **kw)

    backward = reverse
    unapply = reverse
    decode = reverse

    def roundtrip(self, target, **kw):
        """Forward *target* and immediately map it back."""
        output, sidecar = self.forward(target, with_sidecar=True, **kw)
        return self.inverse(output, sidecar=sidecar, **kw)

    round_trip = roundtrip

    def invert(self):
        """Return a reverse view of this mapping."""
        return _BidiReverse(self)

    @property
    def inverted(self):
        return self.invert()

    def extract_sidecar(self, target):
        """Return a :class:`BidiSidecar` of source fields outside the map."""
        return BidiSidecar(_strip_unmapped(target, self._source_tree))

    sidecar = extract_sidecar

    def extract_output_sidecar(self, output):
        """Return a :class:`BidiSidecar` of output fields outside the map."""
        return BidiSidecar(_strip_unmapped(output, self._output_tree))

    def glomit(self, target, scope):
        ret = scope[glom](target, self._forward_spec, scope)
        self._check_source(target)
        return ret

    def __call__(self, target, **kw):
        return self.forward(target, **kw)

    apply = forward
    encode = forward
    to_output = forward
    from_original = forward
    from_output = inverse
    to_original = inverse

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.spec!r},'
                f' unmapped={self.unmapped!r})')


class _BidiReverse:
    """A reversed view over a :class:`BidiSpec`."""

    def __init__(self, mapper):
        self.mapper = mapper

    @property
    def spec(self):
        return self.mapper.forward_spec

    @property
    def source_paths(self):
        return self.mapper.output_paths

    @property
    def output_paths(self):
        return self.mapper.source_paths

    def forward(self, output, **kw):
        return self.mapper.inverse(output, **kw)

    def inverse(self, target, **kw):
        return self.mapper.forward(target, **kw)

    def invert(self):
        return self.mapper

    @property
    def inverted(self):
        return self.mapper

    apply = forward
    encode = forward
    reverse = inverse
    backward = inverse
    unapply = inverse
    decode = inverse

    def __call__(self, output, **kw):
        return self.forward(output, **kw)

    def __repr__(self):
        return f'<{self.__class__.__name__} of {self.mapper!r}>'


def bidi_spec(spec, **kw):
    """Convenience constructor for :class:`BidiSpec`."""
    return BidiSpec(spec, **kw)


bidi = bidi_spec


# Descriptive aliases for discoverability.
Bidi = BidiSpec
BidiMap = BidiSpec
BidiMapping = BidiSpec
BiMap = BidiSpec
Bijection = BidiSpec
BijectionSpec = BidiSpec
Bidirectional = BidiSpec
BidirectionalSpec = BidiSpec
BidirectionalMapping = BidiSpec
InvertibleSpec = BidiSpec
InvertibleMapping = BidiSpec
ReversibleSpec = BidiSpec
ReversibleMapping = BidiSpec

bijection = bidi_spec
bidi_map = bidi_spec
bidirectional_spec = bidi_spec
invertible_spec = bidi_spec
reversible_spec = bidi_spec
make_bidi = bidi_spec


def inverse_spec(spec, **kw):
    """Make the reverse view of a :class:`BidiSpec`."""
    return BidiSpec(spec, **kw).invert()


reverse_spec = inverse_spec

# Error aliases follow the same naming pattern.
BidiError = BidiSpecError
BijectionError = BidiSpecError
NotReversibleError = NotInvertibleError
UnmappedSourceError = UnmappedFieldError
UnmappedOutputError = UnmappedFieldError
