from __future__ import annotations

import dataclasses
import math
import struct
import unicodedata
import copy
import threading
import pickle
import importlib.util
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

if importlib.util.find_spec("numpy") is None:  # pragma: no cover - deterministic import guard
    raise ModuleNotFoundError(
        "The numpy package is required for the Sera runtime. Install it with 'pip install numpy'."
    )
import numpy as np
import hashlib
import json
import zlib

try:
    from .sera_common import (
        ARRAY_HEADER_STRUCT as _TRANSFER_ARRAY_STRUCT,
        ARRAY_MAGIC as _TRANSFER_ARRAY_MAGIC,
        DEFAULT_STATE_FILENAMES as _TRANSFER_STATE_FILENAMES,
        DTYPE_CODES as _TRANSFER_DTYPE_CODES,
        JSON_BYTES_PREFIX as _TRANSFER_JSON_BYTES_PREFIX,
        MANIFEST_MAGIC as _TRANSFER_MANIFEST_MAGIC,
        MANIFEST_VERSION as _TRANSFER_MANIFEST_VERSION,
        PICKLE_SUFFIXES as _PICKLE_SUFFIXES,
    )
except ImportError:  # pragma: no cover - fallback for direct loading
    _sera_common_path = Path(__file__).resolve().parent / "sera_common.py"
    _sera_common_spec = importlib.util.spec_from_file_location(
        "_sera_common", _sera_common_path
    )
    if _sera_common_spec is None or _sera_common_spec.loader is None:
        raise
    _sera_common_module = importlib.util.module_from_spec(_sera_common_spec)
    sys.modules.setdefault(_sera_common_spec.name, _sera_common_module)
    _sera_common_spec.loader.exec_module(_sera_common_module)
    _TRANSFER_ARRAY_STRUCT = _sera_common_module.ARRAY_HEADER_STRUCT
    _TRANSFER_ARRAY_MAGIC = _sera_common_module.ARRAY_MAGIC
    _TRANSFER_STATE_FILENAMES = _sera_common_module.DEFAULT_STATE_FILENAMES
    _TRANSFER_DTYPE_CODES = _sera_common_module.DTYPE_CODES
    _TRANSFER_JSON_BYTES_PREFIX = _sera_common_module.JSON_BYTES_PREFIX
    _TRANSFER_MANIFEST_MAGIC = _sera_common_module.MANIFEST_MAGIC
    _TRANSFER_MANIFEST_VERSION = _sera_common_module.MANIFEST_VERSION
    _PICKLE_SUFFIXES = _sera_common_module.PICKLE_SUFFIXES


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


class BudgetError(RuntimeError):
    """Raised when a runtime budget specified by the Sera spec is violated."""


T = TypeVar("T")


_TRANSFER_MANIFEST_PREFIX_STRUCT = struct.Struct("<I I")
_TRANSFER_EXPECTED_SEED_DIGEST = hashlib.sha256(b"sera-transfer").digest()
_CRC32C_TABLE: List[int] = []
_CRC32C_INIT = 0xFFFFFFFF
_ARRAY_VALIDATION_CHUNK_SIZE = 1024 * 1024


def _coerce_int(value: object) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return int(value)
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> Optional[float]:
    try:
        if isinstance(value, bool):
            return float(value)
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bytes_like(value: object) -> Optional[bytes]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        if value.startswith(_TRANSFER_JSON_BYTES_PREFIX):
            try:
                return bytes.fromhex(value[len(_TRANSFER_JSON_BYTES_PREFIX) :])
            except ValueError:
                return value.encode("utf-8")
        return value.encode("utf-8")
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        try:
            return bytes(int(v) & 0xFF for v in value)
        except (TypeError, ValueError):
            return None
    return None


def _validate_prefix_free(vocabulary: Dict[bytes, int]) -> None:
    """Ensure the vocabulary is prefix-free (spec §2.1).

    The Sardinas–Patterson proof in the specification guarantees decode
    uniqueness.  For the reference implementation we adopt the stricter
    prefix-free condition and verify it eagerly when the tokenizer is
    initialised.  This keeps the encoder and decoder logic simple while still
    satisfying the spec.
    """

    ordered = sorted(vocabulary.keys())
    for i, piece in enumerate(ordered):
        for other in ordered[i + 1 :]:
            if other.startswith(piece):
                raise ValueError(
                    "Tokenizer vocabulary must be prefix-free: "
                    f"{piece!r} is a prefix of {other!r}"
                )


def _sigmoid(x: float) -> float:
    # Stable sigmoid used for the linear-as-gate rule (spec §6.2).
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(x: float) -> float:
    if not 0.0 < x < 1.0:
        raise ValueError("logit is defined on (0, 1)")
    return math.log(x / (1.0 - x))


def _safe_tanh(value: float, scale: float = 1.0) -> float:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if not math.isfinite(value):
        return 0.0
    return math.tanh(value / scale)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if minimum > maximum:
        raise ValueError("invalid clamp bounds")
    return max(minimum, min(maximum, value))


def _kahan_update(sum_: float, c: float, value: float) -> Tuple[float, float]:
    """Apply a single Kahan compensated summation step (spec §5.1)."""

    y = value - c
    t = sum_ + y
    c_new = (t - sum_) - y
    return t, c_new


def _coerce_dataclass_config(
    value: object,
    cls: Type[T],
    factory: Callable[[], T],
) -> T:
    """Return an instance of ``cls`` merging ``value`` with default fields.

    ``SeraConfig`` accepts dictionaries for nested configuration sections so
    that callers can override only the parameters they care about.  The helper
    centralises the merging logic to guarantee that every field inherits
    defaults from the canonical configuration.  ``factory`` is evaluated for
    each coercion to avoid sharing mutable defaults between ``SeraConfig``
    instances.
    """

    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        default = factory()
        init_fields = {field.name for field in dataclasses.fields(cls) if field.init}
        merged = {name: getattr(default, name) for name in init_fields}
        for key, val in value.items():
            if key in init_fields:
                merged[key] = val
        return cls(**merged)  # type: ignore[arg-type]
    raise TypeError(f"Expected {cls.__name__} or dict, got {type(value)!r}")


@dataclass(frozen=True)
class _MPHEntry:
    key_hash: int
    token_id: int
    piece: bytes


class _MPHTable:
    """Two-level minimal perfect hash for tokenizer lookups (spec §4.4)."""

    __slots__ = ("size", "seeds", "entries")

    def __init__(self, size: int, seeds: List[int], entries: List[Optional[_MPHEntry]]):
        self.size = size
        self.seeds = seeds
        self.entries = entries

    @staticmethod
    def build(pieces: List[Tuple[bytes, int]]) -> "_MPHTable":
        if not pieces:
            return _MPHTable(0, [], [])
        key_records: List[Tuple[int, int, bytes]] = []
        for piece, token_id in pieces:
            key_hash = _hash_bytes(piece)
            key_records.append((key_hash, token_id, piece))
        size = max(1, math.ceil(1.23 * len(key_records)))
        seeds = [0 for _ in range(size)]
        entries: List[Optional[_MPHEntry]] = [None for _ in range(size)]
        occupied = [False for _ in range(size)]
        buckets: Dict[int, List[Tuple[int, int, bytes]]] = {}
        for record in key_records:
            bucket = _mix64(record[0]) % size
            buckets.setdefault(bucket, []).append(record)
        # Deterministic processing order for reproducibility.
        for bucket in sorted(buckets.keys(), key=lambda b: (len(buckets[b]), b), reverse=True):
            items = sorted(buckets[bucket], key=lambda rec: (rec[0], rec[1], rec[2]))
            seed, assignment = _MPHTable._find_seed(items, occupied, size)
            seeds[bucket] = seed
            for rec, position in zip(items, assignment):
                entries[position] = _MPHEntry(*rec)
                occupied[position] = True
        return _MPHTable(size, seeds, entries)

    @staticmethod
    def _find_seed(
        items: List[Tuple[int, int, bytes]], occupied: List[bool], size: int
    ) -> Tuple[int, List[int]]:
        if not items:
            return 0, []
        for seed in range(0, 1 << 16):
            options: List[Tuple[int, int]] = []
            for key_hash, _token_id, _piece in items:
                pos0 = _mix64(key_hash, seed) % size
                pos1 = _mix64(key_hash, seed + 1) % size
                options.append((pos0, pos1))
            assignment = _MPHTable._assign_positions(options, occupied)
            if assignment is not None:
                return seed, assignment
        raise RuntimeError("Unable to construct tokenizer minimal perfect hash (seed cap exceeded)")

    @staticmethod
    def _assign_positions(
        options: List[Tuple[int, int]], occupied: List[bool]
    ) -> Optional[List[int]]:
        assignment: List[int] = []
        used: Set[int] = set()

        def backtrack(idx: int) -> bool:
            if idx == len(options):
                return True
            candidates = sorted(set(options[idx]))
            for pos in candidates:
                if occupied[pos] or pos in used:
                    continue
                used.add(pos)
                assignment.append(pos)
                if backtrack(idx + 1):
                    return True
                assignment.pop()
                used.remove(pos)
            return False

        if backtrack(0):
            return assignment.copy()
        return None

    def lookup(self, key_hash: int, candidate: bytes) -> Tuple[Optional[int], int]:
        if self.size == 0:
            return None, 0
        bucket = _mix64(key_hash) % self.size
        seed = self.seeds[bucket]
        probes = 0
        for offset in (0, 1):
            position = _mix64(key_hash, seed + offset) % self.size
            entry = self.entries[position]
            probes += 1
            if entry is not None and entry.key_hash == key_hash and entry.piece == candidate:
                return entry.token_id, probes
        return None, probes


@dataclass(frozen=True)
class _LinearMPHEntry:
    key: int
    slot: int


class _LinearMPHTable:
    """Minimal perfect hash specialised for sparse linear handles (spec §4.4)."""

    __slots__ = ("size", "seeds", "entries")

    def __init__(self, size: int, seeds: List[int], entries: List[Optional[_LinearMPHEntry]]):
        self.size = size
        self.seeds = seeds
        self.entries = entries

    @staticmethod
    def build(items: List[Tuple[int, int]]) -> "_LinearMPHTable":
        if not items:
            return _LinearMPHTable(0, [], [])
        records = sorted({(int(key), int(slot)) for key, slot in items})
        size = max(1, math.ceil(1.23 * len(records)))
        seeds = [0 for _ in range(size)]
        entries: List[Optional[_LinearMPHEntry]] = [None for _ in range(size)]
        occupied = [False for _ in range(size)]
        buckets: Dict[int, List[Tuple[int, int]]] = {}
        for key, slot in records:
            bucket = _mix64(key) % size
            buckets.setdefault(bucket, []).append((key, slot))
        for bucket in sorted(buckets.keys(), key=lambda b: (len(buckets[b]), b), reverse=True):
            bucket_items = sorted(buckets[bucket], key=lambda rec: (rec[0], rec[1]))
            seed, assignment = _LinearMPHTable._find_seed(bucket_items, occupied, size)
            seeds[bucket] = seed
            for (key, slot), position in zip(bucket_items, assignment):
                entries[position] = _LinearMPHEntry(key=key, slot=slot)
                occupied[position] = True
        return _LinearMPHTable(size, seeds, entries)

    @staticmethod
    def _find_seed(
        items: List[Tuple[int, int]], occupied: List[bool], size: int
    ) -> Tuple[int, List[int]]:
        if not items:
            return 0, []
        for seed in range(0, 1 << 16):
            options: List[Tuple[int, int]] = []
            for key, _slot in items:
                pos0 = _mix64(key, seed) % size
                pos1 = _mix64(key, seed + 1) % size
                options.append((pos0, pos1))
            assignment = _MPHTable._assign_positions(options, occupied)
            if assignment is not None:
                return seed, assignment
        raise RuntimeError("Unable to construct linear minimal perfect hash (seed cap exceeded)")

    def manifest(self) -> Dict[str, object]:
        return {
            "size": int(self.size),
            "seeds": [int(seed) for seed in self.seeds],
            "entries": [
                None
                if entry is None
                else {"key": int(entry.key), "slot": int(entry.slot)}
                for entry in self.entries
            ],
        }

# ---------------------------------------------------------------------------
# Tokenizer (spec section 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenizerConfig:
    """Compile-time configuration parameters for the tokenizer."""

    max_piece_length: int = 4
    normalizer_states: int = 8
    vocabulary: Dict[bytes, int] = dataclasses.field(default_factory=dict)
    edit_window: int = 4
    max_event_bytes: int = 64
    max_event_tokens: int = 64

    def __post_init__(self) -> None:
        if self.max_piece_length <= 0:
            raise ValueError("max_piece_length must be positive")
        if self.normalizer_states <= 0:
            raise ValueError("normalizer_states must be positive")
        if self.edit_window < self.max_piece_length:
            raise ValueError("edit_window must be >= max_piece_length")
        if self.max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be positive")
        if self.max_event_tokens <= 0:
            raise ValueError("max_event_tokens must be positive")
        if self.max_event_tokens < self.max_piece_length:
            raise ValueError("max_event_tokens must be >= max_piece_length")
        if not self.vocabulary:
            # Provide a default byte-level vocabulary when none is supplied.
            vocabulary = {bytes([i]): i for i in range(256)}
            object.__setattr__(self, "vocabulary", vocabulary)
        _validate_prefix_free(self.vocabulary)


@dataclass
class TokenizerEncodeStats:
    """Runtime diagnostics collected by :meth:`TokenizerState.encode`."""

    bytes_in: int
    normalized_bytes: int
    tokens_out: int
    max_probes: int
    table_probes: int


@dataclass
class TokenizerState:
    """State for the constant-time tokenizer (spec §2)."""

    config: TokenizerConfig
    _id_to_bytes: Dict[int, bytes] = field(init=False)
    _last_stats: Optional[TokenizerEncodeStats] = field(init=False, default=None)
    _hash_powers: List[int] = field(init=False)
    _mph_tables: Dict[int, _MPHTable] = field(init=False)
    _sp_trace: List[Tuple[int, Tuple[bytes, ...]]] = field(init=False)
    _sp_digest: str = field(init=False)

    def __post_init__(self) -> None:
        self._id_to_bytes = {idx: piece for piece, idx in self.config.vocabulary.items()}
        if len(self._id_to_bytes) != len(self.config.vocabulary):
            raise ValueError("Vocabulary ids must be unique")
        self._hash_powers = [1]
        for _ in range(1, self.config.max_piece_length + 1):
            self._hash_powers.append((self._hash_powers[-1] * _HASH_BASE) & _HASH_MASK)
        self._mph_tables = self._build_mph_tables()
        self._sp_trace, self._sp_digest = self._build_sp_trace()

    # -- Normaliser -----------------------------------------------------

    def _normalise_bytes(self, data: bytes) -> bytes:
        """Normalise UTF-8 bytes using NFC with bounded lookahead (spec §2.2)."""

        if not data:
            return data
        text = data.decode("utf-8", errors="strict")
        for codepoint in map(ord, text):
            if codepoint in _DISALLOWED_UNICODE_POINTS:
                raise ValueError(
                    "Input contains disallowed Unicode control character"
                )
        # The NFC transform used here has a finite lookahead that is bounded by
        # the Unicode specification.  For our tests this is sufficient to meet
        # the requirement L_norm<=4 in the spec.
        text = unicodedata.normalize("NFC", text)
        return text.encode("utf-8")

    # -- Encoder --------------------------------------------------------

    def encode(self, data: bytes) -> List[int]:
        """Encode bytes into token ids following the algorithm in spec §2.4."""

        normalised = self._normalise_bytes(data)
        if len(normalised) > self.config.max_event_bytes:
            raise BudgetError("Tokenizer byte budget exceeded")
        tokens = self._encode_from_normalised(normalised, len(data), update_stats=True)
        return tokens

    # -- Decoder --------------------------------------------------------

    def decode(self, tokens: Sequence[int]) -> bytes:
        """Decode token ids back to bytes (spec §2.5)."""

        if len(tokens) > self.config.max_event_tokens:
            raise BudgetError("Tokenizer token budget exceeded")
        pieces = []
        for token in tokens:
            piece = self._id_to_bytes.get(token)
            if piece is None:
                raise KeyError(f"Unknown token id {token}")
            pieces.append(piece)
        return b"".join(pieces)

    @property
    def last_encode_stats(self) -> Optional[TokenizerEncodeStats]:
        """Return diagnostics from the most recent :meth:`encode` call."""

        return self._last_stats

    @property
    def sp_trace(self) -> Sequence[Tuple[int, Tuple[bytes, ...]]]:
        """Expose the Sardinas–Patterson witness trace (spec §4.2)."""

        return tuple(self._sp_trace)

    @property
    def sp_digest(self) -> str:
        """Return the SHA-256 digest of the minimal SP witness."""

        return self._sp_digest

    def retokenize_window(
        self,
        normalised: bytes,
        tokens: Sequence[int],
        edit_start: int,
        edit_end: int,
        replacement: bytes,
    ) -> Tuple[List[int], Tuple[int, int], bytes]:
        """Retokenize a local edit inside the radius ``W_edit`` (spec §2.6).

        ``normalised`` and ``replacement`` must already satisfy the Unicode
        normalisation policy enforced by :meth:`encode`.  ``edit_start`` and
        ``edit_end`` are byte offsets in ``normalised`` describing the replaced
        span.  The method retokenizes only the affected window and returns the
        replacement tokens, the token index range to splice, and the updated
        normalised byte string.
        """

        if edit_start < 0 or edit_end < edit_start or edit_end > len(normalised):
            raise ValueError("Invalid edit span for retokenization")
        if self.decode(tokens) != normalised:
            raise ValueError("Token sequence does not match the provided bytes")
        if replacement != self._normalise_bytes(replacement):
            raise ValueError("Replacement bytes must already be normalised")
        radius = self.config.edit_window
        window_start = max(0, edit_start - radius)
        window_end_before = min(len(normalised), edit_end + radius)
        edit_replacement_end = edit_start + len(replacement)
        new_normalised = normalised[:edit_start] + replacement + normalised[edit_end:]
        window_end_after = min(len(new_normalised), edit_replacement_end + radius)
        token_start, token_end = self._token_window(tokens, window_start, window_end_before)
        segment = new_normalised[window_start:window_end_after]
        new_tokens = self._encode_from_normalised(segment, len(segment), update_stats=False)
        return new_tokens, (token_start, token_end), new_normalised

    def _encode_from_normalised(
        self, normalised: bytes, bytes_in: int, update_stats: bool
    ) -> List[int]:
        if len(normalised) > self.config.max_event_bytes:
            raise BudgetError("Tokenizer byte budget exceeded")
        prefix_hash = [0] * (len(normalised) + 1)
        for i, byte in enumerate(normalised):
            prefix_hash[i + 1] = (
                (prefix_hash[i] * _HASH_BASE + byte) & _HASH_MASK
            )
        output: List[int] = []
        i = 0
        max_len = self.config.max_piece_length
        max_probe = 0
        total_table_probes = 0
        while i < len(normalised):
            remaining = len(normalised) - i
            attempts = 0
            matched = False
            for length in range(min(max_len, remaining), 0, -1):
                table = self._mph_tables.get(length)
                if table is None:
                    continue
                attempts += 1
                key_hash = self._substring_hash(prefix_hash, i, length)
                candidate = normalised[i : i + length]
                token_id, probes = table.lookup(key_hash, candidate)
                total_table_probes += probes
                if token_id is not None:
                    output.append(token_id)
                    i += length
                    matched = True
                    break
            if not matched:
                # Fallback to single-byte atom per spec §2.4.
                table = self._mph_tables.get(1)
                if table is None:
                    raise KeyError("Single-byte vocabulary missing")
                key_hash = self._substring_hash(prefix_hash, i, 1)
                candidate = normalised[i : i + 1]
                token_id, probes = table.lookup(key_hash, candidate)
                total_table_probes += probes
                if token_id is None:
                    raise KeyError(f"Byte {candidate!r} missing from vocabulary")
                output.append(token_id)
                i += 1
                attempts += 1
            if attempts > self.config.max_piece_length:
                raise BudgetError("Tokenizer probe budget exceeded")
            max_probe = max(max_probe, attempts)
            if len(output) > self.config.max_event_tokens:
                raise BudgetError("Tokenizer token budget exceeded")
        if update_stats:
            self._last_stats = TokenizerEncodeStats(
                bytes_in=bytes_in,
                normalized_bytes=len(normalised),
                tokens_out=len(output),
                max_probes=max_probe,
                table_probes=total_table_probes,
            )
        return output

    def _substring_hash(
        self, prefix_hash: Sequence[int], start: int, length: int
    ) -> int:
        end = start + length
        value = (
            prefix_hash[end]
            - (prefix_hash[start] * self._hash_powers[length])
        ) & _HASH_MASK
        return value

    def _token_window(
        self, tokens: Sequence[int], start_byte: int, end_byte: int
    ) -> Tuple[int, int]:
        if start_byte >= end_byte:
            return len(tokens), len(tokens)
        offsets: List[Tuple[int, int]] = []
        offset = 0
        for token in tokens:
            piece = self._id_to_bytes.get(int(token))
            if piece is None:
                raise KeyError(f"Unknown token id {token}")
            next_offset = offset + len(piece)
            offsets.append((offset, next_offset))
            offset = next_offset
        start_idx = len(tokens)
        end_idx = len(tokens)
        for idx, (start, end) in enumerate(offsets):
            if start_idx == len(tokens) and end > start_byte:
                start_idx = idx
            if end >= end_byte:
                end_idx = idx + 1
                break
        return start_idx, end_idx

    def _build_mph_tables(self) -> Dict[int, _MPHTable]:
        tables: Dict[int, _MPHTable] = {}
        pieces_by_length: Dict[int, List[Tuple[bytes, int]]] = {}
        for piece, token_id in self.config.vocabulary.items():
            pieces_by_length.setdefault(len(piece), []).append((piece, token_id))
        for length, pieces in pieces_by_length.items():
            tables[length] = _MPHTable.build(pieces)
        return tables

    def _build_sp_trace(self) -> Tuple[List[Tuple[int, Tuple[bytes, ...]]], str]:
        vocab = [piece for piece in self.config.vocabulary.keys()]
        trace: List[Tuple[int, Tuple[bytes, ...]]] = []
        visited: Set[Tuple[bytes, ...]] = set()
        current: Set[bytes] = set()
        for x in vocab:
            for y in vocab:
                if x == y:
                    continue
                if x.startswith(y):
                    remainder = x[len(y) :]
                    if not remainder:
                        raise ValueError("Vocabulary fails Sardinas–Patterson (epsilon witness)")
                    current.add(remainder)
        while current:
            if b"" in current:
                raise ValueError("Vocabulary fails Sardinas–Patterson (epsilon witness)")
            snapshot = tuple(sorted(current))
            trace.append((len(trace) + 1, snapshot))
            visited.add(snapshot)
            next_set: Set[bytes] = set()
            for word in current:
                for piece in vocab:
                    if word.startswith(piece):
                        remainder = word[len(piece) :]
                        if remainder:
                            next_set.add(remainder)
                        else:
                            raise ValueError("Vocabulary fails Sardinas–Patterson (epsilon witness)")
                    if piece.startswith(word):
                        remainder = piece[len(word) :]
                        if remainder:
                            next_set.add(remainder)
                        else:
                            raise ValueError("Vocabulary fails Sardinas–Patterson (epsilon witness)")
            canonical = tuple(sorted(next_set))
            if canonical in visited:
                break
            current = next_set
        digest_payload = bytearray()
        for level, sequences in trace:
            digest_payload.extend(level.to_bytes(2, "big"))
            for remainder in sequences:
                digest_payload.extend(len(remainder).to_bytes(2, "big"))
                digest_payload.extend(remainder)
        digest = hashlib.sha256(bytes(digest_payload)).hexdigest()
        return trace, digest


# ---------------------------------------------------------------------------
# PRF attention (spec section 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PRFAttentionConfig:
    dim: int
    value_dim: int
    features: int
    gamma: float
    tau: float
    beta_floor: float
    epsilon: float = 1e-6
    clip_value: Optional[float] = None
    rng_seed: int = 17

    def __post_init__(self) -> None:
        if not (0 < self.gamma < 1):
            raise ValueError("gamma must be in (0,1)")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.beta_floor <= 0:
            raise ValueError("beta_floor must be positive")
        if self.features <= 0:
            raise ValueError("features must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")


@dataclass
class PRFAttentionState:
    config: PRFAttentionConfig
    _weights: np.ndarray = field(init=False)
    _R: np.ndarray = field(init=False)
    _s: np.ndarray = field(init=False)
    _mu: np.ndarray = field(init=False)
    _sig2: np.ndarray = field(init=False)
    _R_comp: np.ndarray = field(init=False)
    _s_comp: np.ndarray = field(init=False)
    _mu_comp: np.ndarray = field(init=False)
    _sig2_comp: np.ndarray = field(init=False)
    _clip_events: int = field(init=False, default=0)
    _clip_candidates: int = field(init=False, default=0)
    _last_denominator: Optional[float] = field(init=False, default=None)

    # Overlay budgets (spec §3.4)
    type_a: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    type_b: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    type_c_H: Optional[np.ndarray] = None
    type_c_deltaW: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.config.rng_seed)
        self._weights = rng.standard_normal((self.config.features, self.config.dim))
        self._R = np.zeros((self.config.features, self.config.value_dim), dtype=float)
        self._R_comp = np.zeros_like(self._R)
        self._s = np.zeros(self.config.features, dtype=float)
        self._s_comp = np.zeros_like(self._s)
        self._mu = np.zeros(self.config.features, dtype=float)
        self._mu_comp = np.zeros_like(self._mu)
        self._sig2 = np.ones(self.config.features, dtype=float)
        self._sig2_comp = np.zeros_like(self._sig2)

    # -- Random feature map (spec §3.1) --------------------------------

    def _phi(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.config.dim,):
            raise ValueError(f"Input must be shape {(self.config.dim,)}, got {x.shape}")
        scaled = self._weights @ x / math.sqrt(self.config.tau)
        norm = np.dot(x, x)
        raw_phi = np.exp(scaled - norm / (2 * self.config.tau)) / math.sqrt(
            self.config.features
        )
        clip_value = self.config.clip_value
        if clip_value is not None:
            clipped_mask = np.abs(raw_phi) > clip_value
            self._clip_events += int(np.count_nonzero(clipped_mask))
            phi = np.clip(raw_phi, -clip_value, clip_value)
        else:
            phi = raw_phi
        self._clip_candidates += raw_phi.size
        return phi

    def _phi_whitened(self, x: np.ndarray) -> np.ndarray:
        phi = self._phi(x)
        centred = phi - self._mu
        denom = np.sqrt(self._sig2 + self.config.epsilon)
        return centred / denom

    # -- Streaming update (spec §3.2) ----------------------------------

    def update(self, key: np.ndarray, value: np.ndarray) -> None:
        key = np.asarray(key, dtype=float)
        value = np.asarray(value, dtype=float)
        if key.shape != (self.config.dim,):
            raise ValueError("Key shape mismatch")
        if value.shape != (self.config.value_dim,):
            raise ValueError("Value shape mismatch")
        phi_k = self._phi(key)
        gamma = self.config.gamma
        self._R, self._R_comp = self._decay_matrix(self._R, self._R_comp, gamma)
        outer = np.outer(phi_k, value)
        self._R, self._R_comp = self._accumulate_matrix(self._R, self._R_comp, outer)

        self._s, self._s_comp = self._decay_vector(self._s, self._s_comp, gamma)
        self._s, self._s_comp = self._accumulate_vector(self._s, self._s_comp, phi_k)

        # Running moments for whitening (compensated EMA)
        self._mu, self._mu_comp = self._decay_vector(self._mu, self._mu_comp, gamma)
        ema_update = (1 - gamma) * phi_k
        self._mu, self._mu_comp = self._accumulate_vector(self._mu, self._mu_comp, ema_update)
        centred = phi_k - self._mu
        self._sig2, self._sig2_comp = self._decay_vector(self._sig2, self._sig2_comp, gamma)
        sig2_update = (1 - gamma) * centred**2
        self._sig2, self._sig2_comp = self._accumulate_vector(self._sig2, self._sig2_comp, sig2_update)

    # -- Base readout (spec §3.3) --------------------------------------

    def read(self, query: np.ndarray, lambda_floor: float) -> Tuple[np.ndarray, float, np.ndarray]:
        phi_q = self._phi_whitened(query)
        numerator = phi_q @ self._R
        denominator = float(np.dot(phi_q, self._s) + lambda_floor)
        if denominator < self.config.beta_floor:
            denominator = self.config.beta_floor
        self._last_denominator = denominator
        return numerator, denominator, phi_q

    # -- Overlay application (spec §3.4) -------------------------------

    def apply_overlays(self, phi_q: np.ndarray) -> Tuple[np.ndarray, float]:
        delta_num = np.zeros(self.config.value_dim, dtype=float)
        delta_den = 0.0
        for basis, vector in self.type_a:
            delta_num += float(np.dot(phi_q, basis)) * vector
        for basis, beta in self.type_b:
            delta_den += float(np.dot(phi_q, basis)) * beta
        if self.type_c_H is not None and self.type_c_deltaW is not None:
            z = phi_q @ self.type_c_H
            delta_num += z @ self.type_c_deltaW
        return delta_num, delta_den

    def read_with_overlays(self, query: np.ndarray, lambda_floor: float) -> Tuple[np.ndarray, np.ndarray]:
        base_num, base_den, phi_q = self.read(query, lambda_floor)
        delta_num, delta_den = self.apply_overlays(phi_q)
        denominator = base_den + delta_den
        if denominator < self.config.beta_floor:
            denominator = self.config.beta_floor
        result = (base_num + delta_num) / denominator
        self._last_denominator = denominator
        return result, phi_q

    # -- Internal helpers ----------------------------------------------

    def _decay_vector(
        self, array: np.ndarray, comp: np.ndarray, gamma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        array = gamma * array
        comp = gamma * comp
        return array, comp

    def _accumulate_vector(
        self, array: np.ndarray, comp: np.ndarray, update: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(array.shape[0]):
            array[i], comp[i] = _kahan_update(array[i], comp[i], float(update[i]))
        return array, comp

    def _decay_matrix(
        self, matrix: np.ndarray, comp: np.ndarray, gamma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        matrix = gamma * matrix
        comp = gamma * comp
        return matrix, comp

    @property
    def clip_rate(self) -> float:
        if self._clip_candidates == 0:
            return 0.0
        return float(self._clip_events) / float(self._clip_candidates)

    @property
    def last_denominator(self) -> Optional[float]:
        return self._last_denominator

    def _accumulate_matrix(
        self, matrix: np.ndarray, comp: np.ndarray, update: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                matrix[i, j], comp[i, j] = _kahan_update(
                    matrix[i, j], comp[i, j], float(update[i, j])
                )
        return matrix, comp


# ---------------------------------------------------------------------------
# Sparse linear learner (spec section 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseLinearConfig:
    capacity: int
    tau_low: float
    tau_high: float
    learning_rate: float
    l2: float = 0.0
    buckets: int = 4
    bucket_size: int = 2
    stash_capacity: int = 8
    max_kicks: int = 8
    ring_capacity: int = 16
    margin: float = 4.0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0 < self.tau_low < self.tau_high < 1):
            raise ValueError("tau_low < tau_high < 1 must hold")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.buckets <= 0:
            raise ValueError("buckets must be positive")
        if self.bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if self.stash_capacity < 0:
            raise ValueError("stash_capacity must be non-negative")
        if self.max_kicks <= 0:
            raise ValueError("max_kicks must be positive")
        if self.ring_capacity < 0:
            raise ValueError("ring_capacity must be non-negative")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")


@dataclass(frozen=True)
class InjectiveHandle:
    """Stable external handle for a sparse linear key (spec §4.3)."""

    generation: int
    slot: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.generation, self.slot)


@dataclass
class InjectiveAddressBook:
    """Tracks injective key -> slot assignments within a generation."""

    generation: int = 0
    _key_to_slot: Dict[int, int] = field(default_factory=dict)
    _next_slot: int = 0

    def slot_of(self, key: int) -> Optional[InjectiveHandle]:
        slot = self._key_to_slot.get(key)
        if slot is None:
            return None
        return InjectiveHandle(self.generation, slot)

    def ensure_slot(self, key: int) -> Tuple[InjectiveHandle, bool]:
        created = False
        slot = self._key_to_slot.get(key)
        if slot is None:
            slot = self._next_slot
            self._key_to_slot[key] = slot
            self._next_slot += 1
            created = True
        return InjectiveHandle(self.generation, slot), created

    @property
    def size(self) -> int:
        return len(self._key_to_slot)

    def snapshot(self) -> Dict[str, object]:
        return {
            "generation": self.generation,
            "key_to_slot": dict(self._key_to_slot),
            "next_slot": self._next_slot,
        }

    @classmethod
    def restore(cls, blob: Dict[str, object]) -> "InjectiveAddressBook":
        book = cls(generation=int(blob["generation"]))
        book._key_to_slot = {int(k): int(v) for k, v in blob["key_to_slot"].items()}
        book._next_slot = int(blob["next_slot"])
        return book

    def manifest(self) -> List[Dict[str, int]]:
        return [
            {"key": int(key), "slot": int(slot), "generation": self.generation}
            for key, slot in sorted(self._key_to_slot.items())
        ]

    def advance_generation(self, generation: int) -> None:
        self.generation = generation


@dataclass
class SparseLinearState:
    config: SparseLinearConfig
    _address_book: InjectiveAddressBook = field(default_factory=InjectiveAddressBook)
    _weights: BoundedCuckooMap = field(init=False, repr=False)
    _bias: float = 0.0
    _capacity_monitor: LinearCapacityMonitor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_weights",
            BoundedCuckooMap(
                capacity=self.config.capacity,
                choices=self.config.buckets,
                bucket_size=self.config.bucket_size,
                stash_capacity=self.config.stash_capacity,
                max_kicks=self.config.max_kicks,
                ring_capacity=self.config.ring_capacity,
            ),
        )
        object.__setattr__(
            self, "_capacity_monitor", LinearCapacityMonitor(self.config)
        )

    def predict(self, features: Iterable[Tuple[int, float]]) -> float:
        total = self._bias
        comp = 0.0
        for idx, value in features:
            handle = self._address_book.slot_of(int(idx))
            if handle is None:
                weight = 0.0
            else:
                weight = self._weights.get(handle.slot, 0.0)
            total, comp = _kahan_update(total, comp, weight * value)
        return total

    def update(self, features: Iterable[Tuple[int, float]], target: float) -> None:
        prediction = self.predict(features)
        error = prediction - target
        lr = self.config.learning_rate
        for idx, value in features:
            key = int(idx)
            handle = self._address_book.slot_of(key)
            created = False
            if handle is None:
                projected_load = (self._address_book.size + 1) / float(self.config.capacity)
                if self._capacity_monitor.should_freeze(projected_load):
                    self._capacity_monitor.freeze(projected_load)
                    raise BudgetError("Sparse linear insert frozen due to slack")
                handle, created = self._address_book.ensure_slot(key)
            if created and self._address_book.size > self.config.capacity:
                del self._address_book._key_to_slot[key]
                self._address_book._next_slot -= 1
                self._capacity_monitor.freeze(self._address_book.size / float(self.config.capacity))
                raise BudgetError("Sparse linear capacity exceeded")
            load = self._address_book.size / float(self.config.capacity)
            self._capacity_monitor.observe(created, load)
            weight = self._weights.get(handle.slot, 0.0)
            grad = error * value + self.config.l2 * weight
            self._weights.insert(handle.slot, weight - lr * grad)
        self._bias -= lr * (error + self.config.l2 * self._bias)

    def handles_for(self, keys: Iterable[int]) -> List[InjectiveHandle]:
        return [handle for key in keys if (handle := self._address_book.slot_of(int(key))) is not None]

    def snapshot(self) -> Dict[str, object]:
        return {
            "address_book": self._address_book.snapshot(),
            "weights": self._weights.snapshot(),
            "bias": float(self._bias),
            "capacity": self._capacity_monitor.snapshot(),
        }

    @classmethod
    def restore(cls, config: SparseLinearConfig, blob: Dict[str, object]) -> "SparseLinearState":
        state = cls(config)
        state._address_book = InjectiveAddressBook.restore(blob["address_book"])
        weights_blob = blob.get("weights", {})
        if isinstance(weights_blob, dict):
            state._weights.restore(weights_blob)
        state._bias = float(blob["bias"])
        capacity_blob = blob.get("capacity")
        if capacity_blob is not None:
            state._capacity_monitor = LinearCapacityMonitor.restore(state.config, capacity_blob)
        state._capacity_monitor.recompute(
            state._address_book.size / float(state.config.capacity)
        )
        return state

    def manifest(self) -> Dict[str, object]:
        key_to_slot = dict(self._address_book._key_to_slot)
        mph = _LinearMPHTable.build(list(key_to_slot.items()))
        handles = [
            {
                "key": int(key),
                "slot": int(slot),
                "generation": self._address_book.generation,
            }
            for key, slot in sorted(key_to_slot.items())
        ]
        return {
            "handles": {
                "generation": self._address_book.generation,
                "size": self._address_book.size,
                "mph": mph.manifest(),
                "list": handles,
            },
            "weights": self._weights.manifest(),
            "bias": float(self._bias),
            "capacity": self._capacity_monitor.manifest(),
        }

    def advance_generation(self, generation: int) -> None:
        self._address_book.advance_generation(generation)
        self._capacity_monitor.recompute(
            self._address_book.size / float(self.config.capacity)
        )


# ---------------------------------------------------------------------------
# Finite rational memory (spec section 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiniteMemoryConfig:
    lift_coordinates: Sequence[int]
    max_active: int
    delay: int
    a_coeffs: Sequence[float] = ()
    b_coeffs: Sequence[float] = (1.0,)
    input_clip: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_active <= 0:
            raise ValueError("max_active must be positive")
        if self.delay < 0:
            raise ValueError("delay must be non-negative")
        if not self.b_coeffs:
            raise ValueError("b_coeffs must not be empty")
        if self.input_clip is not None and self.input_clip <= 0:
            raise ValueError("input_clip must be positive")
        # Normalise coefficients to tuples for fast access during runtime.
        object.__setattr__(self, "a_coeffs", tuple(float(x) for x in self.a_coeffs))
        object.__setattr__(self, "b_coeffs", tuple(float(x) for x in self.b_coeffs))


@dataclass
class FiniteMemoryState:
    config: FiniteMemoryConfig
    _accumulators: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    _delay_buffer: List[float] = field(init=False)
    _delay_index: int = 0
    _lift_set: Set[int] = field(init=False)
    _input_history: List[float] = field(init=False)
    _output_history: List[float] = field(init=False)

    def __post_init__(self) -> None:
        self._delay_buffer = [0.0 for _ in range(max(self.config.delay, 1))]
        self._lift_set = set(self.config.lift_coordinates)
        self._input_history = [0.0 for _ in range(len(self.config.b_coeffs))]
        self._output_history = [0.0 for _ in range(len(self.config.a_coeffs))]

    def accumulate(self, lifts: Iterable[Tuple[int, float]]) -> float:
        count = 0
        aggregate = 0.0
        aggregate_comp = 0.0
        for coord, value in lifts:
            if coord not in self._lift_set:
                continue
            if count >= self.config.max_active:
                raise BudgetError("Lift activation budget exceeded")
            count += 1
            total, comp = self._accumulators.get(coord, (0.0, 0.0))
            total, comp = _kahan_update(total, comp, value)
            self._accumulators[coord] = (total, comp)
            aggregate, aggregate_comp = _kahan_update(aggregate, aggregate_comp, value)

        if self.config.input_clip is not None:
            limit = self.config.input_clip
            aggregate = max(-limit, min(limit, aggregate))

        delayed_input = self._delay_buffer[self._delay_index]
        self._delay_buffer[self._delay_index] = aggregate
        self._delay_index = (self._delay_index + 1) % len(self._delay_buffer)

        if self._input_history:
            self._input_history.insert(0, delayed_input)
            del self._input_history[len(self.config.b_coeffs) :]

        output = 0.0
        comp = 0.0
        for coeff, past in zip(self.config.a_coeffs, self._output_history):
            output, comp = _kahan_update(output, comp, coeff * past)
        for coeff, past in zip(self.config.b_coeffs, self._input_history):
            output, comp = _kahan_update(output, comp, coeff * past)

        if self._output_history:
            self._output_history.insert(0, output)
            del self._output_history[len(self.config.a_coeffs) :]

        return output


# ---------------------------------------------------------------------------
# Fusion and gating (spec section 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionConfig:
    w_att: float = 0.5
    w_lin: float = 0.5
    gate_vector: Tuple[float, float] = (1.0, 1.0)


@dataclass
class FusionState:
    config: FusionConfig

    def fuse(self, y_att: np.ndarray, y_lin: float) -> float:
        if y_att is None:
            return y_lin
        return float(self.config.w_att * float(np.mean(y_att)) + self.config.w_lin * y_lin)

    def gate(self, y_att: np.ndarray, y_lin: float) -> float:
        if y_att is None:
            return y_lin
        z1, z2 = self.config.gate_vector
        gate = _sigmoid(z1 * float(np.mean(y_att)) + z2 * y_lin)
        return gate * float(np.mean(y_att)) + (1 - gate) * y_lin


# ---------------------------------------------------------------------------
# Trust gate (spec section 6.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustGateConfig:
    dimension: int = 16
    salts: int = 5
    acceptance_k: int = 3
    neg_threshold: float = -0.25
    pos_threshold: float = 0.1
    q_min_hat: float = 0.6
    eps_pos_hat: float = 0.2
    B_floor: float = 6.0
    pi0: float = 0.25
    hazard_positive: float = 5.0
    hazard_negative: float = 1.0
    hazard_offset: float = 1.0
    psi: float = 0.5
    beta_min: float = 0.05
    beta_max: float = 0.4
    beta_boost: float = 0.1
    alpha_unknown: float = 0.5
    _logit_pi0: float = field(init=False, repr=False)
    _log_bf_pos: float = field(init=False, repr=False)
    _gamma_threshold: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.salts <= 0:
            raise ValueError("salts must be positive")
        if not 0 < self.acceptance_k <= self.salts:
            raise ValueError("acceptance_k must be in (0, salts]")
        if self.neg_threshold >= self.pos_threshold:
            raise ValueError("neg_threshold must be < pos_threshold")
        if self.beta_min < 0 or self.beta_max < 0:
            raise ValueError("beta bounds must be non-negative")
        if self.beta_min > self.beta_max:
            raise ValueError("beta_min cannot exceed beta_max")
        if not 0 < self.alpha_unknown <= 1:
            raise ValueError("alpha_unknown must be in (0, 1]")
        logit_pi0 = _logit(self.pi0)
        denom = max(self.eps_pos_hat, 10.0 ** (-self.B_floor))
        bf_pos = self.q_min_hat / denom
        if bf_pos <= 1.0:
            raise ValueError("bf_pos must exceed 1 for a meaningful LLR")
        log_bf = math.log(bf_pos)
        gamma = (
            math.log((self.hazard_positive + self.hazard_offset) / (self.hazard_negative + self.hazard_offset))
            - logit_pi0
            + self.psi
        )
        m_star = math.ceil((gamma - logit_pi0) / log_bf)
        if m_star < self.acceptance_k:
            raise ValueError("m_star must be >= acceptance_k for publication")
        object.__setattr__(self, "_logit_pi0", logit_pi0)
        object.__setattr__(self, "_log_bf_pos", log_bf)
        object.__setattr__(self, "_gamma_threshold", gamma)

    @property
    def logit_pi0(self) -> float:
        return self._logit_pi0

    @property
    def log_bf_pos(self) -> float:
        return self._log_bf_pos

    @property
    def gamma_threshold(self) -> float:
        return self._gamma_threshold


@dataclass(frozen=True)
class TrustGateDecision:
    decision: int
    m: int
    llr: float
    gamma: float
    consistent: bool
    audit: bytes
    beta_min: float
    beta_cap: float

    def verdict(self) -> Optional[bool]:
        """Translate the integer decision into the spec tri-state verdict."""

        if self.decision > 0:
            return True
        if self.decision < 0:
            return False
        return None

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly representation of the trust gate result."""

        return {
            "decision": self.decision,
            "m": self.m,
            "llr": float(self.llr),
            "gamma": float(self.gamma),
            "consistent": bool(self.consistent),
            "audit": self.audit,
            "beta_min": float(self.beta_min),
            "beta_cap": float(self.beta_cap),
        }


class TrustGateState:
    """Deterministic verifier implementing the trust gate (spec §6.4)."""

    _AUDIT_SCHEMA = struct.Struct("<ii2dQQ")

    def __init__(self, config: TrustGateConfig) -> None:
        self.config = config
        self._projections = self._build_projections(config)
        norm_list: List[float] = []
        for row in self._projections:
            norm_sq = 0.0
            for value in row:
                norm_sq += float(value) * float(value)
            norm = math.sqrt(norm_sq)
            if norm == 0.0:
                norm = 1.0
            norm_list.append(norm)
        self._row_norms = np.asarray(norm_list, dtype=float)

    @staticmethod
    def _build_projections(config: TrustGateConfig) -> np.ndarray:
        matrix = np.zeros((config.salts, config.dimension), dtype=float)
        for salt in range(config.salts):
            seed = _mix64(salt + 1, 0xC1F651A1AD4101B7)
            for idx in range(config.dimension):
                mixed = _mix64(seed, idx + 1)
                matrix[salt, idx] = ((mixed / _HASH_MASK) * 2.0) - 1.0
        return matrix

    def _default_vector(self, diagnostics: "SeraDiagnostics") -> np.ndarray:
        features = [
            _safe_tanh(diagnostics.attention_den_min, scale=10.0),
            _safe_tanh(diagnostics.attention_clip_rate, scale=1.0),
            _safe_tanh(diagnostics.lambda_star, scale=1.0),
            _safe_tanh(diagnostics.store_load_p99, scale=1.0),
            _safe_tanh(diagnostics.stash_occ_p99, scale=1.0),
            _safe_tanh(diagnostics.kick_len_p99, scale=10.0),
            _safe_tanh(diagnostics.capacity_lambda_hat, scale=10.0),
            _safe_tanh(diagnostics.capacity_load, scale=10.0),
            _safe_tanh(diagnostics.capacity_slack, scale=10.0),
            _safe_tanh(diagnostics.capacity_margin, scale=10.0),
            float(bool(diagnostics.capacity_frozen)),
            _safe_tanh(diagnostics.bridge_guard_rate, scale=1.0),
            _safe_tanh(diagnostics.tokenizer_probe_max, scale=32.0),
            _safe_tanh(diagnostics.tok_emitted % 1024, scale=1024.0),
            _safe_tanh(diagnostics.tok_bytes_in % 1024, scale=1024.0),
            1.0,
        ]
        if len(features) < self.config.dimension:
            features.extend([0.0] * (self.config.dimension - len(features)))
        return np.asarray(features[: self.config.dimension], dtype=float)

    def _coerce_vector(self, vector: Optional[Sequence[float]], diagnostics: "SeraDiagnostics") -> np.ndarray:
        if vector is None:
            return self._default_vector(diagnostics)
        arr_list = [float(v) for v in vector]
        arr = np.asarray(arr_list, dtype=float)
        if arr.size == self.config.dimension:
            return arr
        coerced = np.zeros(self.config.dimension, dtype=float)
        limit = min(arr.size, self.config.dimension)
        coerced[:limit] = arr[:limit]
        return coerced

    def _project(self, vector: np.ndarray) -> np.ndarray:
        projected = self._projections @ vector
        return projected / self._row_norms

    def _audit(self, decision: int, m: int, llr: float, gamma: float, digest: int, collisions: int) -> bytes:
        payload = self._AUDIT_SCHEMA.pack(
            decision,
            m,
            float(llr),
            float(gamma),
            digest & _HASH_MASK,
            collisions & _HASH_MASK,
        )
        return payload.ljust(64, b"\x00")

    def judge(
        self,
        diagnostics: "SeraDiagnostics",
        vector: Optional[Sequence[float]] = None,
    ) -> TrustGateDecision:
        trust_vec = self._coerce_vector(vector, diagnostics)
        projections = self._project(trust_vec)
        neg = False
        pos_count = 0
        digest = 0
        seen_hashes: Set[int] = set()
        consistent = True
        for salt, value in enumerate(projections):
            value = _clamp(float(value), -1.0, 1.0)
            if value <= self.config.neg_threshold:
                neg = True
                digest ^= _mix64(salt + 1, 0)
            elif value >= self.config.pos_threshold:
                pos_count += 1
                witness_hash = _mix64(salt + 1, int(round((value + 1.0) * 1_000_000)))
                if witness_hash in seen_hashes:
                    consistent = False
                seen_hashes.add(witness_hash)
                digest ^= witness_hash
        decision = 0
        beta_min = self.config.beta_min
        beta_cap = 0.0
        llr = self.config.logit_pi0
        gamma = self.config.gamma_threshold
        if neg:
            decision = -1
            beta_cap = 0.0
        else:
            if pos_count >= self.config.acceptance_k and consistent:
                llr = self.config.logit_pi0 + pos_count * self.config.log_bf_pos
                if llr >= gamma:
                    decision = 1
                else:
                    decision = 0
            else:
                decision = 0
            if decision > 0:
                beta_min = self.config.beta_boost
                beta_cap = self.config.beta_max
            elif decision == 0:
                beta_min = self.config.beta_min
                beta_cap = self.config.alpha_unknown * self.config.beta_max
            else:
                beta_min = self.config.beta_min
                beta_cap = 0.0
            if decision < 0:
                beta_cap = 0.0
        if decision < 0:
            beta_min = 0.0
        elif beta_cap > 0.0:
            beta_min = min(beta_min, beta_cap)
        audit = self._audit(decision, pos_count, llr, gamma, digest, len(seen_hashes))
        return TrustGateDecision(
            decision=decision,
            m=pos_count,
            llr=llr,
            gamma=gamma,
            consistent=consistent,
            audit=audit,
            beta_min=beta_min,
            beta_cap=beta_cap,
        )


# ---------------------------------------------------------------------------
# CCR overlap corrector (spec section 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CCRConfig:
    gamma: float = 0.25
    truncation_order: int = 2


@dataclass(frozen=True)
class CCRSmallBlocks:
    """Small precomputed blocks for the CCR contraction (spec §7)."""

    dim: int
    B_blocks: Tuple[np.ndarray, ...]
    W_blocks: Tuple[np.ndarray, ...]
    pi: np.ndarray
    h: np.ndarray
    h_series: np.ndarray


@dataclass(frozen=True)
class CCRResult:
    """Result of a CCR correction step including residual construction."""

    residual: np.ndarray
    correction: np.ndarray
    corrected_locals: np.ndarray
    target: float
    y: float


@dataclass
class CCRState:
    config: CCRConfig
    _certificate: "CCRProof" = field(init=False)
    _blocks: CCRSmallBlocks = field(init=False)

    def __post_init__(self) -> None:
        if not (0 <= self.config.gamma < 1):
            raise ValueError("gamma must be in [0,1)")
        tail = self._tail_bound(self.config.gamma, self.config.truncation_order)
        self._certificate = CCRProof(
            gamma=self.config.gamma,
            truncation_order=self.config.truncation_order,
            tail_bound=tail,
        )
        self._blocks = self._precompute_small_blocks(self.config.truncation_order)

    def correct(self, locals_: np.ndarray) -> CCRResult:
        locals_vec = np.asarray(locals_, dtype=float)
        if locals_vec.ndim != 1:
            raise ValueError("Locals must be a vector")
        if locals_vec.shape[0] != self._blocks.dim:
            raise ValueError("Unexpected locals dimensionality")
        if self.config.gamma >= 1:
            raise BudgetError("CCR contraction gamma must be < 1")
        target = float(self._blocks.pi @ locals_vec)
        residual = locals_vec - target
        correction = -self._blocks.h_series @ residual
        corrected = locals_vec + correction
        y_val = float(self._blocks.pi @ corrected)
        return CCRResult(
            residual=residual,
            correction=correction,
            corrected_locals=corrected,
            target=target,
            y=y_val,
        )

    def _precompute_small_blocks(self, order: int) -> CCRSmallBlocks:
        if order != 2:
            raise NotImplementedError("Only truncation order m=2 is supported")
        dim = 2
        h = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        identity = np.eye(dim, dtype=float)
        contraction = identity - self.config.gamma * h
        B_blocks: List[np.ndarray] = [identity]
        current = identity
        for _ in range(order):
            current = current @ contraction
            B_blocks.append(current)
        gamma_h = -self.config.gamma * h
        W_blocks: List[np.ndarray] = [identity]
        current_w = identity
        for _ in range(order):
            current_w = current_w @ gamma_h
            W_blocks.append(current_w)
        series = np.zeros((dim, dim), dtype=float)
        for block in W_blocks[: order + 1]:
            series += block
        h_series = h @ series
        pi = np.ones(dim, dtype=float) * (1.0 / dim)
        return CCRSmallBlocks(
            dim=dim,
            B_blocks=tuple(B_blocks),
            W_blocks=tuple(W_blocks),
            pi=pi,
            h=h,
            h_series=h_series,
        )

    @staticmethod
    def _tail_bound(gamma: float, order: int) -> float:
        return gamma ** (order + 1) / (1 - gamma) if gamma < 1 else float("inf")

    @property
    def certificate(self) -> "CCRProof":
        return self._certificate

    @property
    def blocks(self) -> CCRSmallBlocks:
        return self._blocks


@dataclass(frozen=True)
class CCRProof:
    gamma: float
    truncation_order: int
    tail_bound: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "gamma": float(self.gamma),
            "truncation_order": int(self.truncation_order),
            "tail_bound": float(self.tail_bound),
        }


@dataclass(frozen=True)
class _CuckooEntry:
    key: int
    value: float


class BoundedCuckooMap:
    """Deterministic bounded cuckoo hash table for sparse weights (spec §4.5)."""

    __slots__ = (
        "capacity",
        "choices",
        "bucket_size",
        "stash_capacity",
        "max_kicks",
        "ring_capacity",
        "_bucket_count",
        "_table",
        "_stash",
        "_ring",
        "_size",
        "_max_kick_length",
    )

    def __init__(
        self,
        *,
        capacity: int,
        choices: int,
        bucket_size: int,
        stash_capacity: int,
        max_kicks: int,
        ring_capacity: int,
    ) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self.choices = max(1, choices)
        self.bucket_size = max(1, bucket_size)
        self.stash_capacity = max(0, stash_capacity)
        self.max_kicks = max(1, max_kicks)
        self.ring_capacity = max(0, ring_capacity)
        bucket_count = max(1, math.ceil(self.capacity / self.bucket_size))
        self._bucket_count = bucket_count
        self._table: List[List[Optional[_CuckooEntry]]] = [
            [None for _ in range(self.bucket_size)] for _ in range(bucket_count)
        ]
        self._stash: List[_CuckooEntry] = []
        self._ring: List[_CuckooEntry] = []
        self._size = 0
        self._max_kick_length = 0

    def _hash_bucket(self, key: int, choice: int) -> int:
        return _mix64(key, choice + 1) % self._bucket_count

    def _candidate_buckets(self, key: int) -> List[int]:
        buckets: List[int] = []
        for choice in range(self.choices):
            bucket = self._hash_bucket(key, choice)
            if bucket not in buckets:
                buckets.append(bucket)
        return buckets

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._size

    def get(self, key: int, default: float = 0.0) -> float:
        key = int(key)
        for bucket in self._candidate_buckets(key):
            for entry in self._table[bucket]:
                if entry is not None and entry.key == key:
                    return entry.value
        for entry in self._stash:
            if entry.key == key:
                return entry.value
        for entry in self._ring:
            if entry.key == key:
                return entry.value
        return float(default)

    def insert(self, key: int, value: float) -> bool:
        key = int(key)
        value = float(value)
        entry = _CuckooEntry(key=key, value=value)
        # Update existing entries first.
        for bucket in self._candidate_buckets(key):
            for slot_idx, slot in enumerate(self._table[bucket]):
                if slot is not None and slot.key == key:
                    self._table[bucket][slot_idx] = entry
                    return False
        for idx, slot in enumerate(self._stash):
            if slot.key == key:
                self._stash[idx] = entry
                return False
        for idx, slot in enumerate(self._ring):
            if slot.key == key:
                self._ring[idx] = entry
                return False
        self._place(entry)
        return True

    def _place(self, entry: _CuckooEntry) -> None:
        # Try direct placements first.
        for bucket in self._candidate_buckets(entry.key):
            slots = self._table[bucket]
            for idx in range(self.bucket_size):
                if slots[idx] is None:
                    slots[idx] = entry
                    self._size += 1
                    self._max_kick_length = max(self._max_kick_length, 0)
                    return
        # Deterministic kick-out procedure.
        current = entry
        for kick in range(self.max_kicks):
            choice = kick % self.choices
            bucket_idx = self._hash_bucket(current.key, choice)
            slot_idx = (bucket_idx + kick) % self.bucket_size
            slots = self._table[bucket_idx]
            slots[slot_idx], current = current, slots[slot_idx]
            if current is None:
                self._size += 1
                self._max_kick_length = max(self._max_kick_length, kick + 1)
                return
        # Stash fallback.
        if current is None:
            return
        if len(self._stash) < self.stash_capacity:
            self._stash.append(current)
            self._size += 1
            self._max_kick_length = max(self._max_kick_length, self.max_kicks)
            return
        if len(self._ring) < self.ring_capacity:
            self._ring.append(current)
            self._size += 1
            self._max_kick_length = max(self._max_kick_length, self.max_kicks)
            return
        raise BudgetError("Bounded cuckoo map overflow")

    def items(self) -> Iterable[Tuple[int, float]]:
        for bucket in self._table:
            for entry in bucket:
                if entry is not None:
                    yield entry.key, entry.value
        for entry in self._stash:
            yield entry.key, entry.value
        for entry in self._ring:
            yield entry.key, entry.value

    @property
    def load_factor(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return min(1.0, self._size / float(self.capacity))

    @property
    def stash_load(self) -> int:
        return len(self._stash)

    @property
    def ring_load(self) -> int:
        return len(self._ring)

    @property
    def max_kick_length(self) -> int:
        return self._max_kick_length

    def snapshot(self) -> Dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "choices": int(self.choices),
            "bucket_size": int(self.bucket_size),
            "stash_capacity": int(self.stash_capacity),
            "max_kicks": int(self.max_kicks),
            "ring_capacity": int(self.ring_capacity),
            "max_kick_length": int(self._max_kick_length),
            "table": [
                [
                    None
                    if entry is None
                    else [int(entry.key), float(entry.value)]
                    for entry in bucket
                ]
                for bucket in self._table
            ],
            "stash": [[int(entry.key), float(entry.value)] for entry in self._stash],
            "ring": [[int(entry.key), float(entry.value)] for entry in self._ring],
        }

    def restore(self, blob: Dict[str, object]) -> None:
        table_blob = blob.get("table")
        if table_blob is None:
            # Backwards compatibility with dict snapshots.
            for key, value in blob.items():
                target_key: Optional[int] = None
                if isinstance(key, int):
                    target_key = key
                elif isinstance(key, str) and key.isdigit():
                    target_key = int(key)
                if target_key is not None:
                    self.insert(int(target_key), float(value))
            return
        self._table = [
            [
                None
                if entry is None
                else _CuckooEntry(key=int(entry[0]), value=float(entry[1]))
                for entry in bucket
            ]
            for bucket in table_blob
        ]
        self._bucket_count = len(self._table)
        self.capacity = int(blob.get("capacity", self.capacity))
        self.choices = int(blob.get("choices", self.choices))
        self.bucket_size = int(blob.get("bucket_size", self.bucket_size))
        self.stash_capacity = int(blob.get("stash_capacity", self.stash_capacity))
        self.max_kicks = int(blob.get("max_kicks", self.max_kicks))
        self.ring_capacity = int(blob.get("ring_capacity", self.ring_capacity))
        self._stash = [
            _CuckooEntry(key=int(entry[0]), value=float(entry[1]))
            for entry in blob.get("stash", [])
        ]
        self._ring = [
            _CuckooEntry(key=int(entry[0]), value=float(entry[1]))
            for entry in blob.get("ring", [])
        ]
        self._size = sum(1 for _ in self.items())
        self._max_kick_length = int(blob.get("max_kick_length", self._max_kick_length))

    def manifest(self) -> Dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "choices": int(self.choices),
            "bucket_size": int(self.bucket_size),
            "stash_capacity": int(self.stash_capacity),
            "ring_capacity": int(self.ring_capacity),
            "load": float(self.load_factor),
            "max_kick_length": int(self._max_kick_length),
            "stash": [
                {"key": int(entry.key), "value": float(entry.value)}
                for entry in self._stash
            ],
            "ring": [
                {"key": int(entry.key), "value": float(entry.value)}
                for entry in self._ring
            ],
            "table": [
                [
                    None
                    if entry is None
                    else {"key": int(entry.key), "value": float(entry.value)}
                    for entry in bucket
                ]
                for bucket in self._table
            ],
        }


class LinearCapacityMonitor:
    """Tracks τ-threshold slack and insert freeze state (spec §4.4)."""

    __slots__ = (
        "capacity",
        "tau_low",
        "tau_high",
        "margin",
        "lambda_hat",
        "events",
        "new_keys",
        "load",
        "slack",
        "frozen",
    )

    def __init__(self, config: SparseLinearConfig) -> None:
        self.capacity = config.capacity
        self.tau_low = config.tau_low
        self.tau_high = config.tau_high
        self.margin = config.margin
        self.lambda_hat = 0.0
        self.events = 0
        self.new_keys = 0
        self.load = 0.0
        self.slack = float(self.capacity) * (self.tau_high - self.tau_low)
        self.frozen = False

    def should_freeze(self, projected_load: float) -> bool:
        if projected_load >= self.tau_high:
            return True
        slack = max(0.0, (self.tau_high - projected_load) * self.capacity)
        return slack < self.margin

    def observe(self, new_key: bool, load: float) -> None:
        self.events += 1
        if new_key:
            self.new_keys += 1
        self.lambda_hat = self.new_keys / float(self.events) if self.events else 0.0
        self.load = load
        self.slack = max(0.0, (self.tau_high - load) * self.capacity)
        if load <= self.tau_low:
            self.frozen = False
        elif self.should_freeze(load):
            self.frozen = True

    def freeze(self, load: float) -> None:
        self.load = load
        self.slack = max(0.0, (self.tau_high - load) * self.capacity)
        self.frozen = True

    def recompute(self, load: float) -> None:
        self.load = load
        self.slack = max(0.0, (self.tau_high - load) * self.capacity)
        if load <= self.tau_low:
            self.frozen = False

    def snapshot(self) -> Dict[str, object]:
        return {
            "lambda_hat": float(self.lambda_hat),
            "events": int(self.events),
            "new_keys": int(self.new_keys),
            "load": float(self.load),
            "slack": float(self.slack),
            "frozen": bool(self.frozen),
            "margin": float(self.margin),
        }

    @classmethod
    def restore(
        cls, config: SparseLinearConfig, blob: Dict[str, object]
    ) -> "LinearCapacityMonitor":
        monitor = cls(config)
        monitor.lambda_hat = float(blob.get("lambda_hat", 0.0))
        monitor.events = int(blob.get("events", 0))
        monitor.new_keys = int(blob.get("new_keys", 0))
        monitor.load = float(blob.get("load", 0.0))
        monitor.slack = float(blob.get("slack", monitor.slack))
        monitor.frozen = bool(blob.get("frozen", False))
        monitor.margin = float(blob.get("margin", monitor.margin))
        return monitor

    def manifest(self) -> Dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "tau_low": float(self.tau_low),
            "tau_high": float(self.tau_high),
            "margin": float(self.margin),
            "lambda_hat": float(self.lambda_hat),
            "events": int(self.events),
            "new_keys": int(self.new_keys),
            "load": float(self.load),
            "slack": float(self.slack),
            "frozen": bool(self.frozen),
        }

# ---------------------------------------------------------------------------
# External bridge (spec section 10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeConfig:
    """Static configuration for the external bridge (spec §10.1–§10.7)."""

    hub_window: int = 8
    candidate_bound: int = 4
    beta_min: float = 0.1
    beta_max: float = 0.9
    projection: Sequence[Sequence[float]] = ((1.0,),)
    load_max: float = 1.0
    stash_max: float = 1.0
    kick_max: float = 8.0
    margin_policy: float = 0.0
    eps_row_policy: float = 0.0
    route_proof_schema_digest: str = ""
    projection_digest: str = field(init=False)
    value_dim: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hub_window <= 0:
            raise ValueError("hub_window must be positive")
        if self.candidate_bound <= 0:
            raise ValueError("candidate_bound must be positive")
        if not 0.0 <= self.beta_min <= self.beta_max <= 1.0:
            raise ValueError("beta range must satisfy 0 <= beta_min <= beta_max <= 1")
        if self.load_max <= 0.0:
            raise ValueError("load_max must be positive")
        if self.stash_max <= 0.0:
            raise ValueError("stash_max must be positive")
        if self.kick_max <= 0.0:
            raise ValueError("kick_max must be positive")
        if self.margin_policy < 0.0:
            raise ValueError("margin_policy must be non-negative")
        if self.eps_row_policy < 0.0:
            raise ValueError("eps_row_policy must be non-negative")

        rows = []
        width = None
        for row in self.projection:
            row_tuple = tuple(float(v) for v in row)
            if width is None:
                width = len(row_tuple)
            elif len(row_tuple) != width:
                raise ValueError("projection rows must have a consistent width")
            rows.append(row_tuple)
        if not rows or width is None or width == 0:
            raise ValueError("projection must be a non-empty 2D matrix")
        object.__setattr__(self, "projection", tuple(rows))
        object.__setattr__(self, "value_dim", width)

        digest = hashlib.sha256()
        for row in rows:
            for value in row:
                digest.update(struct.pack("<d", float(value)))
        object.__setattr__(self, "projection_digest", digest.hexdigest())

        if not self.route_proof_schema_digest:
            schema_digest = hashlib.sha256(BridgeProof._SCHEMA.format.encode("ascii")).hexdigest()
            object.__setattr__(self, "route_proof_schema_digest", schema_digest)


@dataclass
class BridgeGuardRecord:
    """Guard metadata for bridge promotions (spec §10.3–§10.5)."""

    margin: float
    eps_row: float
    leg_bound_in: float
    leg_bound_out: float
    competitor_bounds: Tuple[float, ...] = field(default_factory=tuple)

    def check(self) -> bool:
        """Evaluate the guard inequality from spec §10.5."""

        half_margin = 0.5 * float(self.margin) + float(self.eps_row)
        if float(self.leg_bound_in) + float(self.leg_bound_out) >= half_margin:
            return False
        return all(float(bound) < half_margin for bound in self.competitor_bounds)

    def as_dict(self) -> Dict[str, object]:
        return {
            "margin": float(self.margin),
            "eps_row": float(self.eps_row),
            "leg_bound_in": float(self.leg_bound_in),
            "leg_bound_out": float(self.leg_bound_out),
            "competitor_bounds": [float(b) for b in self.competitor_bounds],
        }


@dataclass
class BridgeEntry:
    value: np.ndarray
    guard: BridgeGuardRecord
    generation: int

    def manifest(self) -> Dict[str, object]:
        return {
            "guard": self.guard.as_dict(),
            "generation": self.generation,
            "value_shape": list(self.value.shape),
        }


@dataclass(frozen=True)
class BridgeProof:
    """64-byte proof record emitted by :meth:`BridgeState.read` (spec §10.3)."""

    hub: int
    margin: float
    leg_bound_in: float
    leg_bound_out: float
    eps_row: float
    best_cost: float
    guard_pass: bool
    ctx: int
    token: int

    _SCHEMA = struct.Struct("<QddddddQ")

    def pack(self) -> bytes:
        guard_flag = 1.0 if self.guard_pass else 0.0
        packed_ids = ((self.ctx & 0xFFFFFFFF) << 32) | (self.token & 0xFFFFFFFF)
        return self._SCHEMA.pack(
            self.hub & 0xFFFFFFFFFFFFFFFF,
            float(self.margin),
            float(self.leg_bound_in),
            float(self.leg_bound_out),
            float(self.eps_row),
            float(self.best_cost),
            guard_flag,
            packed_ids,
        )


@dataclass
class BridgeState:
    config: BridgeConfig
    _hub_bits: Dict[int, List[int]] = field(default_factory=dict)
    _store: Dict[Tuple[int, int], BridgeEntry] = field(default_factory=dict)
    _qdin: Dict[Tuple[int, int], float] = field(default_factory=dict)
    _qdout: Dict[Tuple[int, int], float] = field(default_factory=dict)
    _generation: int = 0
    _projection: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        matrix_data = [list(row) for row in self.config.projection]
        matrix = np.asarray(matrix_data, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Bridge projection must be two-dimensional")
        if matrix.shape[1] != self.config.value_dim:
            raise ValueError("Projection width must match BridgeConfig value_dim")
        self._projection = matrix

    def promote(
        self,
        ctx: int,
        token: int,
        value: np.ndarray,
        *,
        guard_margin: float = 0.0,
        guard_eps_row: float = 0.0,
        leg_bound_in: float = 0.0,
        leg_bound_out: float = 0.0,
        competitor_bounds: Optional[Sequence[float]] = None,
    ) -> None:
        """Promote a dictionary value for ``(ctx, token)`` (spec §10.3)."""

        guard = BridgeGuardRecord(
            guard_margin,
            guard_eps_row,
            leg_bound_in,
            leg_bound_out,
            tuple(float(b) for b in (competitor_bounds or ())),
        )
        if guard.margin < self.config.margin_policy:
            raise ValueError("Guard margin is below configured policy")
        if guard.eps_row < self.config.eps_row_policy:
            raise ValueError("Guard eps_row is below configured policy")
        value_vec = self._coerce_value(value)
        self._store[(ctx, token)] = BridgeEntry(value_vec, guard, self._generation)
        self._hub_bits.setdefault(ctx, [0] * self.config.hub_window)
        self._hub_bits.setdefault(token, [0] * self.config.hub_window)

    # ------------------------------------------------------------------
    # Hub and quantized distance management (spec §10.3–§10.4)
    # ------------------------------------------------------------------

    def set_hub(self, node: int, hub_index: int, enabled: bool = True) -> None:
        if hub_index < 0:
            raise ValueError("hub_index must be non-negative")
        word = hub_index // 64
        bit = hub_index % 64
        if word >= self.config.hub_window:
            raise ValueError("hub_index exceeds configured hub window")
        bits = self._hub_bits.setdefault(node, [0] * self.config.hub_window)
        if enabled:
            bits[word] |= 1 << bit
        else:
            bits[word] &= ~(1 << bit)

    def set_hub_bits(self, node: int, words: Sequence[int]) -> None:
        if len(words) != self.config.hub_window:
            raise ValueError("words length must match hub_window")
        self._hub_bits[node] = [int(word) & 0xFFFFFFFFFFFFFFFF for word in words]

    def set_qdin(self, hub: int, token: int, value: float) -> None:
        self._qdin[(hub, token)] = float(value)

    def set_qdout(self, ctx: int, hub: int, value: float) -> None:
        self._qdout[(ctx, hub)] = float(value)

    def _hub_mask(self, node: int) -> List[int]:
        return self._hub_bits.get(node, [0] * self.config.hub_window)

    def _enumerate_hubs(self, ctx: int, token: Optional[int]) -> List[int]:
        candidates: List[int] = []
        if token is None:
            return candidates
        bits_ctx = self._hub_mask(ctx)
        bits_token = self._hub_mask(token)
        for word_index in range(self.config.hub_window):
            mask = bits_ctx[word_index] & bits_token[word_index]
            base = word_index * 64
            while mask and len(candidates) < self.config.candidate_bound:
                lowest = mask & -mask
                # Map lowest set bit to its index using branchless bit_length.
                bit_index = (lowest.bit_length() - 1) & 0x3F
                candidates.append(base + bit_index)
                mask &= mask - 1
        return candidates

    def _best_route(
        self, ctx: int, token: Optional[int]
    ) -> Tuple[Optional[int], float]:
        if token is None:
            return None, float("inf")
        best_cost = float("inf")
        best_hub: Optional[int] = None
        for hub in self._enumerate_hubs(ctx, token):
            din = self._qdin.get((hub, token), float("inf"))
            dout = self._qdout.get((ctx, hub), float("inf"))
            cost = din + dout
            if cost < best_cost:
                best_cost = cost
                best_hub = hub
        return best_hub, best_cost

    def read(self, ctx: int, token: Optional[int]) -> Tuple[np.ndarray, bool, bytes]:
        if token is None:
            proof = BridgeProof(
                hub=0,
                margin=0.0,
                leg_bound_in=0.0,
                leg_bound_out=0.0,
                eps_row=0.0,
                best_cost=float("inf"),
                guard_pass=False,
                ctx=ctx,
                token=0,
            ).pack()
            return self._empty_value(), False, proof

        entry = self._store.get((ctx, token))
        best_hub, best_cost = self._best_route(ctx, token)
        guard_record = entry.guard if entry is not None else None
        guard_ok = (
            bool(guard_record.check()) and best_hub is not None
            if guard_record is not None
            else False
        )

        if entry is not None and guard_ok:
            value = entry.value.copy()
        else:
            fallback = best_cost if math.isfinite(best_cost) else 0.0
            value = np.full(self.config.value_dim, float(fallback), dtype=float)

        proof = BridgeProof(
            hub=best_hub or 0,
            margin=guard_record.margin if guard_record is not None else 0.0,
            leg_bound_in=guard_record.leg_bound_in if guard_record is not None else 0.0,
            leg_bound_out=guard_record.leg_bound_out if guard_record is not None else 0.0,
            eps_row=guard_record.eps_row if guard_record is not None else 0.0,
            best_cost=best_cost,
            guard_pass=guard_ok,
            ctx=ctx,
            token=token,
        ).pack()

        return value, guard_ok, proof

    def gate(
        self,
        base: float,
        bridge_val: np.ndarray,
        guard: bool,
        witness_quality: float = 1.0,
        *,
        beta_min_override: Optional[float] = None,
        beta_cap_override: Optional[float] = None,
    ) -> float:
        guard_weight = float(bool(guard))
        s_q = min(max(float(witness_quality), 0.0), 1.0)
        beta_cap = self.config.beta_max if beta_cap_override is None else float(beta_cap_override)
        beta_min = self.config.beta_min if beta_min_override is None else float(beta_min_override)
        beta_cap = _clamp(beta_cap, 0.0, 1.0)
        beta_min = _clamp(beta_min, 0.0, beta_cap)
        beta_span = max(beta_cap - beta_min, 0.0)
        beta = guard_weight * (beta_min + beta_span * s_q)
        alpha = 1.0 - beta
        target = float(np.mean(self._project_value(bridge_val)))
        return alpha * float(base) + beta * target

    def _project_value(self, bridge_val: Sequence[float]) -> np.ndarray:
        vector = self._coerce_value(bridge_val)
        return self._projection @ vector

    def _coerce_value(self, value: Sequence[float]) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        raw = arr.tolist()

        flat: List[float] = []

        def _flatten(item: object) -> None:
            if isinstance(item, list):
                for sub in item:
                    _flatten(sub)
            else:
                flat.append(float(item))

        if isinstance(raw, list):
            _flatten(raw)
        else:
            flat.append(float(raw))

        if not flat:
            flat.append(0.0)

        if len(flat) < self.config.value_dim:
            flat.extend([0.0] * (self.config.value_dim - len(flat)))
        elif len(flat) > self.config.value_dim:
            flat = flat[: self.config.value_dim]

        return np.asarray(flat, dtype=float)

    def _empty_value(self) -> np.ndarray:
        return np.zeros(self.config.value_dim, dtype=float)

    def advance_generation(self, generation: int) -> None:
        self._generation = generation

    def guard_record(self, ctx: int, token: Optional[int]) -> Optional[BridgeGuardRecord]:
        if token is None:
            return None
        entry = self._store.get((ctx, token))
        if entry is None:
            return None
        return entry.guard

    def manifest(self) -> Dict[str, object]:
        config_blob = {
            "hub_window": int(self.config.hub_window),
            "candidate_bound": int(self.config.candidate_bound),
            "beta_min": float(self.config.beta_min),
            "beta_max": float(self.config.beta_max),
            "projection": {
                "shape": list(self._projection.shape),
                "digest": self.config.projection_digest,
            },
            "guard": {
                "margin_policy": float(self.config.margin_policy),
                "eps_row_policy": float(self.config.eps_row_policy),
            },
            "store_limits": {
                "load_max": float(self.config.load_max),
                "stash_max": float(self.config.stash_max),
                "kick_max": float(self.config.kick_max),
            },
            "route_proof_schema_digest": self.config.route_proof_schema_digest,
        }
        entries = {
            f"{ctx}:{token}": entry.manifest()
            for (ctx, token), entry in sorted(self._store.items())
        }
        hubs = {
            str(node): [int(word) for word in words]
            for node, words in sorted(self._hub_bits.items())
        }
        qdin = {
            f"{hub}:{token}": float(value)
            for (hub, token), value in sorted(self._qdin.items())
        }
        qdout = {
            f"{ctx}:{hub}": float(value)
            for (ctx, hub), value in sorted(self._qdout.items())
        }
        return {
            "generation": self._generation,
            "config": config_blob,
            "entries": entries,
            "hub_bits": hubs,
            "qDin": qdin,
            "qDout": qdout,
        }


# ---------------------------------------------------------------------------
# Counterfactual replay module (Appendix A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CFRConfig:
    mode: str = "OFF"
    gate_gain: float = 1.0
    beta_cfr_max: float = 0.2
    projector: Sequence[Sequence[float]] = ((1.0, 0.0, 0.0, 0.0),)
    mem_excitation: Sequence[Tuple[int, float]] = ((0, 0.0),)
    drift_epsilon: float = 1e-3
    schedule_period: int = 16
    policy_weights: Sequence[float] = (1.0, 0.0, 0.0, 0.0)
    policy_alpha: float = 0.5
    seed_attention: int = 0x1234_5678_9ABC_DEF0
    seed_memory: int = 0x0FED_CBA9_8765_4321
    seed_schedule: int = 0xCAFEBABE_DEADBEEF

    def __post_init__(self) -> None:
        mode = self.mode.upper().replace("_", "-")
        if mode not in {"OFF", "CFR-REPLAY", "CFR-MIX"}:
            if mode == "REPLAY":
                mode = "CFR-REPLAY"
            elif mode == "MIX":
                mode = "CFR-MIX"
            else:
                raise ValueError("mode must be OFF, CFR-REPLAY, or CFR-MIX")
        object.__setattr__(self, "mode", mode)

        if not 0.0 <= self.gate_gain <= 1.0:
            raise ValueError("gate_gain must lie in [0, 1]")
        if not 0.0 <= self.beta_cfr_max <= 1.0:
            raise ValueError("beta_cfr_max must lie in [0, 1]")
        if self.drift_epsilon < 0.0:
            raise ValueError("drift_epsilon must be non-negative")
        if self.schedule_period <= 0:
            raise ValueError("schedule_period must be positive")
        if not 0.0 <= self.policy_alpha <= 1.0:
            raise ValueError("policy_alpha must lie in [0, 1]")

        rows: List[Tuple[float, ...]] = []
        width = None
        for row in self.projector:
            row_tuple = tuple(float(v) for v in row)
            if width is None:
                width = len(row_tuple)
            elif len(row_tuple) != width:
                raise ValueError("projector rows must have a consistent width")
            rows.append(row_tuple)
        if not rows or width is None or width == 0:
            raise ValueError("projector must be a non-empty 2D matrix")
        object.__setattr__(self, "projector", tuple(rows))

        excitations = []
        for coord, weight in self.mem_excitation:
            excitations.append((int(coord), float(weight)))
        object.__setattr__(self, "mem_excitation", tuple(excitations))

        policy = tuple(float(v) for v in self.policy_weights)
        if not policy:
            policy = (1.0,)
        object.__setattr__(self, "policy_weights", policy)

        def _mask_seed(seed: int) -> int:
            return int(seed) & _HASH_MASK

        object.__setattr__(self, "seed_attention", _mask_seed(self.seed_attention))
        object.__setattr__(self, "seed_memory", _mask_seed(self.seed_memory))
        object.__setattr__(self, "seed_schedule", _mask_seed(self.seed_schedule))


@dataclass(frozen=True)
class CFRComputation:
    mode: str
    cf_vector: np.ndarray
    y_cfr: float
    beta: float
    y_mix: float
    guard: bool
    s_witness: float
    health_ok: bool


@dataclass
class CFRState:
    config: CFRConfig
    _projector: np.ndarray = field(init=False, repr=False)
    _mode: str = field(init=False, repr=False)
    _enabled: bool = field(init=False, repr=False)
    _gate_gain: float = field(init=False, repr=False)
    _mem_excitation: Tuple[Tuple[int, float], ...] = field(init=False, repr=False)
    _policy: Tuple[float, ...] = field(init=False, repr=False)
    _policy_alpha: float = field(init=False, repr=False)
    _seed_attn: int = field(init=False, repr=False)
    _seed_mem: int = field(init=False, repr=False)
    _seed_sched: int = field(init=False, repr=False)
    _schedule_period: int = field(init=False, repr=False)
    _generation: int = field(init=False, default=0)
    _event_counter: int = 0
    _last_beta: float = 0.0
    _last_y_cfr: float = 0.0
    _last_s_wit: float = 0.0

    def __post_init__(self) -> None:
        rows = [list(row) for row in self.config.projector]
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("projector must be 2-dimensional")
        self._projector = matrix
        self._mode = self.config.mode
        self._enabled = self._mode != "OFF"
        self._gate_gain = float(self.config.gate_gain)
        self._mem_excitation = self.config.mem_excitation
        self._policy = self.config.policy_weights
        self._policy_alpha = float(self.config.policy_alpha)
        self._schedule_period = int(self.config.schedule_period)
        self._reseed(0)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def allows_linear_update(self) -> bool:
        return self._mode != "CFR-REPLAY"

    def _reseed(self, generation: int) -> None:
        gen = int(generation)
        mask = gen & _HASH_MASK
        self._generation = gen
        self._seed_attn = _mix64(self.config.seed_attention, mask)
        self._seed_mem = _mix64(self.config.seed_memory, mask)
        self._seed_sched = _mix64(self.config.seed_schedule, mask)
        self._event_counter = 0

    def _drift_vector(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.zeros(0, dtype=float)
        values = np.zeros(length, dtype=float)
        base = (self._seed_attn + self._event_counter + 1) & _HASH_MASK
        for idx in range(length):
            hashed = _mix64(base + idx, self._seed_sched)
            frac = (hashed & _HASH_MASK) / float(1 << 64)
            values[idx] = (frac * 2.0 - 1.0) * self.config.drift_epsilon
        return values

    def _scheduled_excitation(self) -> float:
        if not self._mem_excitation:
            return 0.0
        acc = 0.0
        for coord, weight in self._mem_excitation:
            hashed = _mix64(self._seed_mem + coord, self._event_counter + self._seed_sched)
            frac = (hashed & _HASH_MASK) / float(1 << 64)
            acc += weight * ((frac * 2.0) - 1.0) * self.config.drift_epsilon
        if self._schedule_period > 1 and (self._event_counter % self._schedule_period) != 0:
            acc *= 0.5
        return acc

    def _counterfactual_vector(
        self,
        phi_q: Optional[np.ndarray],
        y_lin: float,
        mem_value: float,
        bridge_value: np.ndarray,
        y_fus: float,
    ) -> np.ndarray:
        width = self._projector.shape[1]
        vector = np.zeros(width, dtype=float)
        phi_mean = 0.0
        phi_norm = 0.0
        if phi_q is not None and phi_q.size:
            drift = self._drift_vector(phi_q.size)
            phi = phi_q[: drift.size] + drift
            phi_mean = float(np.mean(phi))
            phi_norm = float(np.linalg.norm(phi))
        else:
            drift = self._drift_vector(max(width, 1))
            if drift.size:
                phi_mean = float(np.mean(drift[: width]))
                phi_norm = float(np.linalg.norm(drift[: width]))
        anchor = float(np.mean(bridge_value)) if bridge_value.size else 0.0
        mem_adjust = mem_value + self._scheduled_excitation()
        signals = [phi_mean, phi_norm, float(y_lin), float(mem_adjust), anchor, float(y_fus)]
        for idx in range(width):
            vector[idx] = signals[idx] if idx < len(signals) else signals[-1]
        return vector

    def run(
        self,
        *,
        y_fus: float,
        y_lin: float,
        mem_value: float,
        phi_q: Optional[np.ndarray],
        bridge_value: np.ndarray,
        guard_ok: bool,
        trust_beta_min: float,
        trust_beta_cap: float,
    ) -> CFRComputation:
        vector = self._counterfactual_vector(phi_q, y_lin, mem_value, bridge_value, y_fus)
        projected = self._projector @ vector
        if projected.ndim == 0:
            y_cfr = float(projected)
        else:
            y_cfr = float(np.mean(projected))

        s_wit = 1.0 if guard_ok else 0.0
        beta_min = _clamp(float(trust_beta_min), 0.0, float(trust_beta_cap))
        beta_cap = _clamp(float(trust_beta_cap), 0.0, 1.0)
        beta_cap = min(beta_cap, self.config.beta_cfr_max)
        beta_span = max(0.0, beta_cap - beta_min)

        health_ok = guard_ok
        beta = 0.0
        y_mix = float(y_fus)
        if self._mode == "CFR-MIX" and self._enabled and health_ok:
            beta_target = beta_min + beta_span * s_wit
            beta = _clamp(self._gate_gain * beta_target, 0.0, beta_cap)
            if beta > 0.0:
                y_mix = (1.0 - beta) * float(y_fus) + beta * y_cfr
        elif self._mode == "CFR-REPLAY":
            beta = 0.0

        self._last_beta = beta
        self._last_y_cfr = y_cfr
        self._last_s_wit = s_wit
        self._event_counter = (self._event_counter + 1) & _HASH_MASK

        return CFRComputation(
            mode=self._mode,
            cf_vector=vector,
            y_cfr=y_cfr,
            beta=beta,
            y_mix=y_mix,
            guard=guard_ok,
            s_witness=s_wit,
            health_ok=health_ok,
        )

    def manifest(self) -> Dict[str, object]:
        digest = hashlib.sha256()
        for row in self._projector:
            for value in row:
                digest.update(struct.pack("<d", float(value)))
        return {
            "mode": self._mode,
            "beta_cfr_max": float(self.config.beta_cfr_max),
            "gate_gain": float(self._gate_gain),
            "projector_shape": list(self._projector.shape),
            "projector_digest": digest.hexdigest(),
            "policy_weights": [float(v) for v in self._policy],
            "policy_alpha": float(self._policy_alpha),
            "event_counter": int(self._event_counter),
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "event_counter": int(self._event_counter),
            "last_beta": float(self._last_beta),
            "last_y_cfr": float(self._last_y_cfr),
            "last_s_wit": float(self._last_s_wit),
            "generation": int(self._generation),
        }

    @classmethod
    def restore(cls, config: CFRConfig, blob: Optional[Dict[str, object]]) -> "CFRState":
        state = cls(config)
        if blob:
            generation = int(blob.get("generation", 0))
            state._reseed(generation)
            state._event_counter = int(blob.get("event_counter", 0)) & _HASH_MASK
            state._last_beta = float(blob.get("last_beta", 0.0))
            state._last_y_cfr = float(blob.get("last_y_cfr", 0.0))
            state._last_s_wit = float(blob.get("last_s_wit", 0.0))
        return state

    def advance_generation(self, generation: int) -> None:
        self._reseed(generation)


# ---------------------------------------------------------------------------
# Tree search stub (spec section 11 – optional)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeSearchConfig:
    enabled: bool = False
    cpuct: float = 1.0
    max_actions: int = 4
    action_priors: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    action_rewards: Tuple[float, ...] = (0.2, 0.1, 0.05, -0.05)
    selection_depth: int = 4
    rollout_depth: int = 4
    backup_depth: int = 4
    discount: float = 0.9


@dataclass
class TreeNode:
    action_index: Optional[int]
    visits: int = 0
    value_sum: float = 0.0
    children: List["TreeNode"] = field(default_factory=list)

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / float(self.visits)


@dataclass
class TreeSearchState:
    config: TreeSearchConfig
    _simulations: int = 0
    _root: TreeNode = field(init=False)
    _priors: Tuple[float, ...] = field(init=False)
    _rewards: Tuple[float, ...] = field(init=False)
    _max_actions: int = field(init=False)
    _selection_depth: int = field(init=False)
    _rollout_depth: int = field(init=False)
    _backup_depth: int = field(init=False)
    _discount: float = field(init=False)

    def __post_init__(self) -> None:
        self._max_actions = max(1, min(self.config.max_actions, 4))
        self._selection_depth = max(1, min(self.config.selection_depth, 4))
        self._rollout_depth = max(1, min(self.config.rollout_depth, 4))
        self._backup_depth = max(1, min(self.config.backup_depth, 4))
        self._discount = max(0.0, min(self.config.discount, 1.0))
        priors = list(self.config.action_priors[: self._max_actions])
        if not priors:
            priors = [1.0]
        while len(priors) < self._max_actions:
            priors.append(1.0)
        total = sum(priors)
        if total <= 0.0:
            priors = [1.0 for _ in priors]
            total = float(len(priors))
        self._priors = tuple(p / total for p in priors[: self._max_actions])

        rewards = list(self.config.action_rewards[: self._max_actions])
        if not rewards:
            rewards = [0.0]
        while len(rewards) < self._max_actions:
            rewards.append(rewards[-1])
        self._rewards = tuple(
            max(-1.0, min(1.0, r)) for r in rewards[: self._max_actions]
        )

        self._root = TreeNode(action_index=None)

    def maybe_select(self) -> None:
        if not self.config.enabled:
            return
        self._simulate()
        self._simulations = min(self._simulations + 1, 2**31 - 1)

    # ------------------------------------------------------------------
    # Internal helpers implementing best-of-two selection and bounded MCTS
    # ------------------------------------------------------------------

    def _allowed_children(self, node: TreeNode) -> int:
        allowed = 1
        if node.visits >= 2:
            allowed += 1
        if node.visits >= 4:
            allowed += 1
        return min(allowed, self._max_actions)

    def _score(self, parent: TreeNode, child: TreeNode, index: int) -> float:
        prior = self._priors[index]
        q_val = child.q_value()
        inv = 1.0 / (1.0 + float(child.visits))
        explore = self.config.cpuct * prior * math.sqrt(float(parent.visits) + 1.0) * inv
        return q_val + explore

    def _best_of_two(self, parent: TreeNode, idx_a: int, idx_b: int) -> int:
        child_a = parent.children[idx_a]
        child_b = parent.children[idx_b]
        score_a = self._score(parent, child_a, idx_a)
        score_b = self._score(parent, child_b, idx_b)
        if score_b > score_a or (score_a == score_b and idx_b < idx_a):
            return idx_b
        return idx_a

    def _select(self) -> Tuple[List[Tuple[TreeNode, int, TreeNode]], TreeNode]:
        node = self._root
        path: List[Tuple[TreeNode, int, TreeNode]] = []
        depth = 0
        while depth < self._selection_depth:
            allowed = self._allowed_children(node)
            if len(node.children) < allowed:
                index = len(node.children)
                child = TreeNode(action_index=index)
                node.children.append(child)
                path.append((node, index, child))
                return path, child
            if not node.children:
                break
            allowed_indices = min(len(node.children), allowed)
            best_index = 0
            for candidate in range(1, allowed_indices):
                best_index = self._best_of_two(node, best_index, candidate)
            child = node.children[best_index]
            path.append((node, best_index, child))
            node = child
            depth += 1
        return path, node

    def _rollout(self, leaf: TreeNode) -> float:
        base_index = leaf.action_index or 0
        reward = 0.0
        discount = 1.0
        for step in range(self._rollout_depth):
            idx = (base_index + step) % len(self._rewards)
            reward += discount * self._rewards[idx]
            discount *= self._discount
        return max(-1.0, min(1.0, reward))

    def _backup(self, path: List[Tuple[TreeNode, int, TreeNode]], reward: float) -> None:
        if not path:
            self._root.visits = min(self._root.visits + 1, 2**31 - 1)
            self._root.value_sum = max(
                -float(self._root.visits),
                min(float(self._root.visits), self._root.value_sum + reward),
            )
            return
        steps = path[-min(len(path), self._backup_depth):]
        for parent, index, child in reversed(steps):
            child.visits = min(child.visits + 1, 2**31 - 1)
            child.value_sum = max(-float(child.visits), min(float(child.visits), child.value_sum + reward))
            parent.visits = min(parent.visits + 1, 2**31 - 1)
            parent.value_sum = max(
                -float(parent.visits),
                min(float(parent.visits), parent.value_sum + reward),
            )

    def _simulate(self) -> None:
        path, leaf = self._select()
        reward = self._rollout(leaf)
        self._backup(path, reward)


# ---------------------------------------------------------------------------
# Sera top-level API (spec section 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FPContract:
    precision: str = "float64"
    rounding: str = "nearest_even"
    fma: bool = True
    denormals: str = "preserved"
    reduction_order: str = "fixed"

    def as_dict(self) -> Dict[str, object]:
        return {
            "precision": self.precision,
            "rounding": self.rounding,
            "fma": self.fma,
            "denormals": self.denormals,
            "reduction_order": self.reduction_order,
        }


@dataclass(frozen=True)
class ManifestRecord:
    generation: int
    fp_contract: FPContract
    sections: Dict[str, object]

    def digest(self) -> Dict[str, str]:
        payload = json.dumps(
            {
                "generation": self.generation,
                "fp_contract": self.fp_contract.as_dict(),
                "sections": self.sections,
            },
            sort_keys=True,
        ).encode("utf-8")
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        sha = hashlib.sha256(payload).hexdigest()
        return {"crc32c": f"0x{crc:08x}", "sha256": sha}

    def as_dict(self) -> Dict[str, object]:
        return {
            "generation": self.generation,
            "fp_contract": self.fp_contract.as_dict(),
            "sections": self.sections,
            "digest": self.digest(),
        }


@dataclass(frozen=True)
class PublishedGeneration:
    """Immutable record for a published generation."""

    generation: int
    manifest: Dict[str, object]
    epoch: int


class GenerationPin(AbstractContextManager["GenerationPin"]):
    """Context manager representing a live generation pin."""

    def __init__(
        self,
        pointer: "GenerationPointer",
        generation: int,
        epoch: int,
        manifest: Dict[str, object],
    ) -> None:
        self._pointer = pointer
        self.generation = generation
        self.epoch = epoch
        self.manifest = manifest
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pointer._release(self.generation)

    def __enter__(self) -> "GenerationPin":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.release()


class GenerationPointer:
    """Manages single-pointer generation publication with pinning."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: Optional[PublishedGeneration] = None
        self._pins: Dict[int, int] = {}
        self._history: Dict[int, PublishedGeneration] = {}
        self._epoch_counter = 0

    def publish(self, record: ManifestRecord) -> PublishedGeneration:
        manifest = copy.deepcopy(record.as_dict())
        with self._lock:
            self._epoch_counter += 1
            published = PublishedGeneration(
                generation=record.generation,
                manifest=manifest,
                epoch=self._epoch_counter,
            )
            self._current = published
            self._history[published.generation] = published
            self._garbage_collect_locked()
            return published

    def pin(self) -> GenerationPin:
        with self._lock:
            if self._current is None:
                raise RuntimeError("No generation published")
            entry = self._current
            self._pins[entry.generation] = self._pins.get(entry.generation, 0) + 1
            manifest = copy.deepcopy(entry.manifest)
            return GenerationPin(self, entry.generation, entry.epoch, manifest)

    def current_manifest(self) -> Optional[Dict[str, object]]:
        with self._lock:
            if self._current is None:
                return None
            return copy.deepcopy(self._current.manifest)

    def manifest_for(self, generation: int) -> Optional[Dict[str, object]]:
        with self._lock:
            entry = self._history.get(generation)
            if entry is None:
                return None
            return copy.deepcopy(entry.manifest)

    def pinned_generations(self) -> List[int]:
        with self._lock:
            return sorted(self._pins.keys())

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch_counter

    def _release(self, generation: int) -> None:
        with self._lock:
            count = self._pins.get(generation)
            if count is None:
                return
            if count <= 1:
                self._pins.pop(generation, None)
            else:
                self._pins[generation] = count - 1
            self._garbage_collect_locked()

    def _garbage_collect_locked(self) -> None:
        if self._current is None:
            self._history.clear()
            return
        keep = {self._current.generation, *self._pins.keys()}
        for generation in list(self._history.keys()):
            if generation not in keep:
                del self._history[generation]


def _default_tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig()


def _default_attention_config() -> PRFAttentionConfig:
    return PRFAttentionConfig(
        dim=4,
        value_dim=4,
        features=16,
        gamma=0.95,
        tau=1.0,
        beta_floor=1e-3,
    )


def _default_linear_config() -> SparseLinearConfig:
    return SparseLinearConfig(capacity=512, tau_low=0.1, tau_high=0.9, learning_rate=0.05)


def _default_memory_config() -> FiniteMemoryConfig:
    return FiniteMemoryConfig(lift_coordinates=(0,), max_active=8, delay=1)


def _default_fusion_config() -> FusionConfig:
    return FusionConfig()


def _default_trust_config() -> TrustGateConfig:
    return TrustGateConfig()


def _default_ccr_config() -> CCRConfig:
    return CCRConfig()


def _default_bridge_config() -> BridgeConfig:
    return BridgeConfig()


def _default_cfr_config() -> CFRConfig:
    return CFRConfig()


def _default_tree_search_config() -> TreeSearchConfig:
    return TreeSearchConfig()


def _default_fp_contract() -> FPContract:
    return FPContract()


@dataclass(frozen=True)
class SeraConfig:
    tokenizer: TokenizerConfig = field(default_factory=_default_tokenizer_config)
    attention: PRFAttentionConfig = field(default_factory=_default_attention_config)
    linear: SparseLinearConfig = field(default_factory=_default_linear_config)
    memory: FiniteMemoryConfig = field(default_factory=_default_memory_config)
    fusion: FusionConfig = field(default_factory=_default_fusion_config)
    trust: TrustGateConfig = field(default_factory=_default_trust_config)
    ccr: CCRConfig = field(default_factory=_default_ccr_config)
    bridge: BridgeConfig = field(default_factory=_default_bridge_config)
    cfr: CFRConfig = field(default_factory=_default_cfr_config)
    tree_search: TreeSearchConfig = field(default_factory=_default_tree_search_config)
    lambda_floor: float = 0.05
    feature_budget: int = 32
    fp_contract: FPContract = field(default_factory=_default_fp_contract)

    def __post_init__(self) -> None:
        if self.lambda_floor <= 0:
            raise ValueError("lambda_floor must be positive")
        if self.feature_budget <= 0:
            raise ValueError("feature_budget must be positive")
        object.__setattr__(
            self,
            "tokenizer",
            _coerce_dataclass_config(self.tokenizer, TokenizerConfig, _default_tokenizer_config),
        )
        object.__setattr__(
            self,
            "attention",
            _coerce_dataclass_config(self.attention, PRFAttentionConfig, _default_attention_config),
        )
        object.__setattr__(
            self,
            "linear",
            _coerce_dataclass_config(self.linear, SparseLinearConfig, _default_linear_config),
        )
        object.__setattr__(
            self,
            "memory",
            _coerce_dataclass_config(self.memory, FiniteMemoryConfig, _default_memory_config),
        )
        object.__setattr__(
            self,
            "fusion",
            _coerce_dataclass_config(self.fusion, FusionConfig, _default_fusion_config),
        )
        object.__setattr__(
            self,
            "trust",
            _coerce_dataclass_config(self.trust, TrustGateConfig, _default_trust_config),
        )
        object.__setattr__(
            self,
            "ccr",
            _coerce_dataclass_config(self.ccr, CCRConfig, _default_ccr_config),
        )
        object.__setattr__(
            self,
            "bridge",
            _coerce_dataclass_config(self.bridge, BridgeConfig, _default_bridge_config),
        )
        object.__setattr__(
            self,
            "cfr",
            _coerce_dataclass_config(self.cfr, CFRConfig, _default_cfr_config),
        )
        object.__setattr__(
            self,
            "tree_search",
            _coerce_dataclass_config(
                self.tree_search, TreeSearchConfig, _default_tree_search_config
            ),
        )
        object.__setattr__(
            self,
            "fp_contract",
            _coerce_dataclass_config(self.fp_contract, FPContract, _default_fp_contract),
        )


def _model_config_to_sera_config(
    model_config: Mapping[str, object],
    metadata: Optional[Mapping[str, object]] = None,
) -> SeraConfig:
    if not isinstance(model_config, Mapping):
        raise TypeError("Model config must be a mapping")

    required_keys = {"tokenizer", "attention", "linear"}
    if required_keys.issubset(model_config.keys()):
        try:
            return SeraConfig(**model_config)  # type: ignore[arg-type]
        except Exception:
            pass

    base = SeraConfig()
    meta = metadata if isinstance(metadata, Mapping) else {}

    tokenizer_cfg = base.tokenizer
    tokenizer_meta = meta.get("tokenizer") if isinstance(meta, Mapping) else None
    if isinstance(tokenizer_meta, Mapping):
        pieces_blob = tokenizer_meta.get("pieces")
        vocabulary: Dict[bytes, int] = {}
        if isinstance(pieces_blob, Iterable):
            for entry in pieces_blob:
                piece: Optional[bytes] = None
                token_id: Optional[int] = None
                if isinstance(entry, Mapping):
                    piece = _coerce_bytes_like(entry.get("piece"))
                    token_id = _coerce_int(entry.get("token"))
                elif isinstance(entry, Sequence) and len(entry) >= 2:
                    piece = _coerce_bytes_like(entry[0])
                    token_id = _coerce_int(entry[1])
                if piece is None or token_id is None:
                    continue
                vocabulary[piece] = token_id
        max_piece_length = _coerce_int(tokenizer_meta.get("max_piece_length"))
        if vocabulary:
            edit_window = tokenizer_cfg.edit_window
            if max_piece_length is not None:
                edit_window = max(edit_window, max_piece_length)
            tokenizer_cfg = dataclasses.replace(
                tokenizer_cfg,
                vocabulary=vocabulary,
                max_piece_length=max_piece_length
                if max_piece_length is not None
                else tokenizer_cfg.max_piece_length,
                edit_window=edit_window,
            )
        elif max_piece_length is not None and max_piece_length > tokenizer_cfg.max_piece_length:
            tokenizer_cfg = dataclasses.replace(
                tokenizer_cfg,
                max_piece_length=max_piece_length,
                edit_window=max(tokenizer_cfg.edit_window, max_piece_length),
            )

    d_model = _coerce_int(model_config.get("d_model"))
    if d_model is None:
        d_model = _coerce_int(model_config.get("hidden_size"))
    if d_model is None or d_model <= 0:
        d_model = base.attention.dim

    attention_cfg = base.attention
    attention_meta = meta.get("attention") if isinstance(meta, Mapping) else None
    overlays_meta = meta.get("overlays") if isinstance(meta, Mapping) else None
    features = None
    tau = None
    beta_floor = None
    if isinstance(attention_meta, Mapping):
        features = _coerce_int(attention_meta.get("features"))
        tau = _coerce_float(attention_meta.get("tau"))
        whitening = attention_meta.get("whitening_sig2")
        if isinstance(whitening, Sequence):
            positives: List[float] = []
            for value in whitening:
                coerced = _coerce_float(value)
                if coerced is not None and coerced > 0.0:
                    positives.append(float(coerced))
            if positives:
                beta_floor = max(1e-3, min(positives))
    if tau is None:
        tau = _coerce_float(model_config.get("tau"))
    value_dim = None
    if isinstance(overlays_meta, Mapping):
        value_dim = _coerce_int(overlays_meta.get("cols"))
        if value_dim is None:
            value_dim = _coerce_int(overlays_meta.get("rank"))
    if value_dim is None or value_dim <= 0:
        value_dim = d_model if d_model > 0 else attention_cfg.value_dim
    if features is None or features <= 0:
        features = attention_cfg.features
    if tau is None or tau <= 0.0:
        tau = attention_cfg.tau
    if beta_floor is None or beta_floor <= 0.0:
        beta_floor = attention_cfg.beta_floor
    attention_cfg = dataclasses.replace(
        attention_cfg,
        dim=d_model,
        value_dim=value_dim,
        features=features,
        tau=tau,
        beta_floor=beta_floor,
    )

    linear_cfg = base.linear
    linear_meta = meta.get("linear") if isinstance(meta, Mapping) else None
    if isinstance(linear_meta, Mapping):
        capacity_candidates = [linear_cfg.capacity]
        keys_blob = linear_meta.get("keys")
        if isinstance(keys_blob, Sequence):
            capacity_candidates.append(len(keys_blob))
        slot_lookup = linear_meta.get("slot_lookup")
        if isinstance(slot_lookup, Mapping):
            capacity_candidates.append(len(slot_lookup))
        capacity = max(value for value in capacity_candidates if value > 0)
        linear_cfg = dataclasses.replace(linear_cfg, capacity=capacity)

    bridge_cfg = base.bridge
    bridge_meta = meta.get("bridge") if isinstance(meta, Mapping) else None
    if isinstance(bridge_meta, Mapping):
        legs = _coerce_int(bridge_meta.get("legs"))
        if legs is not None and legs > 0:
            bridge_cfg = dataclasses.replace(
                bridge_cfg,
                hub_window=max(bridge_cfg.hub_window, int(legs) * 4),
            )

    return dataclasses.replace(
        base,
        tokenizer=tokenizer_cfg,
        attention=attention_cfg,
        linear=linear_cfg,
        bridge=bridge_cfg,
    )


def _append_window(samples: Tuple[float, ...], window: int, value: float) -> Tuple[float, ...]:
    """Append a value to a bounded window represented as a tuple."""

    limit = int(window)
    if limit <= 0:
        return tuple()
    if limit == 1:
        return (float(value),)
    trimmed = list(samples[-(limit - 1) :]) if samples else []
    trimmed.append(float(value))
    if len(trimmed) > limit:
        trimmed = trimmed[-limit:]
    return tuple(trimmed)


def _quantile(samples: Sequence[float], percentile: float) -> float:
    """Compute the requested quantile for a bounded sample window."""

    if not samples:
        return 0.0
    ordered = sorted(float(v) for v in samples)
    rank = math.ceil(percentile * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return float(ordered[rank])


@dataclass
class SeraDiagnostics:
    tok_bytes_in: int = 0
    tok_emitted: int = 0
    tokenizer_probe_max: int = 0
    tokenizer_table_probes: int = 0
    tokens_emitted: int = 0
    attention_updates: int = 0
    attention_clip_rate: float = 0.0
    attention_den_min: float = field(default=float("inf"))
    lambda_star: float = 0.0
    bridge_hits: int = 0
    bridge_misses: int = 0
    tree_simulations: int = 0
    bridge_last_proof: bytes = field(default=b"", repr=False)
    p99_window: int = 128
    store_load_samples: Tuple[float, ...] = field(default_factory=tuple, repr=False)
    stash_occ_samples: Tuple[float, ...] = field(default_factory=tuple, repr=False)
    kick_len_samples: Tuple[float, ...] = field(default_factory=tuple, repr=False)
    store_load_p99: float = 0.0
    stash_occ_p99: float = 0.0
    kick_len_p99: float = 0.0
    capacity_lambda_hat: float = 0.0
    capacity_load: float = 0.0
    capacity_slack: float = 0.0
    capacity_margin: float = 0.0
    capacity_frozen: bool = False
    bridge_guard_rate: float = 0.0
    trust_decision: int = 0
    trust_m: int = 0
    trust_llr: float = 0.0
    trust_gamma: float = 0.0
    trust_consistent: bool = True
    trust_audit: bytes = field(default=b"", repr=False)
    trust_beta_min: float = 0.0
    trust_beta_cap: float = 0.0
    cfr_mode: str = "OFF"
    cfr_beta: float = 0.0
    cfr_guard: bool = False
    cfr_health_ok: bool = True
    cfr_y_cfr: float = 0.0

    def __post_init__(self) -> None:
        if self.p99_window < 1:
            self.p99_window = 1
        self.store_load_samples = tuple(float(v) for v in self.store_load_samples[-self.p99_window :])
        self.stash_occ_samples = tuple(float(v) for v in self.stash_occ_samples[-self.p99_window :])
        self.kick_len_samples = tuple(float(v) for v in self.kick_len_samples[-self.p99_window :])
        self.store_load_p99 = _quantile(self.store_load_samples, 0.99)
        self.stash_occ_p99 = _quantile(self.stash_occ_samples, 0.99)
        self.kick_len_p99 = _quantile(self.kick_len_samples, 0.99)
        if isinstance(self.bridge_last_proof, str):
            self.bridge_last_proof = bytes.fromhex(self.bridge_last_proof)
        if isinstance(self.trust_audit, str):
            self.trust_audit = bytes.fromhex(self.trust_audit)


@dataclass
class Sera:
    config: SeraConfig
    tokenizer: TokenizerState = field(init=False)
    attention: PRFAttentionState = field(init=False)
    linear: SparseLinearState = field(init=False)
    memory: FiniteMemoryState = field(init=False)
    fusion: FusionState = field(init=False)
    trust_gate: TrustGateState = field(init=False)
    ccr: CCRState = field(init=False)
    bridge: BridgeState = field(init=False)
    cfr: CFRState = field(init=False)
    tree_search: TreeSearchState = field(init=False)
    diagnostics: SeraDiagnostics = field(default_factory=SeraDiagnostics)
    generation: int = field(init=False, default=0)
    _generation_pointer: GenerationPointer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokenizer", TokenizerState(self.config.tokenizer))
        object.__setattr__(self, "attention", PRFAttentionState(self.config.attention))
        object.__setattr__(self, "linear", SparseLinearState(self.config.linear))
        object.__setattr__(self, "memory", FiniteMemoryState(self.config.memory))
        object.__setattr__(self, "fusion", FusionState(self.config.fusion))
        object.__setattr__(self, "trust_gate", TrustGateState(self.config.trust))
        object.__setattr__(self, "ccr", CCRState(self.config.ccr))
        object.__setattr__(self, "bridge", BridgeState(self.config.bridge))
        object.__setattr__(self, "cfr", CFRState(self.config.cfr))
        object.__setattr__(self, "tree_search", TreeSearchState(self.config.tree_search))
        self.bridge.advance_generation(self.generation)
        self.cfr.advance_generation(self.generation)
        object.__setattr__(self, "_generation_pointer", GenerationPointer())
        self._generation_pointer.publish(self.manifest_record())

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def configure(cls, params: Optional[Dict] = None) -> "Sera":
        """Create a Sera instance honouring the immutable configuration rules."""

        if params is None:
            return cls(SeraConfig())
        config = SeraConfig(**params)
        return cls(config)

    # ------------------------------------------------------------------
    # API surface (spec §1.4)
    # ------------------------------------------------------------------

    def step(
        self,
        bytes_data: Optional[bytes] = None,
        sparse_features: Optional[Iterable[Tuple[int, float]]] = None,
        key: Optional[Sequence[float]] = None,
        value: Optional[Sequence[float]] = None,
        query: Optional[Sequence[float]] = None,
        target: Optional[float] = None,
        bridge_ctx: Optional[int] = None,
        bridge_token: Optional[int] = None,
        trust_vector: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """Process a single event in O(1) time, returning the outputs.

        Each optional argument mirrors the inputs listed in spec §1.2.  The
        return dictionary exposes the outputs from spec §1.3.
        """

        with self.pin_generation():
            tokens: List[int] = []
            if bytes_data is not None:
                tokens = self.tokenizer.encode(bytes_data)
                stats = self.tokenizer.last_encode_stats
                if stats is not None:
                    diag = self.diagnostics
                    diag.tok_bytes_in += stats.bytes_in
                    diag.tok_emitted += stats.tokens_out
                    diag.tokenizer_probe_max = max(
                        diag.tokenizer_probe_max, stats.max_probes
                    )
                    diag.tokenizer_table_probes += stats.table_probes
                    diag.tokens_emitted = diag.tok_emitted

            y_att = None
            if key is not None and value is not None:
                self.attention.update(
                    np.asarray(key, dtype=float), np.asarray(value, dtype=float)
                )
                self.diagnostics.attention_updates += 1
                self.diagnostics.attention_clip_rate = self.attention.clip_rate

            att_features: List[Tuple[int, float]] = []
            phi_q = None
            if query is not None:
                y_att, phi_q = self.attention.read_with_overlays(
                    np.asarray(query, dtype=float), self.config.lambda_floor
                )
                att_features = self._attention_features(y_att, phi_q)
                denominator = self.attention.last_denominator
                if denominator is not None:
                    self.diagnostics.attention_den_min = min(
                        self.diagnostics.attention_den_min, float(denominator)
                    )
                self.diagnostics.attention_clip_rate = self.attention.clip_rate
                self.diagnostics.lambda_star = self.config.lambda_floor

            features = self._collect_features(sparse_features or [])
            for feature in att_features:
                if len(features) >= self.config.feature_budget:
                    raise BudgetError("Feature budget exceeded")
                features.append(feature)
            y_lin = self.linear.predict(features)
            if target is not None and self.cfr.allows_linear_update:
                self.linear.update(features, target)

            mem_value = self.memory.accumulate((idx, val) for idx, val in features)

            fused = self.fusion.fuse(y_att, y_lin)
            gated = self.fusion.gate(y_att, y_lin)

            trust = self.trust_gate.judge(self.diagnostics, trust_vector)
            diag = self.diagnostics
            diag.trust_decision = trust.decision
            diag.trust_m = trust.m
            diag.trust_llr = float(trust.llr)
            diag.trust_gamma = float(trust.gamma)
            diag.trust_consistent = trust.consistent
            diag.trust_audit = trust.audit
            diag.trust_beta_min = float(trust.beta_min)
            diag.trust_beta_cap = float(trust.beta_cap)
            trust_record = trust.as_dict()
            audit_blob = trust_record.get("audit")
            if isinstance(audit_blob, (bytes, bytearray)):
                trust_record["audit_hex"] = bytes(audit_blob).hex()
            else:
                trust_record["audit_hex"] = ""

            bridge_val = np.zeros(self.bridge.config.value_dim, dtype=float)
            guard = False
            proof = bytes(BridgeProof._SCHEMA.size)
            if bridge_ctx is not None:
                bridge_val, guard, proof = self.bridge.read(bridge_ctx, bridge_token)

            cfr_result = self.cfr.run(
                y_fus=fused,
                y_lin=y_lin,
                mem_value=mem_value,
                phi_q=phi_q,
                bridge_value=bridge_val,
                guard_ok=guard,
                trust_beta_min=trust.beta_min,
                trust_beta_cap=trust.beta_cap,
            )
            fused = float(cfr_result.y_mix)
            diag.cfr_mode = cfr_result.mode
            diag.cfr_beta = float(cfr_result.beta)
            diag.cfr_guard = bool(cfr_result.guard)
            diag.cfr_health_ok = bool(cfr_result.health_ok or bridge_ctx is None)
            diag.cfr_y_cfr = float(cfr_result.y_cfr)

            locals_vec = np.array([fused, gated], dtype=float)
            ccr_result = self.ccr.correct(locals_vec)

            if bridge_ctx is not None:
                if guard:
                    diag.bridge_hits += 1
                else:
                    diag.bridge_misses += 1
            diag.bridge_last_proof = proof
            total_guards = diag.bridge_hits + diag.bridge_misses
            if total_guards:
                diag.bridge_guard_rate = diag.bridge_hits / float(total_guards)
            else:
                diag.bridge_guard_rate = 0.0
            bridged = self.bridge.gate(
                gated,
                bridge_val,
                guard,
                witness_quality=cfr_result.s_witness,
                beta_min_override=trust.beta_min,
                beta_cap_override=trust.beta_cap,
            )

            self.tree_search.maybe_select()
            self.diagnostics.tree_simulations = self.tree_search._simulations
            self.diagnostics.lambda_star = self.config.lambda_floor

            store = self.linear._weights
            diag = self.diagnostics
            diag.store_load_samples = _append_window(
                diag.store_load_samples, diag.p99_window, store.load_factor
            )
            diag.store_load_p99 = _quantile(diag.store_load_samples, 0.99)
            stash_capacity = max(1, store.stash_capacity)
            diag.stash_occ_samples = _append_window(
                diag.stash_occ_samples,
                diag.p99_window,
                store.stash_load / float(stash_capacity),
            )
            diag.stash_occ_p99 = _quantile(diag.stash_occ_samples, 0.99)
            diag.kick_len_samples = _append_window(
                diag.kick_len_samples, diag.p99_window, store.max_kick_length
            )
            diag.kick_len_p99 = _quantile(diag.kick_len_samples, 0.99)

            capacity_monitor = self.linear._capacity_monitor
            diag.capacity_lambda_hat = capacity_monitor.lambda_hat
            diag.capacity_load = capacity_monitor.load
            diag.capacity_slack = capacity_monitor.slack
            diag.capacity_margin = capacity_monitor.margin
            diag.capacity_frozen = capacity_monitor.frozen

            return {
                "tokens": tokens,
                "y_att": y_att,
                "y_lin": y_lin,
                "memory": mem_value,
                "y_fus": fused,
                "y_gate": gated,
                "y_out": ccr_result.y,
                "ccr_residual": ccr_result.residual,
                "ccr_correction": ccr_result.correction,
                "y_bridge": bridged,
                "y_cfr": cfr_result.y_cfr,
                "trust_decision": trust.verdict(),
                "trust": trust_record,
            }

    def pin_generation(self) -> GenerationPin:
        """Pin the current generation for the duration of a critical section."""

        return self._generation_pointer.pin()

    @property
    def manifest_pointer(self) -> GenerationPointer:
        """Expose the generation pointer for diagnostics and testing."""

        return self._generation_pointer

    # ------------------------------------------------------------------
    # Bridge API (spec §1.4)
    # ------------------------------------------------------------------

    def bridge_read(self, ctx: int, token: Optional[int] = None) -> Dict[str, object]:
        value, guard, proof = self.bridge.read(ctx, token)
        guard_record = self.bridge.guard_record(ctx, token)
        guard_blob: Optional[Dict[str, float]] = None
        if guard_record is not None:
            guard_blob = guard_record.as_dict()
        return {"r_t": value, "guard_ok": guard, "proof64B": proof, "guard": guard_blob}

    # ------------------------------------------------------------------
    # Tree search hook (spec §1.4)
    # ------------------------------------------------------------------

    def maybe_selection_one_step(self, ctx: Optional[object] = None) -> None:
        del ctx
        self.tree_search.maybe_select()

    # ------------------------------------------------------------------
    # Persistence (spec §1.4)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, object]:
        return {
            "config": dataclasses.asdict(self.config),
            "linear_state": self.linear.snapshot(),
            "cfr_state": self.cfr.snapshot(),
            "diagnostics": self.diagnostics_record(),
            "generation": self.generation,
            "manifest": self.manifest_record().as_dict(),
        }

    @classmethod
    def restore(cls, blob: Dict[str, object]) -> "Sera":
        config = SeraConfig(**blob["config"])
        model = cls(config)
        linear_blob = blob.get("linear_state")
        if linear_blob is not None:
            model.linear = SparseLinearState.restore(model.config.linear, linear_blob)
        else:
            # Backwards compatibility
            for k, v in dict(blob["linear_weights"]).items():
                model.linear._weights.insert(int(k), float(v))
            model.linear._bias = float(blob.get("linear_bias", 0.0))
            model.linear._capacity_monitor.recompute(
                model.linear._address_book.size / float(model.linear.config.capacity)
            )
        diag_blob = dict(blob["diagnostics"])
        if diag_blob.get("attention_den_min") is None:
            diag_blob["attention_den_min"] = float("inf")
        model.diagnostics = SeraDiagnostics(**diag_blob)
        model.generation = int(blob.get("generation", 0))
        model.linear.advance_generation(model.generation)
        model.bridge.advance_generation(model.generation)
        cfr_blob = blob.get("cfr_state")
        if cfr_blob is not None:
            model.cfr = CFRState.restore(model.config.cfr, cfr_blob)
        model.cfr.advance_generation(model.generation)
        model._generation_pointer.publish(model.manifest_record())
        return model

    @classmethod
    def transfer(
        cls,
        artefact_root: Union[str, Path],
        *,
        state_file: Optional[Union[str, Path]] = None,
        allow_pickle: bool = False,
    ) -> Tuple["Sera", Dict[str, object]]:
        """Restore a model from a Sera Transfer Kit bundle (spec §17).

        Parameters
        ----------
        artefact_root:
            Directory containing the ``sera_manifest.bin`` file and snapshot(s).
        state_file:
            Optional explicit path to the runtime snapshot.  When omitted the
            loader probes for :data:`_TRANSFER_STATE_FILENAMES` in order.
        allow_pickle:
            If ``True`` the loader may restore ``.pkl``/``.pickle`` snapshots.
            Pickle files can execute arbitrary code during loading so the
            default ``False`` guards against accidental use.
        """

        manifest_dir = Path(artefact_root).expanduser()
        if manifest_dir.is_file():
            manifest_path = manifest_dir.resolve()
            manifest_dir = manifest_path.parent
        else:
            manifest_path = (manifest_dir / "sera_manifest.bin").resolve()

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Sera manifest not found at {manifest_path}"  # pragma: no cover - defensive
            )

        with manifest_path.open("rb") as fh:
            prefix_raw = fh.read(_TRANSFER_MANIFEST_PREFIX_STRUCT.size)
            if len(prefix_raw) != _TRANSFER_MANIFEST_PREFIX_STRUCT.size:
                raise ValueError("Sera manifest file is truncated")
            magic, version = _TRANSFER_MANIFEST_PREFIX_STRUCT.unpack(prefix_raw)
            if magic != _TRANSFER_MANIFEST_MAGIC:
                raise ValueError("Invalid Sera manifest magic")
            if version != _TRANSFER_MANIFEST_VERSION:
                raise ValueError(
                    f"Unsupported Sera manifest version: {version}"
                )
            seed_digest = fh.read(32)
            if len(seed_digest) != 32:
                raise ValueError("Sera manifest file is truncated")
            if seed_digest != _TRANSFER_EXPECTED_SEED_DIGEST:
                raise ValueError("Unexpected Sera manifest seed digest")
            schema_digest = fh.read(32)
            if len(schema_digest) != 32:
                raise ValueError("Sera manifest file is truncated")
        manifest_header = {
            "version": version,
            "seed_digest": seed_digest.hex(),
            "schema_sha256": schema_digest.hex(),
        }

        refused_pickle: Optional[Path] = None
        if state_file is not None:
            state_path = Path(state_file).expanduser()
            if not state_path.is_absolute():
                state_path = manifest_dir / state_path
            state_path = state_path.resolve()
            if (
                state_path.suffix.lower() in _PICKLE_SUFFIXES
                and not allow_pickle
                and state_path.exists()
            ):
                refused_pickle = state_path
        else:
            state_path = None
            for name in _TRANSFER_STATE_FILENAMES:
                candidate = (manifest_dir / name).resolve()
                if not candidate.exists():
                    continue
                if candidate.suffix.lower() in _PICKLE_SUFFIXES and not allow_pickle:
                    refused_pickle = candidate
                    continue
                state_path = candidate
                break
        if state_path is None or not state_path.exists():
            if refused_pickle is not None and not allow_pickle:
                raise RuntimeError(
                    "Refusing to load pickle snapshot without allow_pickle=True. "
                    "Pickle deserialisation can execute arbitrary code."
                )
            raise FileNotFoundError(
                "No Sera runtime snapshot found alongside the manifest"
            )

        state_blob = _load_transfer_state(state_path, allow_pickle=allow_pickle)
        if not isinstance(state_blob, Mapping):
            raise TypeError("Transfer snapshot must be a mapping")
        state_blob = _strip_transfer_private_fields(state_blob)

        runtime_blob_obj = state_blob.get("sera_snapshot")
        if runtime_blob_obj is None:
            runtime_blob = state_blob
        elif isinstance(runtime_blob_obj, Mapping):
            runtime_blob = _strip_transfer_private_fields(runtime_blob_obj)
        else:
            raise TypeError("Transfer snapshot 'sera_snapshot' must be a mapping")

        arrays_meta = state_blob.get("artefacts")
        if arrays_meta is None:
            arrays_info = {}
        elif isinstance(arrays_meta, Mapping):
            arrays_dir = (manifest_dir / "arrays").resolve()
            if not arrays_dir.exists():
                raise FileNotFoundError(
                    f"Sera arrays directory is missing: {arrays_dir}"  # pragma: no cover - defensive
                )
            if not arrays_dir.is_dir():
                raise NotADirectoryError(
                    f"Sera arrays path is not a directory: {arrays_dir}"  # pragma: no cover - defensive
                )
            arrays_info = _validate_transfer_arrays(arrays_dir, arrays_meta)
        else:
            raise TypeError("Transfer snapshot 'artefacts' must be a mapping if present")

        if isinstance(runtime_blob, Mapping) and "config" in runtime_blob:
            model = cls.restore(dict(runtime_blob))
        else:
            config_blob = state_blob.get("model_config")
            if not isinstance(config_blob, Mapping):
                raise ValueError("Transfer snapshot missing model configuration")
            metadata_blob = state_blob.get("metadata")
            metadata_map = metadata_blob if isinstance(metadata_blob, Mapping) else None
            config = _model_config_to_sera_config(config_blob, metadata_map)
            model = cls(config)

        metadata: Dict[str, object] = {
            "manifest_path": manifest_path,
            "state_path": state_path,
            "manifest_header": manifest_header,
            "arrays": {
                name: {
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                    "byte_len": info["byte_len"],
                    "path": info["path"],
                    "sha256": info.get("sha256"),
                    "flags": info.get("flags", 0),
                }
                for name, info in arrays_info.items()
            },
        }
        if "metadata" in state_blob:
            metadata["metadata"] = state_blob["metadata"]
        if "model_config" in state_blob:
            metadata["model_config"] = state_blob["model_config"]

        return model, metadata

    # ------------------------------------------------------------------
    # Generation + Manifest
    # ------------------------------------------------------------------

    def publish_generation(self) -> None:
        self.generation += 1
        self.linear.advance_generation(self.generation)
        self.bridge.advance_generation(self.generation)
        self.cfr.advance_generation(self.generation)
        self._generation_pointer.publish(self.manifest_record())

    def _manifest_sections(self) -> Dict[str, object]:
        return {
            "linear": self.linear.manifest(),
            "bridge": self.bridge.manifest(),
            "cfr": self.cfr.manifest(),
            "ccr_proof": self.ccr.certificate.as_dict(),
            "diagnostics": self.diagnostics_record(),
        }

    def manifest_record(self) -> ManifestRecord:
        return ManifestRecord(
            generation=self.generation,
            fp_contract=self.config.fp_contract,
            sections=self._manifest_sections(),
        )

    # ------------------------------------------------------------------
    # Diagnostics (spec §1.4)
    # ------------------------------------------------------------------

    def diagnostics_record(self) -> Dict[str, object]:
        record = dataclasses.asdict(self.diagnostics)
        if math.isinf(record["attention_den_min"]):
            record["attention_den_min"] = None
        proof_blob = record.get("bridge_last_proof")
        if isinstance(proof_blob, (bytes, bytearray)):
            record["bridge_last_proof"] = bytes(proof_blob).hex()
        audit_blob = record.get("trust_audit")
        if isinstance(audit_blob, (bytes, bytearray)):
            record["trust_audit"] = bytes(audit_blob).hex()
        return record

    # ------------------------------------------------------------------
    # Feature adapters
    # ------------------------------------------------------------------

    @staticmethod
    def _attention_features(y_att: Optional[np.ndarray], phi_q: Optional[np.ndarray]) -> List[Tuple[int, float]]:
        if y_att is None or phi_q is None:
            return []
        mean_att = float(np.mean(y_att))
        norm_phi = float(np.linalg.norm(phi_q))
        return [(-1, mean_att), (-2, norm_phi)]

    def _collect_features(
        self, features: Iterable[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        collected: List[Tuple[int, float]] = []
        budget = self.config.feature_budget
        for idx, value in features:
            if len(collected) >= budget:
                raise BudgetError("Feature budget exceeded")
            collected.append((int(idx), float(value)))
        return collected


__all__ = [
    "GenerationPin",
    "GenerationPointer",
    "ManifestRecord",
    "Sera",
    "SeraConfig",
    "CFRConfig",
    "TrustGateConfig",
]

_HASH_BASE = 257
_HASH_MASK = (1 << 64) - 1
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
_BIDI_CONTROL_CODEPOINTS: Set[int] = {
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2066,
    0x2067,
    0x2068,
    0x2069,
}
_DISALLOWED_UNICODE_POINTS: Set[int] = set(_BIDI_CONTROL_CODEPOINTS)
_DISALLOWED_UNICODE_POINTS.add(0x200D)  # ZERO WIDTH JOINER


def _mix64(value: int, seed: int = 0) -> int:
    """Deterministic 64-bit mixer derived from splitmix64 (spec §4.3)."""

    x = (value + seed * _SPLITMIX64_GAMMA) & _HASH_MASK
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _HASH_MASK
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _HASH_MASK
    x ^= x >> 31
    return x & _HASH_MASK


def _hash_bytes(data: bytes) -> int:
    """Compute the rolling hash RH_n for a byte sequence (spec §4.3)."""

    h = 0
    for byte in data:
        h = ((h * _HASH_BASE) + byte) & _HASH_MASK
    return h


def _decode_transfer_blob(blob):
    if isinstance(blob, str) and blob.startswith(_TRANSFER_JSON_BYTES_PREFIX):
        return bytes.fromhex(blob[len(_TRANSFER_JSON_BYTES_PREFIX) :])
    if isinstance(blob, Mapping):
        return {
            key: _decode_transfer_blob(value)
            for key, value in blob.items()
        }
    if isinstance(blob, list):
        return [_decode_transfer_blob(value) for value in blob]
    if isinstance(blob, tuple):
        return tuple(_decode_transfer_blob(value) for value in blob)
    return blob


def _strip_transfer_private_fields(blob):
    if isinstance(blob, Mapping):
        return {
            key: _strip_transfer_private_fields(value)
            for key, value in blob.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
    if isinstance(blob, list):
        return [_strip_transfer_private_fields(value) for value in blob]
    if isinstance(blob, tuple):
        return tuple(_strip_transfer_private_fields(value) for value in blob)
    return blob


def _load_transfer_state(path: Path, *, allow_pickle: bool = False):
    """Load a transfer snapshot from ``path`` with safe defaults.

    Parameters
    ----------
    path:
        Snapshot file to load.
    allow_pickle:
        Enable unpickling ``.pkl``/``.pickle`` snapshots.  Pickle deserialisation
        can execute arbitrary code so the default ``False`` protects callers.

    Raises
    ------
    RuntimeError
        If the snapshot format is unsupported or when refusing to load a pickle
        snapshot without explicit opt-in.
    """

    suffix = path.suffix.lower()
    if suffix in _PICKLE_SUFFIXES:
        if not allow_pickle:
            raise RuntimeError(
                "Pickle snapshots are disabled by default. Pass allow_pickle=True "
                "to acknowledge the code-execution risk."
            )
        with path.open("rb") as fh:
            return pickle.load(fh)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            blob = json.load(fh)
        return _decode_transfer_blob(blob)
    if suffix in {".msgpack", ".mpk"}:
        if importlib.util.find_spec("msgpack") is None:  # pragma: no cover - deterministic import guard
            raise RuntimeError("Support for msgpack snapshots requires the 'msgpack' package")
        import msgpack  # type: ignore
        with path.open("rb") as fh:
            blob = msgpack.unpack(fh, raw=False)
        return _decode_transfer_blob(blob)
    raise RuntimeError(f"Unsupported snapshot format: {path}")


def _parse_transfer_array_header(values: Sequence[int]) -> Dict[str, object]:
    magic = int(values[0])
    dtype_code = int(values[1])
    rank = int(values[2])
    dims = tuple(int(dim) for dim in values[3:8])
    shape = tuple(dims[:rank]) if rank > 0 else tuple()
    header = {
        "magic": magic,
        "dtype_code": dtype_code,
        "dtype": _TRANSFER_DTYPE_CODES.get(dtype_code, f"code{dtype_code}"),
        "rank": rank,
        "dims": dims,
        "shape": shape,
        "byte_len": int(values[8]),
        "crc32c": int(values[9]),
        "sha256_low64": int(values[10]),
        "flags": int(values[11]),
        "reserved": int(values[12]),
    }
    return header


def _validate_transfer_arrays(
    array_dir: Path, artefacts: Mapping[str, Mapping[str, object]]
) -> Dict[str, Dict[str, object]]:
    arrays: Dict[str, Dict[str, object]] = {}
    for name, record in sorted(artefacts.items()):
        if not isinstance(record, Mapping):
            raise TypeError("Array metadata entries must be mappings")
        array_path = array_dir / f"{name}.bin"
        if not array_path.exists():
            raise FileNotFoundError(f"Required array {array_path} is missing")
        file_size = array_path.stat().st_size
        with array_path.open("rb") as fh:
            header_raw = fh.read(_TRANSFER_ARRAY_STRUCT.size)
            if len(header_raw) != _TRANSFER_ARRAY_STRUCT.size:
                raise ValueError(f"Array file {array_path} is truncated")
            values = _TRANSFER_ARRAY_STRUCT.unpack(header_raw)
            header = _parse_transfer_array_header(values)
            if header["magic"] != _TRANSFER_ARRAY_MAGIC:
                raise ValueError(f"Invalid array magic for {array_path}")
            expected_len = header["byte_len"]
            crc_state = _CRC32C_INIT
            sha256 = hashlib.sha256()
            remaining = expected_len
            while remaining > 0:
                chunk = fh.read(min(_ARRAY_VALIDATION_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                crc_state = _crc32c_update(crc_state, chunk)
                sha256.update(chunk)
                remaining -= len(chunk)
            bytes_consumed = expected_len - remaining
            crc = (~crc_state) & 0xFFFFFFFF
            digest_bytes = sha256.digest()
        payload_len_on_disk = max(file_size - _TRANSFER_ARRAY_STRUCT.size, 0)
        if bytes_consumed != expected_len or payload_len_on_disk != expected_len:
            raise ValueError(
                f"Array payload length mismatch for {array_path}: expected {expected_len}, got {payload_len_on_disk}"
            )
        if crc != header["crc32c"]:
            raise ValueError(f"CRC32C mismatch for {array_path}")
        sha_low64 = int.from_bytes(digest_bytes[-8:], "little")
        if sha_low64 != header["sha256_low64"]:
            raise ValueError(f"sha256_low64 mismatch for {array_path}")
        expected_sha = record.get("sha256")
        if expected_sha:
            if digest_bytes.hex() != expected_sha:
                raise ValueError(
                    f"Array {array_path} failed SHA-256 validation: expected {expected_sha}, got {digest_bytes.hex()}"
                )
        flags = header["flags"]
        if flags & ~0x7:
            raise ValueError(
                f"Array {array_path} declares unsupported flag bits: 0x{flags:X}"
            )
        if not (flags & 0x1):
            raise ValueError(
                f"Array {array_path} must declare row-major layout via flag bit 0"
            )
        if header["reserved"]:
            raise ValueError(
                f"Array {array_path} reserved header field must be zero"
            )

        arrays[name] = {
            "dtype": header["dtype"],
            "shape": header["shape"],
            "byte_len": header["byte_len"],
            "path": array_path,
            "sha256": expected_sha,
            "flags": flags,
        }
    return arrays


def _init_crc32c_table() -> None:
    if _CRC32C_TABLE:
        return
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x82F63B78
            else:
                crc >>= 1
        _CRC32C_TABLE.append(crc & 0xFFFFFFFF)


def _crc32c_update(crc: int, data: bytes) -> int:
    _init_crc32c_table()
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc & 0xFFFFFFFF


def _crc32c(data: bytes) -> int:
    crc = _crc32c_update(_CRC32C_INIT, data)
    return (~crc) & 0xFFFFFFFF
