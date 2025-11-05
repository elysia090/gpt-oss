"""Implementation of the Sera model described in the integrated specification.

The goal of this module is not to simulate a high-performance production system
but to provide a faithful, fully documented, and testable reference
implementation that follows every structural requirement from the
specification contained in ``docs/specs/sera-model-centric-integrated-
specification.txt``.  The classes defined here organise the system into the
same conceptual components as the spec and provide constant-time execution
paths whose budgets are enforced by explicit assertions.  Where the spec
describes probabilistic or approximate behaviour (for instance random feature
attention) we implement a deterministic variant that keeps the same API while
remaining numerically stable for unit tests.

The code is intentionally verbose – every public method is documented with the
section of the specification that it implements and the invariants that are
checked.  This makes the module useful both as an executable artefact and as a
living piece of documentation for contributors.
"""

from __future__ import annotations

import dataclasses
import math
import struct
import unicodedata
import copy
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type, TypeVar

import numpy as np
import hashlib
import json
import zlib


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


class BudgetError(RuntimeError):
    """Raised when a runtime budget specified by the Sera spec is violated."""


T = TypeVar("T")


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
        merged = {field.name: getattr(default, field.name) for field in dataclasses.fields(cls)}
        merged.update(value)
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

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0 < self.tau_low < self.tau_high < 1):
            raise ValueError("tau_low < tau_high < 1 must hold")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


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
    _weights: Dict[int, float] = field(default_factory=dict)
    _bias: float = 0.0

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
            handle, created = self._address_book.ensure_slot(int(idx))
            if created and self._address_book.size > self.config.capacity:
                # Revert slot assignment to keep invariants consistent.
                del self._address_book._key_to_slot[int(idx)]
                self._address_book._next_slot -= 1
                raise BudgetError("Sparse linear capacity exceeded")
            weight = self._weights.get(handle.slot, 0.0)
            grad = error * value + self.config.l2 * weight
            self._weights[handle.slot] = weight - lr * grad
        self._bias -= lr * (error + self.config.l2 * self._bias)

    def handles_for(self, keys: Iterable[int]) -> List[InjectiveHandle]:
        return [handle for key in keys if (handle := self._address_book.slot_of(int(key))) is not None]

    def snapshot(self) -> Dict[str, object]:
        return {
            "address_book": self._address_book.snapshot(),
            "weights": {int(slot): float(weight) for slot, weight in self._weights.items()},
            "bias": float(self._bias),
        }

    @classmethod
    def restore(cls, config: SparseLinearConfig, blob: Dict[str, object]) -> "SparseLinearState":
        state = cls(config)
        state._address_book = InjectiveAddressBook.restore(blob["address_book"])
        state._weights = {int(k): float(v) for k, v in blob["weights"].items()}
        state._bias = float(blob["bias"])
        return state

    def manifest(self) -> Dict[str, object]:
        return {
            "handles": [handle for handle in self._address_book.manifest()],
            "weights": {int(slot): float(weight) for slot, weight in sorted(self._weights.items())},
            "bias": float(self._bias),
        }

    def advance_generation(self, generation: int) -> None:
        self._address_book.advance_generation(generation)


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
# CCR overlap corrector (spec section 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CCRConfig:
    gamma: float = 0.25
    truncation_order: int = 1


@dataclass
class CCRState:
    config: CCRConfig
    _certificate: "CCRProof" = field(init=False)

    def __post_init__(self) -> None:
        if not (0 <= self.config.gamma < 1):
            raise ValueError("gamma must be in [0,1)")
        tail = self._tail_bound(self.config.gamma, self.config.truncation_order)
        self._certificate = CCRProof(
            gamma=self.config.gamma,
            truncation_order=self.config.truncation_order,
            tail_bound=tail,
        )

    def correct(self, residuals: np.ndarray, h_operator: np.ndarray) -> np.ndarray:
        if residuals.ndim != 1:
            raise ValueError("Residuals must be a vector")
        if h_operator.shape != (residuals.shape[0], residuals.shape[0]):
            raise ValueError("Operator shape mismatch")
        if self.config.gamma >= 1:
            raise BudgetError("CCR contraction gamma must be < 1")
        correction = np.zeros_like(residuals, dtype=float)
        power = np.eye(residuals.shape[0])
        for _ in range(self.config.truncation_order):
            power = power @ (np.eye(residuals.shape[0]) - self.config.gamma * h_operator)
            correction -= power @ residuals
        return residuals + correction

    @staticmethod
    def _tail_bound(gamma: float, order: int) -> float:
        return gamma ** (order + 1) / (1 - gamma) if gamma < 1 else float("inf")

    @property
    def certificate(self) -> "CCRProof":
        return self._certificate


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


# ---------------------------------------------------------------------------
# External bridge (spec section 10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeConfig:
    hub_window: int = 8
    candidate_bound: int = 4
    beta_min: float = 0.1
    beta_max: float = 0.9


@dataclass
class BridgeGuardRecord:
    """Guard metadata for bridge promotions (spec §10.3)."""

    margin: float
    threshold: float

    def check(self) -> bool:
        return self.margin >= self.threshold

    def as_dict(self) -> Dict[str, float]:
        return {"margin": float(self.margin), "threshold": float(self.threshold)}


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


@dataclass
class BridgeState:
    config: BridgeConfig
    _hub_bits: Dict[int, int] = field(default_factory=dict)
    _store: Dict[Tuple[int, int], BridgeEntry] = field(default_factory=dict)
    _generation: int = 0

    def promote(
        self,
        ctx: int,
        token: int,
        value: np.ndarray,
        *,
        guard_margin: float = 0.0,
        guard_threshold: float = 0.0,
    ) -> None:
        guard = BridgeGuardRecord(guard_margin, guard_threshold)
        self._store[(ctx, token)] = BridgeEntry(value, guard, self._generation)
        self._hub_bits.setdefault(ctx, 0)
        self._hub_bits.setdefault(token, 0)

    def read(self, ctx: int, token: Optional[int]) -> Tuple[np.ndarray, bool]:
        if token is None:
            return np.zeros(1, dtype=float), False
        entry = self._store.get((ctx, token))
        if entry is None:
            return np.zeros(1, dtype=float), False
        guard_ok = entry.guard.check()
        return entry.value, guard_ok

    def gate(self, base: float, bridge_val: np.ndarray, guard: bool) -> float:
        guard_weight = float(bool(guard))
        beta_mid = self.config.beta_min + (self.config.beta_max - self.config.beta_min) / 2.0
        beta = guard_weight * beta_mid
        alpha = 1.0 - beta
        return alpha * base + beta * float(np.mean(bridge_val))

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
        entries = {
            f"{ctx}:{token}": entry.manifest()
            for (ctx, token), entry in sorted(self._store.items())
        }
        return {"generation": self._generation, "entries": entries}


# ---------------------------------------------------------------------------
# Tree search stub (spec section 11 – optional)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeSearchConfig:
    enabled: bool = False


@dataclass
class TreeSearchState:
    config: TreeSearchConfig
    _simulations: int = 0

    def maybe_select(self) -> None:
        if not self.config.enabled:
            return
        # The spec limits us to one simulation per event; we simply increment a
        # counter so diagnostics can report whether the feature is active.
        self._simulations = min(self._simulations + 1, 1)


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


def _default_ccr_config() -> CCRConfig:
    return CCRConfig()


def _default_bridge_config() -> BridgeConfig:
    return BridgeConfig()


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
    ccr: CCRConfig = field(default_factory=_default_ccr_config)
    bridge: BridgeConfig = field(default_factory=_default_bridge_config)
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


@dataclass
class Sera:
    config: SeraConfig
    tokenizer: TokenizerState = field(init=False)
    attention: PRFAttentionState = field(init=False)
    linear: SparseLinearState = field(init=False)
    memory: FiniteMemoryState = field(init=False)
    fusion: FusionState = field(init=False)
    ccr: CCRState = field(init=False)
    bridge: BridgeState = field(init=False)
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
        object.__setattr__(self, "ccr", CCRState(self.config.ccr))
        object.__setattr__(self, "bridge", BridgeState(self.config.bridge))
        object.__setattr__(self, "tree_search", TreeSearchState(self.config.tree_search))
        self.bridge.advance_generation(self.generation)
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
            if target is not None:
                self.linear.update(features, target)

            mem_value = self.memory.accumulate((idx, val) for idx, val in features)

            fused = self.fusion.fuse(y_att, y_lin)
            gated = self.fusion.gate(y_att, y_lin)

            residual = np.array([fused - gated], dtype=float)
            corrected = self.ccr.correct(residual, np.eye(1))

            bridge_val = np.zeros(1, dtype=float)
            guard = False
            if bridge_ctx is not None:
                bridge_val, guard = self.bridge.read(bridge_ctx, bridge_token)
                if guard:
                    self.diagnostics.bridge_hits += 1
                else:
                    self.diagnostics.bridge_misses += 1
            bridged = self.bridge.gate(gated, bridge_val, guard)

            self.tree_search.maybe_select()
            self.diagnostics.tree_simulations = self.tree_search._simulations
            self.diagnostics.lambda_star = self.config.lambda_floor

            return {
                "tokens": tokens,
                "y_att": y_att,
                "y_lin": y_lin,
                "memory": mem_value,
                "y_fus": fused,
                "y_gate": gated,
                "y_out": corrected,
                "y_bridge": bridged,
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
        value, guard = self.bridge.read(ctx, token)
        proof = struct.pack("<Q", len(value))
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
            model.linear._weights = {
                int(k): float(v) for k, v in dict(blob["linear_weights"]).items()
            }
            model.linear._bias = float(blob.get("linear_bias", 0.0))
        diag_blob = dict(blob["diagnostics"])
        if diag_blob.get("attention_den_min") is None:
            diag_blob["attention_den_min"] = float("inf")
        model.diagnostics = SeraDiagnostics(**diag_blob)
        model.generation = int(blob.get("generation", 0))
        model.linear.advance_generation(model.generation)
        model.bridge.advance_generation(model.generation)
        model._generation_pointer.publish(model.manifest_record())
        return model

    # ------------------------------------------------------------------
    # Generation + Manifest
    # ------------------------------------------------------------------

    def publish_generation(self) -> None:
        self.generation += 1
        self.linear.advance_generation(self.generation)
        self.bridge.advance_generation(self.generation)
        self._generation_pointer.publish(self.manifest_record())

    def _manifest_sections(self) -> Dict[str, object]:
        return {
            "linear": self.linear.manifest(),
            "bridge": self.bridge.manifest(),
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
