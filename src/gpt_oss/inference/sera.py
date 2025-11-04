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
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


class BudgetError(RuntimeError):
    """Raised when a runtime budget specified by the Sera spec is violated."""


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
class TokenizerState:
    """State for the constant-time tokenizer (spec §2)."""

    config: TokenizerConfig
    _id_to_bytes: Dict[int, bytes] = field(init=False)

    def __post_init__(self) -> None:
        self._id_to_bytes = {idx: piece for piece, idx in self.config.vocabulary.items()}
        if len(self._id_to_bytes) != len(self.config.vocabulary):
            raise ValueError("Vocabulary ids must be unique")

    # -- Normaliser -----------------------------------------------------

    def _normalise_bytes(self, data: bytes) -> bytes:
        """Normalise UTF-8 bytes using NFC with bounded lookahead (spec §2.2)."""

        if not data:
            return data
        text = data.decode("utf-8", errors="strict")
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
        output: List[int] = []
        i = 0
        max_len = self.config.max_piece_length
        vocab = self.config.vocabulary
        while i < len(normalised):
            remaining = len(normalised) - i
            probe_count = 0
            for length in range(min(max_len, remaining), 0, -1):
                probe_count += 1
                if probe_count > max_len:
                    raise BudgetError("Tokenizer probe budget exceeded")
                window = normalised[i : i + length]
                token_id = vocab.get(window)
                if token_id is not None:
                    output.append(token_id)
                    i += length
                    break
            else:
                # Fallback to single-byte atom per spec §2.4.
                byte = normalised[i : i + 1]
                token_id = vocab.get(byte)
                if token_id is None:
                    raise KeyError(f"Byte {byte!r} missing from vocabulary")
                output.append(token_id)
                i += 1
            if len(output) > self.config.max_event_tokens:
                raise BudgetError("Tokenizer token budget exceeded")
        return output

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
        phi = np.exp(scaled - norm / (2 * self.config.tau)) / math.sqrt(self.config.features)
        if self.config.clip_value is not None:
            phi = np.clip(phi, -self.config.clip_value, self.config.clip_value)
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


@dataclass
class SparseLinearState:
    config: SparseLinearConfig
    _weights: Dict[int, float] = field(default_factory=dict)
    _bias: float = 0.0

    def predict(self, features: Iterable[Tuple[int, float]]) -> float:
        total = self._bias
        comp = 0.0
        for idx, value in features:
            weight = self._weights.get(idx, 0.0)
            total, comp = _kahan_update(total, comp, weight * value)
        return total

    def update(self, features: Iterable[Tuple[int, float]], target: float) -> None:
        prediction = self.predict(features)
        error = prediction - target
        lr = self.config.learning_rate
        for idx, value in features:
            if idx not in self._weights and len(self._weights) >= self.config.capacity:
                raise BudgetError("Sparse linear capacity exceeded")
            grad = error * value + self.config.l2 * self._weights.get(idx, 0.0)
            self._weights[idx] = self._weights.get(idx, 0.0) - lr * grad
        self._bias -= lr * (error + self.config.l2 * self._bias)


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

    def correct(self, residuals: np.ndarray, h_operator: np.ndarray) -> np.ndarray:
        if residuals.ndim != 1:
            raise ValueError("Residuals must be a vector")
        if h_operator.shape != (residuals.shape[0], residuals.shape[0]):
            raise ValueError("Operator shape mismatch")
        correction = np.zeros_like(residuals, dtype=float)
        power = np.eye(residuals.shape[0])
        for _ in range(self.config.truncation_order):
            power = power @ (np.eye(residuals.shape[0]) - self.config.gamma * h_operator)
            correction -= power @ residuals
        return residuals + correction


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
class BridgeState:
    config: BridgeConfig
    _hub_bits: Dict[int, int] = field(default_factory=dict)
    _store: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)

    def promote(self, ctx: int, token: int, value: np.ndarray) -> None:
        self._store[(ctx, token)] = value
        self._hub_bits.setdefault(ctx, 0)
        self._hub_bits.setdefault(token, 0)

    def read(self, ctx: int, token: Optional[int]) -> Tuple[np.ndarray, bool]:
        if token is None:
            return np.zeros(1, dtype=float), False
        value = self._store.get((ctx, token))
        if value is None:
            return np.zeros(1, dtype=float), False
        return value, True

    def gate(self, base: float, bridge_val: np.ndarray, guard: bool) -> float:
        guard_weight = float(bool(guard))
        beta_mid = self.config.beta_min + (self.config.beta_max - self.config.beta_min) / 2.0
        beta = guard_weight * beta_mid
        alpha = 1.0 - beta
        return alpha * base + beta * float(np.mean(bridge_val))


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
class SeraConfig:
    tokenizer: TokenizerConfig = TokenizerConfig()
    attention: PRFAttentionConfig = PRFAttentionConfig(
        dim=4,
        value_dim=4,
        features=16,
        gamma=0.95,
        tau=1.0,
        beta_floor=1e-3,
    )
    linear: SparseLinearConfig = SparseLinearConfig(
        capacity=512, tau_low=0.1, tau_high=0.9, learning_rate=0.05
    )
    memory: FiniteMemoryConfig = FiniteMemoryConfig(lift_coordinates=(0,), max_active=8, delay=1)
    fusion: FusionConfig = FusionConfig()
    ccr: CCRConfig = CCRConfig()
    bridge: BridgeConfig = BridgeConfig()
    tree_search: TreeSearchConfig = TreeSearchConfig()
    lambda_floor: float = 0.05
    feature_budget: int = 32

    def __post_init__(self) -> None:
        if self.lambda_floor <= 0:
            raise ValueError("lambda_floor must be positive")
        if self.feature_budget <= 0:
            raise ValueError("feature_budget must be positive")


@dataclass
class SeraDiagnostics:
    tokens_emitted: int = 0
    attention_updates: int = 0
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokenizer", TokenizerState(self.config.tokenizer))
        object.__setattr__(self, "attention", PRFAttentionState(self.config.attention))
        object.__setattr__(self, "linear", SparseLinearState(self.config.linear))
        object.__setattr__(self, "memory", FiniteMemoryState(self.config.memory))
        object.__setattr__(self, "fusion", FusionState(self.config.fusion))
        object.__setattr__(self, "ccr", CCRState(self.config.ccr))
        object.__setattr__(self, "bridge", BridgeState(self.config.bridge))
        object.__setattr__(self, "tree_search", TreeSearchState(self.config.tree_search))

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

        tokens: List[int] = []
        if bytes_data is not None:
            tokens = self.tokenizer.encode(bytes_data)
            self.diagnostics.tokens_emitted += len(tokens)

        y_att = None
        if key is not None and value is not None:
            self.attention.update(np.asarray(key, dtype=float), np.asarray(value, dtype=float))
            self.diagnostics.attention_updates += 1

        att_features: List[Tuple[int, float]] = []
        phi_q = None
        if query is not None:
            y_att, phi_q = self.attention.read_with_overlays(
                np.asarray(query, dtype=float), self.config.lambda_floor
            )
            att_features = self._attention_features(y_att, phi_q)

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

    # ------------------------------------------------------------------
    # Bridge API (spec §1.4)
    # ------------------------------------------------------------------

    def bridge_read(self, ctx: int, token: Optional[int] = None) -> Dict[str, object]:
        value, guard = self.bridge.read(ctx, token)
        proof = struct.pack("<Q", len(value))
        return {"r_t": value, "guard_ok": guard, "proof64B": proof}

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
            "linear_weights": dict(self.linear._weights),
            "linear_bias": self.linear._bias,
            "diagnostics": dataclasses.asdict(self.diagnostics),
        }

    @classmethod
    def restore(cls, blob: Dict[str, object]) -> "Sera":
        config = SeraConfig(**blob["config"])
        model = cls(config)
        model.linear._weights = dict(blob["linear_weights"])
        model.linear._bias = float(blob["linear_bias"])
        model.diagnostics = SeraDiagnostics(**blob["diagnostics"])
        return model

    # ------------------------------------------------------------------
    # Diagnostics (spec §1.4)
    # ------------------------------------------------------------------

    def diagnostics_record(self) -> Dict[str, object]:
        return dataclasses.asdict(self.diagnostics)

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


__all__ = ["Sera", "SeraConfig"]

