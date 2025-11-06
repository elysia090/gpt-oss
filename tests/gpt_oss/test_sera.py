import hashlib
import importlib.util
import pathlib
import importlib.util
import pathlib
import sys

import pytest


try:  # pragma: no cover - mirrors module fallback for test envs
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - deterministic fallback
    from gpt_oss._compat import numpy_stub as np


_SERA_PATH = pathlib.Path(__file__).resolve().parents[2] / "src" / "gpt_oss" / "inference" / "sera.py"
_SERA_SPEC = importlib.util.spec_from_file_location("_sera_module", _SERA_PATH)
if _SERA_SPEC is None or _SERA_SPEC.loader is None:  # pragma: no cover - defensive guard
    raise RuntimeError("Unable to load Sera module for tests")
_SERA_MODULE = importlib.util.module_from_spec(_SERA_SPEC)
sys.modules[_SERA_SPEC.name] = _SERA_MODULE
_SERA_SPEC.loader.exec_module(_SERA_MODULE)

BudgetError = _SERA_MODULE.BudgetError
Sera = _SERA_MODULE.Sera
SeraConfig = _SERA_MODULE.SeraConfig
TokenizerConfig = _SERA_MODULE.TokenizerConfig
TokenizerState = _SERA_MODULE.TokenizerState
SparseLinearConfig = _SERA_MODULE.SparseLinearConfig
CCRConfig = _SERA_MODULE.CCRConfig
CCRState = _SERA_MODULE.CCRState
CCRResult = _SERA_MODULE.CCRResult
TrustGateConfig = _SERA_MODULE.TrustGateConfig
BridgeConfig = _SERA_MODULE.BridgeConfig
BridgeState = _SERA_MODULE.BridgeState


def test_tokenizer_enforces_event_budgets() -> None:
    config = TokenizerConfig(max_event_bytes=4, max_event_tokens=4)
    tokenizer_model = SeraConfig(tokenizer=config)
    model = Sera(tokenizer_model)

    assert model.tokenizer.encode(b"abcd")
    assert model.tokenizer.decode([0, 1, 2, 3])

    too_long = b"abcde"
    try:
        model.tokenizer.encode(too_long)
    except BudgetError:
        pass
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Byte budget violation should raise BudgetError")

    try:
        model.tokenizer.decode(list(range(5)))
    except BudgetError:
        pass
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Token budget violation should raise BudgetError")


def test_step_enforces_feature_budget() -> None:
    config = SeraConfig(feature_budget=2)
    model = Sera(config)

    # Exactly at the budget works when sparse features fill the limit.
    outputs = model.step(sparse_features=[(0, 1.0), (1, 2.0)])
    assert "y_lin" in outputs

    # Exceeding the budget raises an error even before attention features.
    try:
        model.step(sparse_features=[(0, 1.0), (1, 2.0), (2, 3.0)])
    except BudgetError:
        pass
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Feature budget violation should raise BudgetError")

    # Query-only call uses attention features and still respects the budget.
    query = [1.0] * model.config.attention.dim
    outputs = model.step(query=query)
    assert outputs["y_att"] is not None


def test_sera_config_allows_partial_nested_overrides() -> None:
    config = SeraConfig(attention={"dim": 8})

    assert config.attention.dim == 8
    # Other values come from the canonical defaults.
    assert config.attention.value_dim == 4
    assert config.attention.features == 16


def test_sera_config_provides_fresh_defaults() -> None:
    config_a = SeraConfig()
    config_b = SeraConfig()

    assert config_a is not config_b
    assert config_a.attention is not config_b.attention
    assert config_a.linear is not config_b.linear


def test_tokenizer_sp_trace_and_digest() -> None:
    tokenizer = TokenizerState(TokenizerConfig())

    assert tokenizer.sp_trace == ()
    assert tokenizer.sp_digest == hashlib.sha256(b"").hexdigest()


def test_tokenizer_retokenize_window_within_radius() -> None:
    config = TokenizerConfig(edit_window=8, max_piece_length=4)
    tokenizer = TokenizerState(config)
    original = b"hello world"
    tokens = tokenizer.encode(original)
    normalised = tokenizer._normalise_bytes(original)
    replacement = tokenizer._normalise_bytes(b"there")

    new_tokens, token_range, new_normalised = tokenizer.retokenize_window(
        normalised, tokens, 6, 11, replacement
    )

    assert new_normalised == tokenizer._normalise_bytes(b"hello there")
    stitched = tokens[: token_range[0]] + new_tokens + tokens[token_range[1] :]
    assert tokenizer.decode(stitched) == new_normalised


@pytest.mark.parametrize("payload", ["a\u200db", "a\u202E b"])
def test_tokenizer_rejects_disallowed_unicode(payload: str) -> None:
    tokenizer = TokenizerState(TokenizerConfig())

    with pytest.raises(ValueError):
        tokenizer.encode(payload.encode("utf-8"))


def test_sparse_linear_manifest_includes_mph_and_cuckoo() -> None:
    config = SeraConfig()
    model = Sera(config)

    model.linear.update([(1, 1.0)], target=0.5)

    manifest = model.linear.manifest()
    handles = manifest["handles"]
    assert handles["size"] == 1
    assert "mph" in handles
    mph = handles["mph"]
    assert mph["size"] >= handles["size"]
    weights = manifest["weights"]
    assert weights["capacity"] == config.linear.capacity
    assert weights["choices"] == config.linear.buckets
    assert weights["stash_capacity"] == config.linear.stash_capacity
    capacity = manifest["capacity"]
    assert capacity["tau_low"] == pytest.approx(config.linear.tau_low)
    assert "slack" in capacity


def test_bridge_manifest_includes_config_metadata() -> None:
    bridge_config = BridgeConfig(
        projection=((0.0, 1.0), (1.0, 0.0)),
        margin_policy=0.1,
        eps_row_policy=0.05,
        load_max=0.75,
        stash_max=0.5,
        kick_max=4.0,
    )
    config = SeraConfig(bridge=bridge_config)
    model = Sera(config)

    manifest = model.bridge.manifest()
    bridge_blob = manifest["config"]

    assert bridge_blob["hub_window"] == config.bridge.hub_window
    assert bridge_blob["candidate_bound"] == config.bridge.candidate_bound
    projection_blob = bridge_blob["projection"]
    expected_shape = (len(config.bridge.projection), config.bridge.value_dim)
    assert tuple(projection_blob["shape"]) == expected_shape
    assert projection_blob["digest"] == config.bridge.projection_digest
    guard_blob = bridge_blob["guard"]
    assert guard_blob["margin_policy"] == pytest.approx(config.bridge.margin_policy)
    assert guard_blob["eps_row_policy"] == pytest.approx(config.bridge.eps_row_policy)
    limits = bridge_blob["store_limits"]
    assert limits["load_max"] == pytest.approx(config.bridge.load_max)
    assert limits["stash_max"] == pytest.approx(config.bridge.stash_max)
    assert limits["kick_max"] == pytest.approx(config.bridge.kick_max)
    assert bridge_blob["route_proof_schema_digest"] == config.bridge.route_proof_schema_digest


def test_sparse_linear_tau_freeze() -> None:
    config = SeraConfig(
        linear=SparseLinearConfig(
            capacity=4,
            tau_low=0.1,
            tau_high=0.5,
            learning_rate=0.1,
            buckets=2,
            bucket_size=2,
            stash_capacity=1,
            max_kicks=2,
            ring_capacity=1,
            margin=0.0,
        )
    )
    model = Sera(config)

    model.linear.update([(0, 1.0)], target=0.0)
    with pytest.raises(BudgetError):
        model.linear.update([(1, 1.0)], target=0.0)
    capacity = model.linear.manifest()["capacity"]
    assert capacity["frozen"] is True


def test_ccr_small_block_precomputation() -> None:
    state = CCRState(CCRConfig())
    blocks = state.blocks

    assert blocks.dim == 2
    assert len(blocks.B_blocks) == 3
    assert len(blocks.W_blocks) == 3
    assert blocks.h.shape == (2, 2)
    assert blocks.h_series.shape == (2, 2)
    assert blocks.pi.shape == (2,)
    assert float(sum(blocks.pi.tolist())) == pytest.approx(1.0)


def test_ccr_correct_constructs_residual_and_output() -> None:
    state = CCRState(CCRConfig())
    locals_vec = np.array([3.0, 1.0], dtype=float)

    result = state.correct(locals_vec)

    assert isinstance(result, CCRResult)
    assert result.residual.shape == (2,)
    assert float(sum(result.residual.tolist())) == pytest.approx(0.0)
    assert result.correction.shape == (2,)
    assert result.corrected_locals.shape == (2,)
    assert result.y == pytest.approx(float(np.mean(result.corrected_locals)))


def test_step_exposes_ccr_outputs() -> None:
    model = Sera(SeraConfig())

    outputs = model.step()

    residual = outputs["ccr_residual"]
    correction = outputs["ccr_correction"]

    assert isinstance(residual, np.ndarray)
    assert residual.shape == (2,)
    assert isinstance(correction, np.ndarray)
    assert correction.shape == (2,)


def test_step_includes_trust_outputs() -> None:
    model = Sera(SeraConfig())

    outputs = model.step()

    assert "trust_decision" in outputs
    assert outputs["trust_decision"] in {True, False, None}
    assert "trust" in outputs
    trust_blob = outputs["trust"]
    assert trust_blob["decision"] in (-1, 0, 1)
    assert "audit_hex" in trust_blob


def test_trust_gate_adjusts_beta_caps() -> None:
    trust_config = TrustGateConfig(
        dimension=6,
        salts=2,
        acceptance_k=1,
        pos_threshold=0.05,
        neg_threshold=-0.2,
        q_min_hat=0.9,
        eps_pos_hat=0.1,
        pi0=0.5,
        hazard_positive=2.0,
        hazard_negative=1.0,
        psi=0.0,
        beta_min=0.05,
        beta_max=0.4,
        beta_boost=0.2,
    )
    config = SeraConfig(trust=trust_config, bridge=BridgeConfig(beta_min=0.05, beta_max=0.4))
    model = Sera(config)

    trust_state = model.trust_gate
    rows = trust_state._projections  # type: ignore[attr-defined]
    norms = trust_state._row_norms  # type: ignore[attr-defined]
    k = trust_state.config.acceptance_k
    pos_vec = np.zeros(trust_state.config.dimension, dtype=float)
    for idx in range(k):
        pos_vec += rows[idx] / norms[idx]
    neg_vec = -pos_vec

    model.step(trust_vector=pos_vec.tolist())
    diag = model.diagnostics
    assert diag.trust_decision == 1
    assert diag.trust_m >= trust_config.acceptance_k
    assert diag.trust_beta_min == pytest.approx(trust_config.beta_boost)
    assert diag.trust_beta_cap == pytest.approx(trust_config.beta_max)
    assert isinstance(diag.trust_audit, bytes)
    assert len(diag.trust_audit) == 64

    model.step(trust_vector=neg_vec.tolist())
    diag = model.diagnostics
    assert diag.trust_decision == -1
    assert diag.trust_beta_cap == pytest.approx(0.0)
    assert diag.trust_beta_min == pytest.approx(0.0)


def test_bridge_promote_respects_guard_policies() -> None:
    config = BridgeConfig(margin_policy=0.25, eps_row_policy=0.1)
    bridge = BridgeState(config)

    with pytest.raises(ValueError):
        bridge.promote(1, 1, np.array([0.0]), guard_margin=0.2)

    with pytest.raises(ValueError):
        bridge.promote(1, 1, np.array([0.0]), guard_margin=0.25, guard_eps_row=0.05)

    bridge.promote(1, 1, np.array([0.0]), guard_margin=0.3, guard_eps_row=0.1)


def test_bridge_gate_projects_values() -> None:
    config = BridgeConfig(beta_min=0.2, beta_max=0.6, projection=((0.0, 1.0),))
    bridge = BridgeState(config)

    base = 2.0
    value = np.array([3.0, 5.0])
    result = bridge.gate(base, value, guard=True, witness_quality=0.5)

    beta = config.beta_min + (config.beta_max - config.beta_min) * 0.5
    expected = (1.0 - beta) * base + beta * 5.0
    assert result == pytest.approx(expected)
