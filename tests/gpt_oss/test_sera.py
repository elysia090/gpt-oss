import importlib.util
import pathlib
import importlib.util
import pathlib
import sys

import pytest


pytest.importorskip("numpy")


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
