import importlib.util
import sys
from pathlib import Path

import pytest

import gpt_oss._compat.numpy_stub as numpy_stub

SERA_TRANSFER_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "gpt_oss" / "tools" / "sera_transfer.py"
)


def test_sera_transfer_import_requires_real_numpy(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_sera_transfer_requires_numpy", SERA_TRANSFER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        spec.loader.exec_module(module)

    sys.modules.pop(spec.name, None)
    assert "pip install numpy" in str(excinfo.value)
