import importlib.util
import builtins
import sys
from pathlib import Path

import pytest

SERA_TRANSFER_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "gpt_oss" / "tools" / "sera_transfer.py"
)


def test_sera_transfer_import_requires_real_numpy(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_sera_transfer_requires_numpy", SERA_TRANSFER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ARG001
        if name == "numpy":
            raise ModuleNotFoundError("No module named 'numpy'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        spec.loader.exec_module(module)

    sys.modules.pop(spec.name, None)
    assert "pip install numpy" in str(excinfo.value)
