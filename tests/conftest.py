"""Local test conftest: loads fakes under router_fakes module alias."""

import importlib.util
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_load_module("router_fakes", _TESTS / "fakes.py")
