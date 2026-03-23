"""
pytest conftest.py — path setup for consciousness miner tests.

Inserts src submodule directories directly into sys.path so that test files
can import `from core_system import ...` and `from consciousness import ...`
without triggering the package-level __init__.py files (which have heavy
optional dependencies like torch).
"""
import sys
import os

_src = os.path.join(os.path.dirname(__file__), "..", "src")
_fsot = os.path.join(_src, "fsot_core")
_neuro = os.path.join(_src, "neuromorphic_brain")
_root = os.path.join(os.path.dirname(__file__), "..")

for _p in [_root, _src, _fsot, _neuro]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
