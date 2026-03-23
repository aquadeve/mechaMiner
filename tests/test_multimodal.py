#!/usr/bin/env python3
"""
Tests for the ConsciousnessMiner class (without real network connections).
Uses a mock socket to validate the miner's internal logic.
"""

import math
import sys
import os
import json
import struct
import hashlib
import binascii
from unittest.mock import MagicMock, patch, call
from typing import List

_root = os.path.join(os.path.dirname(__file__), "..")
_src = os.path.join(_root, "src")
for _d in [_root, _src, os.path.join(_src, "fsot_core"), os.path.join(_src, "neuromorphic_brain")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from consciousness_miner import (
    ConsciousnessMiner,
    calculate_merkle_root,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_USERNAME,
)
from core_system import ConsciousnessState, evaluate_hash_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_sock(messages: list):
    """Return a mock socket that yields encoded JSON messages."""
    mock_sock = MagicMock()
    encoded = []
    for msg in messages:
        encoded.append((json.dumps(msg) + "\n").encode())
    encoded.append(b"")  # simulate connection close
    mock_sock.recv.side_effect = encoded
    return mock_sock


def _make_miner() -> ConsciousnessMiner:
    return ConsciousnessMiner(
        host="localhost",
        port=9999,
        username="test_user",
        password="x",
    )


# ---------------------------------------------------------------------------
# ConsciousnessMiner initialisation tests
# ---------------------------------------------------------------------------


def test_miner_defaults():
    miner = ConsciousnessMiner()
    assert miner.host == DEFAULT_HOST
    assert miner.port == DEFAULT_PORT
    assert miner.username == DEFAULT_USERNAME


def test_miner_custom_args():
    miner = ConsciousnessMiner(host="mypool.com", port=1234, username="wallet.worker")
    assert miner.host == "mypool.com"
    assert miner.port == 1234


def test_miner_initial_counters():
    miner = _make_miner()
    assert miner.total_hashes == 0
    assert miner.shares_submitted == 0
    assert miner.shares_accepted == 0
    assert miner.extranonce2_counter == 0


def test_miner_consciousness_state():
    miner = _make_miner()
    assert isinstance(miner.consciousness, ConsciousnessState)
    assert len(miner.consciousness.vector) == ConsciousnessState.TOTAL_DIM


# ---------------------------------------------------------------------------
# calculate_merkle_root tests
# ---------------------------------------------------------------------------


def test_merkle_root_empty_branch():
    coinbase_hash = hashlib.sha256(b"coinbase").digest()
    root = calculate_merkle_root(coinbase_hash, [])
    # With no branch, root is just the reversed coinbase hash
    expected = coinbase_hash[::-1]
    assert root == expected


def test_merkle_root_single_branch():
    coinbase_hash = hashlib.sha256(b"cb").digest()
    branch_hash = hashlib.sha256(b"branch").hexdigest()
    root = calculate_merkle_root(coinbase_hash, [branch_hash])
    assert len(root) == 32  # 32 bytes


def test_merkle_root_deterministic():
    cb = hashlib.sha256(b"test").digest()
    branch = [hashlib.sha256(b"b0").hexdigest(), hashlib.sha256(b"b1").hexdigest()]
    r1 = calculate_merkle_root(cb, branch)
    r2 = calculate_merkle_root(cb, branch)
    assert r1 == r2


# ---------------------------------------------------------------------------
# mine_job internal logic tests (no network)
# ---------------------------------------------------------------------------


def _minimal_job_params():
    """Return a minimal set of valid-looking job parameters."""
    version = "20000000"
    prevhash = "00" * 32
    ntime = "60aabbcc"
    nbits = "1d00ffff"
    # Simple coinbase (empty body)
    coinb1 = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff"
    coinb2 = "00000000"
    merkle_branch: List[str] = []
    job_id = "test_job_001"
    return job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime


def test_mine_job_increments_hashes():
    miner = _make_miner()
    miner.sock = MagicMock()
    miner.extranonce1 = "00000001"
    miner.extranonce2_size = 4

    job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime = _minimal_job_params()
    before = miner.total_hashes
    miner.mine_job(job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime)
    assert miner.total_hashes > before


def test_mine_job_updates_consciousness():
    miner = _make_miner()
    miner.sock = MagicMock()
    miner.extranonce1 = "00000001"
    miner.extranonce2_size = 4

    before_step = miner.consciousness._step
    job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime = _minimal_job_params()
    miner.mine_job(job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime)
    assert miner.consciousness._step > before_step


def test_mine_job_increments_extranonce2_counter():
    miner = _make_miner()
    miner.sock = MagicMock()
    miner.extranonce1 = "00000001"
    miner.extranonce2_size = 4

    before = miner.extranonce2_counter
    job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime = _minimal_job_params()
    miner.mine_job(job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime)
    assert miner.extranonce2_counter == before + 1


def test_mine_job_memory_grows():
    miner = _make_miner()
    miner.sock = MagicMock()
    miner.extranonce1 = "00000001"
    miner.extranonce2_size = 4

    job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime = _minimal_job_params()
    miner.mine_job(job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime)
    assert len(miner.thought_engine.memory) > 0


def test_mine_job_returns_true():
    miner = _make_miner()
    miner.sock = MagicMock()
    miner.extranonce1 = "00000001"
    miner.extranonce2_size = 4

    result = miner.mine_job(*_minimal_job_params())
    assert result is True


# ---------------------------------------------------------------------------
# print_status smoke test
# ---------------------------------------------------------------------------


def test_print_status_no_crash(capsys):
    miner = _make_miner()
    miner.sock = MagicMock()
    miner.extranonce1 = "00000001"
    miner.extranonce2_size = 4
    # Run one job so there is real data
    miner.mine_job(*_minimal_job_params())
    miner.print_status()
    captured = capsys.readouterr()
    assert "Hashes" in captured.out


# ---------------------------------------------------------------------------
# evaluate_hash_quality edge cases
# ---------------------------------------------------------------------------


def test_evaluate_invalid_hex():
    error = evaluate_hash_quality("GGGG", target_bits=1)
    assert error == 1.0


def test_evaluate_all_zeros():
    error = evaluate_hash_quality("0" * 64, target_bits=64)
    assert error == 0.0


# ---------------------------------------------------------------------------
# Standalone runner (same pattern as test_brain.py)
# ---------------------------------------------------------------------------


def pytest_approx(expected, abs=None, rel=None):
    return _Approx(expected, abs=abs, rel=rel)


class _Approx:
    def __init__(self, expected, abs=None, rel=None):
        self._expected = expected
        self._abs = abs if abs is not None else 1e-6
        self._rel = rel

    def __eq__(self, other):
        return math.isclose(other, self._expected, abs_tol=self._abs, rel_tol=self._rel or 1e-6)

    def __repr__(self):
        return f"~{self._expected}"


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, func in list(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            func()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as exc:
            print(f"  ❌ {name}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
