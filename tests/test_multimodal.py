#!/usr/bin/env python3
"""
Tests for the ConsciousnessMiner class (without real network connections).
Uses a mock socket to validate the miner's internal logic, including all
SHA-256 Stratum protocol handlers.
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
    StratumFramer,
    calculate_merkle_root,
    difficulty_to_target,
    target_to_bits,
    handle_server_request,
    DIFF1_TARGET,
    MINER_USER_AGENT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_USERNAME,
)
from core_system import ConsciousnessState, evaluate_hash_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_miner() -> ConsciousnessMiner:
    return ConsciousnessMiner(
        host="localhost",
        port=9999,
        username="test_user",
        password="x",
    )


def _setup_miner_with_sock() -> ConsciousnessMiner:
    miner = _make_miner()
    miner.sock = MagicMock()
    miner.extranonce1 = "00000001"
    miner.extranonce2_size = 4
    return miner


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


def test_miner_initial_difficulty():
    miner = _make_miner()
    assert miner.difficulty == 1.0
    assert miner.share_target == difficulty_to_target(1.0)


def test_miner_reconnect_delay_stored():
    miner = ConsciousnessMiner(reconnect_delay=15.0)
    assert miner.reconnect_delay == 15.0


# ---------------------------------------------------------------------------
# Difficulty / target helpers
# ---------------------------------------------------------------------------


def test_difficulty_to_target_diff1():
    t = difficulty_to_target(1.0)
    assert t == DIFF1_TARGET


def test_difficulty_to_target_diff2():
    t2 = difficulty_to_target(2.0)
    t1 = difficulty_to_target(1.0)
    assert t2 == t1 // 2


def test_difficulty_to_target_zero_clamp():
    # Zero or negative difficulty should not cause division errors.
    t = difficulty_to_target(0)
    assert t == difficulty_to_target(1.0)


def test_difficulty_to_target_large():
    t = difficulty_to_target(1_000_000)
    assert t < DIFF1_TARGET
    assert t >= 0


def test_target_to_bits_known():
    # A target that requires at least 32 leading zero bits
    target = 2 ** (256 - 32) - 1
    bits = target_to_bits(target)
    assert bits >= 32


def test_target_to_bits_zero():
    # target = 0 → 256 leading zeros required (impossible hash)
    assert target_to_bits(0) == 256


def test_target_to_bits_max():
    # target = 2^255 → bit_length = 256, so leading zeros = 256 - 256 = 0
    bits = target_to_bits(2 ** 255)
    assert bits == 0


# ---------------------------------------------------------------------------
# StratumFramer tests
# ---------------------------------------------------------------------------


def test_framer_single_message():
    f = StratumFramer()
    lines = f.feed('{"id":1}\n')
    assert lines == ['{"id":1}']


def test_framer_multiple_messages_one_chunk():
    f = StratumFramer()
    lines = f.feed('{"a":1}\n{"b":2}\n')
    assert len(lines) == 2
    assert lines[0] == '{"a":1}'
    assert lines[1] == '{"b":2}'


def test_framer_partial_message():
    f = StratumFramer()
    lines1 = f.feed('{"id":1')
    assert lines1 == []
    lines2 = f.feed('}\n')
    assert lines2 == ['{"id":1}']


def test_framer_empty_lines_skipped():
    f = StratumFramer()
    lines = f.feed('\n\n{"x":1}\n')
    assert lines == ['{"x":1}']


def test_framer_reset():
    f = StratumFramer()
    f.feed('partial')
    f.reset()
    lines = f.feed('{"y":1}\n')
    assert lines == ['{"y":1}']


# ---------------------------------------------------------------------------
# handle_server_request tests
# ---------------------------------------------------------------------------


def test_handle_ping():
    sock = MagicMock()
    msg = {"id": 42, "method": "mining.ping", "params": []}
    handled = handle_server_request(sock, msg)
    assert handled is True
    sock.sendall.assert_called_once()
    sent = json.loads(sock.sendall.call_args[0][0].decode().strip())
    assert sent["id"] == 42
    assert sent["result"] is True


def test_handle_get_version():
    sock = MagicMock()
    msg = {"id": 7, "method": "client.get_version", "params": []}
    handled = handle_server_request(sock, msg)
    assert handled is True
    sent = json.loads(sock.sendall.call_args[0][0].decode().strip())
    assert sent["id"] == 7
    assert MINER_USER_AGENT in sent["result"]


def test_handle_unknown_method_returns_false():
    sock = MagicMock()
    msg = {"id": 1, "method": "mining.notify", "params": []}
    handled = handle_server_request(sock, msg)
    assert handled is False


def test_handle_reconnect_not_consumed():
    sock = MagicMock()
    msg = {"id": None, "method": "client.reconnect", "params": []}
    handled = handle_server_request(sock, msg)
    assert handled is False  # main loop should see it


# ---------------------------------------------------------------------------
# _on_set_difficulty tests
# ---------------------------------------------------------------------------


def test_set_difficulty_updates_target():
    miner = _make_miner()
    miner._on_set_difficulty([16384.0])
    assert miner.difficulty == 16384.0
    assert miner.share_target == difficulty_to_target(16384.0)


def test_set_difficulty_empty_params_noop():
    miner = _make_miner()
    before = miner.difficulty
    miner._on_set_difficulty([])
    assert miner.difficulty == before


def test_set_difficulty_bad_value_noop():
    miner = _make_miner()
    before = miner.difficulty
    miner._on_set_difficulty(["not_a_number"])
    assert miner.difficulty == before


def test_set_difficulty_updates_share_target():
    miner = _make_miner()
    miner._on_set_difficulty([1.0])
    t1 = miner.share_target
    miner._on_set_difficulty([2.0])
    t2 = miner.share_target
    assert t2 < t1  # harder difficulty → lower target


# ---------------------------------------------------------------------------
# _on_set_extranonce tests
# ---------------------------------------------------------------------------


def test_set_extranonce_updates_fields():
    miner = _make_miner()
    miner._on_set_extranonce(["deadbeef", 4])
    assert miner.extranonce1 == "deadbeef"
    assert miner.extranonce2_size == 4


def test_set_extranonce_resets_counter():
    miner = _make_miner()
    miner.extranonce2_counter = 999
    miner._on_set_extranonce(["aabb", 2])
    assert miner.extranonce2_counter == 0


def test_set_extranonce_bad_params_noop():
    miner = _make_miner()
    before_en1 = miner.extranonce1
    miner._on_set_extranonce([])  # too short
    assert miner.extranonce1 == before_en1


def test_set_extranonce_bad_size_noop():
    miner = _make_miner()
    before = miner.extranonce2_size
    miner._on_set_extranonce(["aabb", "not_an_int"])
    assert miner.extranonce2_size == before


# ---------------------------------------------------------------------------
# calculate_merkle_root tests
# ---------------------------------------------------------------------------


def test_merkle_root_empty_branch():
    coinbase_hash = hashlib.sha256(b"coinbase").digest()
    root = calculate_merkle_root(coinbase_hash, [])
    expected = coinbase_hash[::-1]
    assert root == expected


def test_merkle_root_single_branch():
    coinbase_hash = hashlib.sha256(b"cb").digest()
    branch_hash = hashlib.sha256(b"branch").hexdigest()
    root = calculate_merkle_root(coinbase_hash, [branch_hash])
    assert len(root) == 32


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
    coinb1 = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff"
    coinb2 = "00000000"
    merkle_branch: List[str] = []
    job_id = "test_job_001"
    return job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime


def test_mine_job_increments_hashes():
    miner = _setup_miner_with_sock()
    before = miner.total_hashes
    miner.mine_job(*_minimal_job_params())
    assert miner.total_hashes > before


def test_mine_job_updates_consciousness():
    miner = _setup_miner_with_sock()
    before_step = miner.consciousness._step
    miner.mine_job(*_minimal_job_params())
    assert miner.consciousness._step > before_step


def test_mine_job_increments_extranonce2_counter():
    miner = _setup_miner_with_sock()
    before = miner.extranonce2_counter
    miner.mine_job(*_minimal_job_params())
    assert miner.extranonce2_counter == before + 1


def test_mine_job_memory_grows():
    miner = _setup_miner_with_sock()
    miner.mine_job(*_minimal_job_params())
    assert len(miner.thought_engine.memory) > 0


def test_mine_job_returns_true():
    miner = _setup_miner_with_sock()
    result = miner.mine_job(*_minimal_job_params())
    assert result is True


def test_mine_job_uses_share_target():
    """mine_job should use self.share_target, not a hard-coded constant."""
    miner = _setup_miner_with_sock()
    # Set an impossibly easy target (all hashes should pass) to ensure submit path runs.
    miner.share_target = 2 ** 256 - 1  # every hash will be < this
    # Reset submitted count
    miner.shares_submitted = 0
    miner.mine_job(*_minimal_job_params())
    # Every nonce candidate should trigger submit_share
    assert miner.shares_submitted > 0


def test_mine_job_difficulty_integrated():
    """After set_difficulty, mine_job uses the new target."""
    miner = _setup_miner_with_sock()
    miner._on_set_difficulty([1024.0])
    assert miner.share_target == difficulty_to_target(1024.0)
    # Just ensure it runs without error with the new target
    miner.mine_job(*_minimal_job_params())


# ---------------------------------------------------------------------------
# print_status smoke test
# ---------------------------------------------------------------------------


def test_print_status_no_crash(capsys):
    miner = _setup_miner_with_sock()
    miner.mine_job(*_minimal_job_params())
    miner.print_status()
    captured = capsys.readouterr()
    assert "Hashes" in captured.out


def test_print_status_shows_difficulty(capsys):
    miner = _setup_miner_with_sock()
    miner._on_set_difficulty([512.0])
    miner.mine_job(*_minimal_job_params())
    miner.print_status()
    captured = capsys.readouterr()
    assert "512" in captured.out


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
# Standalone runner
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

