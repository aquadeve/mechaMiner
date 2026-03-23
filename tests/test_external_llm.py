#!/usr/bin/env python3
"""
Tests for the lightweight external LLM adapter and local training flow.
"""

import asyncio
import json
import os
import sys
import tempfile


_root = os.path.join(os.path.dirname(__file__), "..")
_src = os.path.join(_root, "src")
for _d in [
    _root,
    _src,
    os.path.join(_src, "fsot_core"),
    os.path.join(_src, "neuromorphic_brain"),
]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from external_llm import ExternalLLMAdapter, TinyLocalLanguageModel
from fsot_core import FSOT2NeuromorphicIntegration


def test_tiny_local_language_model_trains_saves_and_loads():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(tmpdir, "tiny_model.json")
        model = TinyLocalLanguageModel(model_name="tiny-test")
        stats = model.train(
            [
                {"prompt": "remember the mining session", "response": "I remember the mining session."},
                {"prompt": "plan a training run", "response": "Train locally or in CI with the lightweight checkpoint."},
            ]
        )
        path = model.save_checkpoint(checkpoint)

        restored = TinyLocalLanguageModel(checkpoint_path=path)
        generated = restored.generate("remember the mining session", goal="recall")

        assert stats["sample_count"] == 2
        assert os.path.exists(path)
        assert "I remember the mining session." in generated


def test_external_llm_adapter_generates_with_consciousness_memory_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(tmpdir, "model.json")
        adapter = ExternalLLMAdapter(
            model_name="tiny-consciousness",
            local_model_path=checkpoint,
            memory_base_dir=tmpdir,
            session_id="session_x",
        )
        adapter.train_local(
            samples=[
                {
                    "prompt": "remember the benchmark",
                    "response": "I remember the benchmark and can compress it for future sessions.",
                }
            ],
            output_path=checkpoint,
        )

        result = adapter.generate(
            "Please remember the benchmark because I feel happy about it.",
            goal="archive benchmark memory",
            prefer_external=False,
        )

        assert result["provider_used"] == "local"
        assert result["response"]
        assert os.path.exists(checkpoint)
        assert os.path.exists(os.path.join(tmpdir, "session_x", "thoughts.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "session_x", "memories.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "session_x", "emotions.jsonl"))
        assert result["compressed_memories"]["memory_count"] >= 1


def test_fsot_integration_exposes_external_llm_response():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        config = {
            "external_llm": {
                "enabled": True,
                "provider": "auto",
                "endpoint": None,
                "model_name": "tiny-consciousness",
                "checkpoint_path": os.path.join(tmpdir, "artifacts", "lightweight_external_llm.json"),
                "memory_base_dir": os.path.join(tmpdir, "memory"),
            }
        }
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)

        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            integration = FSOT2NeuromorphicIntegration(config_path=config_path)
            integration.external_llm.train_local(
                samples=[
                    {
                        "prompt": "teach me mining",
                        "response": "I can teach mining concepts with a lightweight locally trained checkpoint.",
                    }
                ],
                output_path=os.path.join(tmpdir, "artifacts", "lightweight_external_llm.json"),
            )
            result = asyncio.run(
                integration.process_integrated_request(
                    "Teach me mining and remember this lesson.",
                    {"goal": "learning", "user": "tester"},
                )
            )
        finally:
            os.chdir(cwd)

        assert "external_llm" in result["fsot_responses"]
        assert result["fsot_responses"]["external_llm"]
        assert result["fsot_responses"]["external_llm_metadata"]["provider"] == "local"
