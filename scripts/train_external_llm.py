#!/usr/bin/env python3
"""
Train the lightweight external LLM checkpoint locally or in GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
FSOT_DIR = SRC_DIR / "fsot_core"
NEURO_DIR = SRC_DIR / "neuromorphic_brain"

for candidate in (str(REPO_ROOT), str(SRC_DIR), str(FSOT_DIR), str(NEURO_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from external_llm import ExternalLLMAdapter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the lightweight external LLM checkpoint.")
    parser.add_argument("--dataset", help="Optional JSON or JSONL file containing prompt/response pairs.")
    parser.add_argument(
        "--output",
        default="artifacts/lightweight_external_llm.json",
        help="Checkpoint path to write after training.",
    )
    parser.add_argument(
        "--memory-dir",
        default="artifacts/lightweight_external_llm_memory",
        help="Directory used for consciousness-backed session memory files during training.",
    )
    parser.add_argument(
        "--session-id",
        default="ci_training_session",
        help="Session id used for the consciousness-backed trainer.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain how to reuse stored memories for future generations.",
        help="Optional smoke-test prompt run after training.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    adapter = ExternalLLMAdapter(
        model_name="tiny-consciousness",
        local_model_path=args.output,
        memory_base_dir=args.memory_dir,
        session_id=args.session_id,
    )
    stats = adapter.train_local(dataset_path=args.dataset, output_path=args.output)
    sample = adapter.generate(args.prompt, goal="validate lightweight training", prefer_external=False)

    output = {
        "training": stats,
        "sample_generation": {
            "response": sample["response"],
            "provider_used": sample["provider_used"],
            "session_id": sample["session_id"],
        },
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
