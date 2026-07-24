from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .corpora import prepare_corpora
from .runner import consolidate, plan, run_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the sharded Japanese-English production speech corpus."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get(
                "S2ST_CORPUS_CONFIG",
                "/workspace/configs/production-qwen17b.yaml",
            )
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare",
        help="Download, filter, de-duplicate, and select JESC/KFTT.",
    )
    commands.add_parser(
        "build",
        help="Run the complete resumable pipeline from downloads to release.",
    )
    commands.add_parser("plan", help="Validate the input manifest and estimate storage.")
    shard = commands.add_parser("run-shard", help="Synthesize and QC one stable shard.")
    shard.add_argument("--shard-index", type=int, required=True)
    merge = commands.add_parser("consolidate", help="Build release manifests.")
    merge.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        result = prepare_corpora(config)
    elif args.command == "build":
        preparation = prepare_corpora(config)
        planning = plan(config)
        if not planning["within_pair_limit"]:
            raise RuntimeError(
                "prepared manifest exceeds run.max_pairs_per_shard; "
                "increase run.num_shards"
            )
        if not planning["within_audio_budget"]:
            raise RuntimeError(
                "configured corpus exceeds run.maximum_output_gib"
            )
        for shard_index in range(config.run.num_shards):
            print(
                f"[build] shard {shard_index + 1}/{config.run.num_shards}",
                flush=True,
            )
            run_shard(config, shard_index)
        release = consolidate(config)
        result = {
            "preparation": preparation,
            "plan": planning,
            "release": release,
        }
    elif args.command == "plan":
        result = plan(config)
    elif args.command == "run-shard":
        result = {"manifest": str(run_shard(config, args.shard_index))}
    else:
        result = consolidate(config, allow_incomplete=args.allow_incomplete)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
