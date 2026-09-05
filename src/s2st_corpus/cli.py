from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_config
from .corpora import prepare_corpora
from .runner import consolidate, plan, recheck_shard, run_shard
from .smoke import select_smoke_pairs, smoke_app_config


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
    commands.add_parser(
        "smoke-test",
        help="Run the production TTS and Whisper pipeline on five corpus pairs.",
    )
    commands.add_parser("plan", help="Validate the input manifest and estimate storage.")
    commands.add_parser("recheck-smoke", help="Recheck saved smoke audio without TTS; back up old reports.")
    shard = commands.add_parser("run-shard", help="Synthesize and QC one stable shard.")
    shard.add_argument("--shard-index", type=int, required=True)
    commands.add_parser(
        "run-next-shard",
        help="Resume the first shard without a completed QC manifest.",
    )
    commands.add_parser("status", help="Report completed and remaining shards.")
    commands.add_parser("run-remaining-shards", help="Run all unfinished shards sequentially, then consolidate.")
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
    elif args.command == "smoke-test":
        preparation = prepare_corpora(config)
        selected = select_smoke_pairs(
            config.run.input_jsonl,
            config.smoke.input_jsonl,
            config.smoke.pair_count,
        )
        smoke_config = smoke_app_config(config)
        planning = plan(smoke_config)
        run_shard(smoke_config, 0)
        release = consolidate(smoke_config)
        result = {
            "preparation": preparation,
            "selection": {
                "pairs": len(selected),
                "pair_ids": [pair["pair_id"] for pair in selected],
                "input_jsonl": str(config.smoke.input_jsonl),
            },
            "plan": planning,
            "release": {
                **release,
                "all_pairs_accepted": (
                    release["accepted_pairs"] == config.smoke.pair_count
                ),
            },
        }
    elif args.command == "recheck-smoke":
        smoke_config = smoke_app_config(config)
        recheck_shard(smoke_config, 0)
        result = consolidate(smoke_config)
        result["all_pairs_accepted"] = result["accepted_pairs"] == config.smoke.pair_count
    elif args.command == "plan":
        result = plan(config)
    elif args.command == "run-shard":
        result = {"manifest": str(run_shard(config, args.shard_index))}
    elif args.command in {"run-next-shard", "run-remaining-shards", "status"}:
        completed = []
        remaining = []
        for index in range(config.run.num_shards):
            name = f"shard-{index:05d}-of-{config.run.num_shards:05d}.jsonl"
            path = config.run.output_dir / "manifests" / "qc" / name
            (completed if path.is_file() else remaining).append(index)
        result = {
            "completed_shards": len(completed),
            "remaining_shards": len(remaining),
            "next_shard": remaining[0] if remaining else None,
            "total_shards": config.run.num_shards,
        }
        if args.command == "run-next-shard" and remaining:
            result["manifest"] = str(run_shard(config, remaining[0]))
            completed.append(remaining.pop(0))
        elif args.command == "run-remaining-shards":
            for index in remaining.copy():
                print(f"[run-all] shard={index} completed={len(completed)}/{config.run.num_shards}", flush=True)
                result["manifest"] = str(run_shard(config, index))
                completed.append(index)
                remaining.remove(index)
                print(f"[run-all] completed={len(completed)}/{config.run.num_shards} remaining={len(remaining)}", flush=True)
            result["release"] = consolidate(config)
        result.update(completed_shards=len(completed), remaining_shards=len(remaining),
                      next_shard=remaining[0] if remaining else None)
    else:
        result = consolidate(config, allow_incomplete=args.allow_incomplete)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
