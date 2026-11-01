from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .corpora import prepare_corpora
from .runner import consolidate, plan, recheck_shard, run_shard
from .smoke import select_smoke_pairs, smoke_app_config
from .external import is_external, validate_external, ensure_identity, verify_inputs, reset_external_cache
from .input_index import reset_indexes, index_for_config


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
    smoke = commands.add_parser(
        "smoke-test",
        help="Run the production TTS and Whisper pipeline on five corpus pairs.",
    )
    smoke.add_argument("--pairs", type=int)
    smoke.add_argument("--benchmark-seed", type=int, default=42)
    performance = commands.add_parser("benchmark-performance", help="Isolated TTS-worker / Whisper-batch tuning; never starts production.")
    performance.add_argument("--stage", choices=["tts", "asr", "all"], default="all")
    performance.add_argument("--local-dir", type=Path)
    performance.add_argument("--report-dir", type=Path)
    performance.add_argument("--workers", nargs="+", type=int)
    performance.add_argument("--asr-batches", nargs="+", type=int)
    commands.add_parser("preflight", help="Check storage and GPU without loading models.")
    commands.add_parser("preload", help="Load pinned Kokoro and G2P, then exit.")
    voices = commands.add_parser("voice-samples", help="Audition voices using fixed sentences outside production.")
    voices.add_argument("--output-dir", type=Path, required=True)
    voices.add_argument("--en-voices", nargs="+", default=["af_heart", "af_bella", "am_adam"])
    parser.add_argument("--allow-production", action="store_true", help="Explicitly enable external-profile production, after reviewing smoke results.")
    commands.add_parser("plan", help="Validate the input manifest and estimate storage.")
    recheck = commands.add_parser("recheck-smoke", help="Recheck saved smoke audio without TTS; back up old reports.")
    recheck.add_argument("--pairs", type=int)
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
    from .io import reset_budgets
    reset_budgets()
    reset_external_cache()
    reset_indexes()
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command in {"smoke-test", "recheck-smoke"}:
        count = args.pairs if args.pairs is not None else config.smoke.pair_count
        if not 1 <= count <= 1000:
            raise ValueError("smoke --pairs must be between 1 and 1000")
        if is_external(config) or args.pairs is not None:
            root = config.smoke.output_dir / f"pairs-{count}"
            config = replace(config, smoke=replace(config.smoke, pair_count=count,
                input_jsonl=root / "input.jsonl", output_dir=root / "production"))
    if is_external(config) and args.command in {"build", "run-shard", "run-next-shard", "run-remaining-shards"} and not args.allow_production:
        raise RuntimeError("Review 10/100-pair smoke reports first; production requires --allow-production before the command")
    if args.command == "benchmark-performance":
        from .performance import benchmark_performance
        result = benchmark_performance(config, args.stage, args.local_dir, args.report_dir, args.workers, args.asr_batches)
    elif args.command == "preflight":
        from .runner import gpu_preflight
        result = gpu_preflight(config)
    elif args.command == "preload":
        if config.tts.backend != "kokoro":
            raise ValueError("preload command is for Kokoro")
        from .kokoro_backend import KokoroBackend
        backend = KokoroBackend(config.tts)
        result = {"model": config.tts.model_id, "revision": config.tts.revision,
                  "device": backend.device, "load_seconds": backend.load_seconds}
    elif args.command == "voice-samples":
        if config.tts.backend != "kokoro":
            raise ValueError("voice-samples command is for Kokoro")
        if args.output_dir.resolve().is_relative_to(config.run.output_dir.resolve()):
            raise ValueError("voice samples must be outside production output_dir")
        from .kokoro_backend import voice_samples
        result = voice_samples(config, args.output_dir, args.en_voices)
    elif args.command == "prepare":
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
            external=is_external(config),
            benchmark_seed=args.benchmark_seed,
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
        provenance = None
        if is_external(config):
            validate_external(config)
            if (config.run.output_dir / "provenance.json").exists():
                provenance = ensure_identity(config)
        completed = []
        remaining = []
        for index in range(config.run.num_shards):
            name = f"shard-{index:05d}-of-{config.run.num_shards:05d}.jsonl"
            path = config.run.output_dir / "manifests" / "qc" / name
            if path.is_file() and is_external(config):
                if provenance is None:
                    raise RuntimeError("completed output has no provenance")
                # Status checks metadata; run_shard/consolidate additionally verify every WAV.
                from .manifest import read_jsonl
                rows = list(read_jsonl(path))
                expected_ids = index_for_config(config).pair_ids(index, config.run.max_pairs_per_shard)
                if len(rows) != len(expected_ids) or {r["pair_id"] for r in rows} != expected_ids or any(r.get("provenance") != provenance for r in rows):
                    raise RuntimeError(f"invalid completed shard: {path}")
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
    if is_external(config) and args.command in {"run-shard", "run-next-shard"}:
        verify_inputs(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
