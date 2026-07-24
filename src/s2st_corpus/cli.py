from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .runner import consolidate, plan, run_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the sharded Japanese-English production speech corpus."
    )
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="Validate the input manifest and estimate storage.")
    shard = commands.add_parser("run-shard", help="Synthesize and QC one stable shard.")
    shard.add_argument("--shard-index", type=int, required=True)
    merge = commands.add_parser("consolidate", help="Build release manifests.")
    merge.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "plan":
        result = plan(config)
    elif args.command == "run-shard":
        result = {"manifest": str(run_shard(config, args.shard_index))}
    else:
        result = consolidate(config, allow_incomplete=args.allow_incomplete)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
