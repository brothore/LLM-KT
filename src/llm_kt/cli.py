import argparse
import asyncio
import json
from pathlib import Path
from .config import load_config
from .data import import_snapshot
from .protocols import prepare, load_prepared
from .runtime import infer, preview
from .evaluation import evaluate, compare
from .costs import estimate, save_estimate


def main():
    parser = argparse.ArgumentParser(prog="llm-kt")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("import-pykt", help="Copy complete original sequence CSVs into a standalone snapshot")
    p.add_argument("--test", required=True)
    p.add_argument("--train-valid")
    p.add_argument("--dataset", required=True)
    p.add_argument("--representation", choices=["question_native", "single_concept", "concept_expanded"], default="question_native")
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    for name in ("validate", "infer", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--prepared", required=True)
        if name != "validate":
            p.add_argument("--run", required=True)
            p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--prepared", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--profile", choices=["standard_v1", "pykt_legacy_v1"])
    p = sub.add_parser("compare")
    p.add_argument("--prepared", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("estimate", help="Offline full-split inference cost estimate; no API calls or key reads")
    p.add_argument("--config", required=True)
    p.add_argument("--snapshot", help="Override dataset snapshot path for estimation only")
    p.add_argument("--provider", help="Estimate a specific provider/tag; does not change actual API routing")
    p.add_argument("--chars-per-token", type=float, default=3)
    p.add_argument("--chars-per-token-low", type=float, default=2)
    p.add_argument("--chars-per-token-high", type=float, default=4)
    p.add_argument("--output-tokens", type=float, help="Expected total output tokens per request, INCLUDING reasoning")
    p.add_argument("--cache-read-fraction", type=float, default=0)
    p.add_argument("--cache-write-fraction", type=float, default=0)
    p.add_argument("--local-hit-fraction", type=float, default=0)
    p.add_argument("--extra-attempt-fraction", type=float, default=0)
    p.add_argument("--calibration-run")
    p.add_argument("--output", help="Write JSON and Markdown reports")
    args = parser.parse_args()
    try:
        if args.command == "import-pykt":
            result = import_snapshot(args.test, args.output, args.dataset, args.representation, args.train_valid)
        elif args.command == "prepare":
            result = prepare(load_config(args.config), args.output)
        elif args.command == "evaluate":
            result = evaluate(args.prepared, args.run, args.profile)
        elif args.command == "compare":
            if len(args.runs) < 2:
                raise ValueError("compare requires at least two runs")
            result = compare(args.prepared, args.runs, args.output)
        elif args.command == "estimate":
            config = load_config(args.config)
            if args.snapshot:
                config["dataset"]["snapshot"] = str(Path(args.snapshot).resolve())
            result = estimate(config, provider=args.provider, chars_per_token=args.chars_per_token,
                              chars_per_token_low=args.chars_per_token_low, chars_per_token_high=args.chars_per_token_high,
                              output_tokens=args.output_tokens, cache_read_fraction=args.cache_read_fraction,
                              cache_write_fraction=args.cache_write_fraction, local_hit_fraction=args.local_hit_fraction,
                              extra_attempt_fraction=args.extra_attempt_fraction, calibration_run=args.calibration_run)
            if args.output:
                save_estimate(args.output, result)
        else:
            config = load_config(args.config)
            if args.command == "validate" or args.dry_run:
                result = preview(config, args.prepared)
            else:
                result = asyncio.run(infer(config, args.prepared, args.run))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
