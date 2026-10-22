"""Classify RL seeds offline and in Lean, with the same admission path as training.

Reads at most --max-scan-rows source rows, replays candidates in one Mathlib
session until --max-admitted pass or --time-budget runs out, and writes a new
report into --report-dir (an existing report is never overwritten).
"""

import argparse
import asyncio
from pathlib import Path

from pantograph_async import PantographEnv

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DATASET_NAME, DATASET_REVISION
from maths_ai.gnn_inference.atp_lean_gnn.seed_admission import (
    AdmissionReport,
    ValidationCache,
    dataset_candidates,
    environment_identity,
    file_candidates,
    preflight,
    write_report,
)
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client


async def run(args) -> AdmissionReport:
    env = PantographEnv(
        source_root=args.source_root,
        pantograph_repl=args.pantograph_repl,
        imports=("Mathlib",),
        timeout=args.request_timeout,
        server_start_timeout_s=args.start_timeout,
    )
    env.verify()
    report = AdmissionReport()
    report.environment = environment_identity(env)
    if args.theorem_file is not None:
        candidates = file_candidates(
            args.theorem_file,
            max_scan_rows=args.max_scan_rows,
            max_state_chars=args.max_state_chars,
            report=report,
        )
    else:
        candidates = dataset_candidates(
            dataset_name=args.dataset_name,
            split=args.split,
            revision=args.revision,
            max_scan_rows=args.max_scan_rows,
            max_state_chars=args.max_state_chars,
            report=report,
        )
    await preflight(
        candidates,
        lambda: create_model_sexpr_client(env),
        max_admitted=args.max_admitted,
        time_budget_s=args.time_budget,
        report=report,
        cache=ValidationCache(args.cache, report.environment),
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pantograph-repl", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--theorem-file", type=Path)
    source.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scan-rows", type=int, default=1000)
    parser.add_argument("--max-admitted", type=int, default=1000)
    parser.add_argument("--max-state-chars", type=int, default=None)
    parser.add_argument("--time-budget", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--start-timeout", type=float, default=600.0)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Validated-replay cache; reused only for the same environment identity.",
    )
    args = parser.parse_args(argv)
    report = asyncio.run(run(args))
    print(report.summary())
    print(f"Report: {write_report(report, args.report_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
