import argparse
import asyncio
from pathlib import Path

from binquant_bench.report import print_report
from binquant_bench.run import RunOptions, run_benchmark
from binquant_bench.vectors import DATASETS

DEFAULT_IMAGE = "qdrant/qdrant:v1.19"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="Quantization benchmark for Qdrant")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the benchmark")
    run.add_argument("--dataset", choices=DATASETS, default="ada002-1536")
    run.add_argument("--n-points", type=int, default=100_000, help="first N rows of the dataset")
    run.add_argument("--n-queries", type=int, default=100, help="held out of the index")
    run.add_argument("--limits", type=int, nargs="+", default=[10, 100])
    run.add_argument("--only", nargs="+", help="config names to run, default all")
    run.add_argument("--image", default=DEFAULT_IMAGE)
    run.add_argument("--append", action="store_true", help="add to an existing results file")
    run.add_argument("--out", type=Path, help="results file, default results/<dataset>-<N>.jsonl")
    run.add_argument("--smoke", action="store_true", help="tiny synthetic run to check the setup")

    report = commands.add_parser("report", help="print a Markdown table from a results file")
    report.add_argument("file", type=Path)
    report.add_argument("--limit", type=int, default=10)
    return parser


def run_options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        dataset=args.dataset,
        n_points=args.n_points,
        n_queries=args.n_queries,
        limits=args.limits,
        only=args.only,
        image=args.image,
        append=args.append,
        smoke=args.smoke,
        out=args.out,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        asyncio.run(run_benchmark(run_options(args)))
    else:
        print_report(args.file, args.limit)
