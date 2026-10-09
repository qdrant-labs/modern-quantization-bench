import argparse
import asyncio
from pathlib import Path

from binquant_bench.cloud import COLLECTION_PREFIX, CloudSettings, delete_bench_collections
from binquant_bench.report import print_report
from binquant_bench.run import RunOptions, run_benchmark
from binquant_bench.vectors import DATASETS

DEFAULT_DATASET = "ada002-1536"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Quantization benchmark for Qdrant Cloud. Set QDRANT_URL and QDRANT_API_KEY.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the benchmark")
    run.add_argument("--dataset", choices=DATASETS, help=f"default {DEFAULT_DATASET}")
    run.add_argument("--repo", help="any Hugging Face dataset with a dense vector column")
    run.add_argument("--column", help="the vector column of --repo")
    run.add_argument("--n-points", type=int, default=100_000, help="first N rows of the dataset")
    run.add_argument("--n-queries", type=int, default=100, help="held out of the index")
    run.add_argument("--limits", type=int, nargs="+", default=[10, 100])
    run.add_argument("--only", nargs="+", help="config names to run, default all")
    run.add_argument("--append", action="store_true", help="add to an existing results file")
    run.add_argument("--out", type=Path, help="results file, default results/<dataset>-<N>.jsonl")
    run.add_argument("--smoke", action="store_true", help="tiny synthetic run to check the setup")

    commands.add_parser("datasets", help="list the registered datasets")
    commands.add_parser(
        "cleanup", help=f"delete leftover collections named {COLLECTION_PREFIX}* from a failed run"
    )

    report = commands.add_parser("report", help="print a Markdown table from a results file")
    report.add_argument("file", type=Path)
    report.add_argument("--limit", type=int, default=10)
    return parser


def run_options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        dataset=args.dataset or (None if args.repo else DEFAULT_DATASET),
        repo=args.repo,
        column=args.column,
        n_points=args.n_points,
        n_queries=args.n_queries,
        limits=args.limits,
        only=args.only,
        append=args.append,
        smoke=args.smoke,
        out=args.out,
    )


def print_datasets() -> None:
    for name, spec in DATASETS.items():
        print(f"{name:28s} {spec.repo} [{spec.column}]")


def check_dataset_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if bool(args.repo) != bool(args.column):
        parser.error("--repo and --column go together")
    if args.repo and args.dataset:
        parser.error("use either --dataset or --repo with --column")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    match args.command:
        case "run":
            check_dataset_args(parser, args)
            asyncio.run(run_benchmark(run_options(args)))
        case "datasets":
            print_datasets()
        case "cleanup":
            deleted = asyncio.run(delete_bench_collections(CloudSettings.from_env()))
            print(f"deleted {len(deleted)} collections", *deleted, sep="\n")
        case _:
            print_report(args.file, args.limit)
