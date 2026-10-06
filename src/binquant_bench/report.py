from pathlib import Path

from binquant_bench.results import ResultRow, read_rows

COLUMNS = (
    "Config | Rescore | Oversampling | Recall | p50 ms | p95 ms | "
    "RSS idle MB | RSS warm MB | Segments MB"
)


def rescore_label(rescore: bool | None) -> str:
    if rescore is None:
        return "n/a"
    return "yes" if rescore else "no"


def oversampling_label(oversampling: float | None) -> str:
    return "n/a" if oversampling is None else f"{oversampling:g}x"


def format_row(row: ResultRow) -> str:
    cells = (
        row.config,
        rescore_label(row.rescore),
        oversampling_label(row.oversampling),
        f"{row.recall:.4f}",
        str(row.p50_ms),
        str(row.p95_ms),
        str(row.idle_rss_mb),
        str(row.rss_mb),
        str(row.segments_mb),
    )
    return "| " + " | ".join(cells) + " |"


def format_report(rows: list[ResultRow], limit: int) -> str:
    selected = [row for row in rows if row.limit == limit]
    first = selected[0]
    title = (
        f"{first.dataset}, {first.n_points:,} points x {first.dim} dims, "
        f"{first.n_queries} held-out queries, recall@{limit} vs exact search\n"
    )
    header = [f"| {COLUMNS} |", "|" + "---|" * len(COLUMNS.split("|"))]
    return "\n".join([title, *header, *map(format_row, selected)])


def print_report(path: Path, limit: int) -> None:
    print(format_report(read_rows(path), limit))
