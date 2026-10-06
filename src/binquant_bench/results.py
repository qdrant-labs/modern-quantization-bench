import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultRow:
    dataset: str
    n_points: int
    n_queries: int
    dim: int
    config: str
    notes: str
    rescore: bool | None
    oversampling: float | None
    limit: int
    recall: float
    p50_ms: float
    p95_ms: float
    upload_s: float
    index_s: float
    idle_rss_mb: float
    idle_rss_anon_mb: float
    idle_rss_file_mb: float
    rss_mb: float
    rss_anon_mb: float
    rss_file_mb: float
    segments_mb: float
    wal_mb: float


def append_rows(path: Path, rows: list[ResultRow]) -> None:
    with path.open("a") as out:
        out.writelines(json.dumps(asdict(row)) + "\n" for row in rows)


def read_rows(path: Path) -> list[ResultRow]:
    return [ResultRow(**json.loads(line)) for line in path.read_text().splitlines()]
