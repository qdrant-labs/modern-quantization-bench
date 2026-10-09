import json
from dataclasses import asdict, dataclass, fields
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
    server_p50_ms: float | None = None
    server_p95_ms: float | None = None
    heap_mb: float | None = None
    vectors_mb: float | None = None
    cpu_per_query: float | None = None
    vector_io_read_kb_per_query: float | None = None


FIELD_NAMES = {field.name for field in fields(ResultRow)}


def append_rows(path: Path, rows: list[ResultRow]) -> None:
    with path.open("a") as out:
        out.writelines(json.dumps(asdict(row)) + "\n" for row in rows)


def read_rows(path: Path) -> list[ResultRow]:
    """Read result files from any version. Fields the current schema dropped are ignored."""
    records = (json.loads(line) for line in path.read_text().splitlines())
    return [
        ResultRow(**{k: v for k, v in record.items() if k in FIELD_NAMES}) for record in records
    ]
