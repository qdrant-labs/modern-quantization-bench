import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from grpc.aio import AioRpcError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from binquant_bench.cloud import CloudSettings, collection_name
from binquant_bench.configs import Config, SearchVariant, build_configs
from binquant_bench.results import ResultRow, append_rows
from binquant_bench.retry import is_already_exists, with_retry
from binquant_bench.telemetry import KIB, MIB, fetch, heap_bytes, usage, vectors_bytes
from binquant_bench.vectors import (
    Vectors,
    exact_neighbours,
    load_vectors,
    resolve_dataset,
    split_queries,
    synthetic_vectors,
)

RESULTS_DIR = Path("results")
UPLOAD_BATCH_SIZE = 256
UPLOAD_PARALLEL = 2
WARMUP_QUERIES = 10
GREEN_CHECKS = 3
GREEN_TIMEOUT_S = 3600
SMOKE_POINTS, SMOKE_DIM, SMOKE_QUERIES = 3000, 128, 20


@dataclass(frozen=True)
class RunOptions:
    dataset: str | None
    repo: str | None
    column: str | None
    n_points: int
    n_queries: int
    limits: list[int]
    only: list[str] | None
    append: bool
    smoke: bool
    out: Path | None


@dataclass(frozen=True)
class Workload:
    dataset: str
    index: Vectors
    queries: list[list[float]]
    truth: list[list[int]]
    limits: list[int]


@dataclass(frozen=True)
class Target:
    client: AsyncQdrantClient
    collection: str


@dataclass(frozen=True)
class Search:
    ids: set[int]
    client_ms: float
    server_ms: float


@dataclass(frozen=True)
class SearchStats:
    recall: float
    p50_ms: float
    p95_ms: float
    server_p50_ms: float
    server_p95_ms: float


@dataclass(frozen=True)
class QueryCost:
    cpu: float
    vector_io_read_kb: float


@dataclass(frozen=True)
class Measurement:
    variant: SearchVariant
    limit: int
    stats: SearchStats
    cost: QueryCost


@dataclass(frozen=True)
class Footprint:
    heap_mb: float
    vectors_mb: float


@dataclass(frozen=True)
class Timings:
    upload_s: float
    index_s: float


def build_workload(options: RunOptions) -> Workload:
    if options.smoke:
        name, vectors = "smoke", synthetic_vectors(SMOKE_POINTS, SMOKE_DIM)
        n_queries, limits = SMOKE_QUERIES, [10]
    else:
        name, spec = resolve_dataset(options.dataset, options.repo, options.column)
        print(f"loading {options.n_points:,} vectors of {spec.repo}", flush=True)
        vectors = load_vectors(name, spec, options.n_points)
        n_queries, limits = options.n_queries, options.limits
    index, queries = split_queries(vectors, n_queries)
    print("computing exact ground truth", flush=True)
    truth = exact_neighbours(index, queries, max(limits))
    return Workload(name, index, queries.tolist(), truth.tolist(), limits)


@with_retry
async def create_collection(target: Target, config: Config, dim: int) -> None:
    # Quantized configs keep the originals cold and only the quantized copy pinned.
    memory = models.Memory.COLD if config.quantization else models.Memory.CACHED
    vectors_config = models.VectorParams(
        size=dim, distance=models.Distance.COSINE, datatype=config.datatype, memory=memory
    )
    try:
        await target.client.create_collection(
            target.collection,
            vectors_config=vectors_config,
            quantization_config=config.quantization,
        )
    except (UnexpectedResponse, AioRpcError) as error:
        # After a retry, "already exists" means an earlier attempt succeeded but its reply was lost.
        attempt = create_collection.statistics["attempt_number"]  # ty: ignore[unresolved-attribute]
        if is_already_exists(error) and attempt > 1:
            return
        raise


@with_retry
def upload_vectors(settings: CloudSettings, collection: str, index: Vectors) -> None:
    client = settings.sync_client()
    try:
        client.upload_collection(
            collection,
            vectors=index,
            ids=range(len(index)),
            batch_size=UPLOAD_BATCH_SIZE,
            parallel=UPLOAD_PARALLEL,
        )
    finally:
        client.close()


async def timed_upload(settings: CloudSettings, collection: str, index: Vectors) -> float:
    started = time.monotonic()
    await asyncio.to_thread(upload_vectors, settings, collection, index)
    return round(time.monotonic() - started, 1)


@with_retry
async def collection_status(target: Target) -> models.CollectionStatus:
    info = await target.client.get_collection(target.collection)
    return info.status


@with_retry
async def drop_collection(target: Target) -> None:
    await target.client.delete_collection(target.collection)


async def wait_green(target: Target) -> float:
    started, greens = time.monotonic(), 0
    while time.monotonic() - started < GREEN_TIMEOUT_S:
        status = await collection_status(target)
        greens = greens + 1 if status == models.CollectionStatus.GREEN else 0
        if greens >= GREEN_CHECKS:
            return round(time.monotonic() - started, 1)
        await asyncio.sleep(2)
    raise TimeoutError(f"{target.collection} is not green after {GREEN_TIMEOUT_S}s")


def point_ids(points: list[models.ScoredPoint]) -> set[int]:
    ids = {point.id for point in points}
    if not all(isinstance(point_id, int) for point_id in ids):
        raise TypeError("expected integer point ids")
    return {point_id for point_id in ids if isinstance(point_id, int)}


@with_retry
async def timed_search(
    target: Target, vector: list[float], limit: int, variant: SearchVariant
) -> Search:
    """Uses the REST API, because only its response reports the server-side time.

    Each retry restarts the timer, so only the successful attempt is measured.
    """
    request = models.QueryRequest(query=vector, limit=limit, params=variant.params())
    started = time.perf_counter()
    response = await target.client.http.search_api.query_points(
        collection_name=target.collection, query_request=request
    )
    client_ms = (time.perf_counter() - started) * 1000
    if response.result is None or response.time is None:
        raise RuntimeError(f"incomplete query response from {target.collection}")
    return Search(point_ids(response.result.points), client_ms, response.time * 1000)


def percentiles(values: list[float]) -> tuple[float, float]:
    return round(float(np.percentile(values, 50)), 2), round(float(np.percentile(values, 95)), 2)


async def measure(
    target: Target, workload: Workload, variant: SearchVariant, limit: int
) -> tuple[SearchStats, QueryCost]:
    before = usage(await fetch(target.client), target.collection)
    recalls: list[float] = []
    searches: list[Search] = []
    for query, truth in zip(workload.queries, workload.truth, strict=True):
        search = await timed_search(target, query, limit, variant)
        recalls.append(len(search.ids & set(truth[:limit])) / limit)
        searches.append(search)
    after = usage(await fetch(target.client), target.collection)
    p50, p95 = percentiles([s.client_ms for s in searches])
    server_p50, server_p95 = percentiles([s.server_ms for s in searches])
    stats = SearchStats(round(float(np.mean(recalls)), 4), p50, p95, server_p50, server_p95)
    n_queries = len(workload.queries)
    cost = QueryCost(
        cpu=round((after.cpu - before.cpu) / n_queries, 1),
        vector_io_read_kb=round(
            (after.vector_io_read - before.vector_io_read) / KIB / n_queries, 2
        ),
    )
    return stats, cost


async def warm_up(target: Target, workload: Workload, variant: SearchVariant) -> None:
    for query in workload.queries[:WARMUP_QUERIES]:
        await timed_search(target, query, max(workload.limits), variant)


async def sweep(target: Target, config: Config, workload: Workload) -> list[Measurement]:
    measurements: list[Measurement] = []
    for variant in config.variants:
        await warm_up(target, workload, variant)
        for limit in workload.limits:
            stats, cost = await measure(target, workload, variant, limit)
            measurements.append(Measurement(variant, limit, stats, cost))
    return measurements


def to_rows(
    config: Config,
    workload: Workload,
    timings: Timings,
    footprint: Footprint,
    measurements: list[Measurement],
) -> list[ResultRow]:
    return [
        ResultRow(
            dataset=workload.dataset,
            n_points=len(workload.index),
            n_queries=len(workload.queries),
            dim=workload.index.shape[1],
            config=config.name,
            notes=config.notes,
            rescore=m.variant.rescore,
            oversampling=m.variant.oversampling,
            limit=m.limit,
            recall=m.stats.recall,
            p50_ms=m.stats.p50_ms,
            p95_ms=m.stats.p95_ms,
            upload_s=timings.upload_s,
            index_s=timings.index_s,
            server_p50_ms=m.stats.server_p50_ms,
            server_p95_ms=m.stats.server_p95_ms,
            heap_mb=footprint.heap_mb,
            vectors_mb=footprint.vectors_mb,
            cpu_per_query=m.cost.cpu,
            vector_io_read_kb_per_query=m.cost.vector_io_read_kb,
        )
        for m in measurements
    ]


async def measure_footprint(target: Target, heap_before: int) -> Footprint:
    """Heap added by the collection, and the storage of its original vectors."""
    data = await fetch(target.client)
    return Footprint(
        heap_mb=round((heap_bytes(data) - heap_before) / MIB, 1),
        vectors_mb=round(vectors_bytes(data, target.collection) / MIB, 1),
    )


async def run_config(
    config: Config, workload: Workload, client: AsyncQdrantClient, settings: CloudSettings
) -> list[ResultRow]:
    target = Target(client, collection_name(workload.dataset, config.slug))
    heap_before = heap_bytes(await fetch(client))
    await create_collection(target, config, workload.index.shape[1])
    try:
        upload_s = await timed_upload(settings, target.collection, workload.index)
        index_s = await wait_green(target)
        print(f"[{config.name}] uploaded in {upload_s}s, green after {index_s}s more", flush=True)
        measurements = await sweep(target, config, workload)
        footprint = await measure_footprint(target, heap_before)
    finally:
        await drop_collection(target)
    return to_rows(config, workload, Timings(upload_s, index_s), footprint, measurements)


def select_configs(only: list[str] | None) -> list[Config]:
    return [config for config in build_configs() if not only or config.name in only]


def output_path(options: RunOptions, workload: Workload) -> Path:
    path = options.out or RESULTS_DIR / f"{workload.dataset}-{len(workload.index)}.jsonl"
    if path.exists() and not options.append:
        raise SystemExit(f"{path} exists. Delete it or pass --append.")
    path.parent.mkdir(exist_ok=True)
    return path


async def run_benchmark(options: RunOptions) -> None:
    settings = CloudSettings.from_env()
    workload = build_workload(options)
    path = output_path(options, workload)
    client = settings.async_client()
    try:
        for config in select_configs(options.only):
            rows = await run_config(config, workload, client, settings)
            append_rows(path, rows)
            print(f"[{config.name}] {len(rows)} rows written", flush=True)
    finally:
        await client.close()
    print(f"done, now: uv run bench report {path}")
