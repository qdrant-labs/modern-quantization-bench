import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from binquant_bench.configs import Config, SearchVariant, build_configs
from binquant_bench.results import ResultRow, append_rows
from binquant_bench.server import COLLECTION, DiskUsage, Memory, Server
from binquant_bench.vectors import (
    Vectors,
    exact_neighbours,
    load_vectors,
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
    dataset: str
    n_points: int
    n_queries: int
    limits: list[int]
    only: list[str] | None
    image: str
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
class SearchStats:
    recall: float
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class Measurement:
    variant: SearchVariant
    limit: int
    stats: SearchStats


@dataclass(frozen=True)
class Timings:
    upload_s: float
    index_s: float


@dataclass(frozen=True)
class Footprint:
    idle: Memory
    warm: Memory
    disk: DiskUsage


def build_workload(options: RunOptions) -> Workload:
    if options.smoke:
        name, vectors = "smoke", synthetic_vectors(SMOKE_POINTS, SMOKE_DIM)
        n_queries, limits = SMOKE_QUERIES, [10]
    else:
        print(f"loading {options.n_points:,} vectors of {options.dataset}", flush=True)
        name, vectors = options.dataset, load_vectors(options.dataset, options.n_points)
        n_queries, limits = options.n_queries, options.limits
    index, queries = split_queries(vectors, n_queries)
    print("computing exact ground truth", flush=True)
    truth = exact_neighbours(index, queries, max(limits))
    return Workload(name, index, queries.tolist(), truth.tolist(), limits)


async def create_collection(client: AsyncQdrantClient, config: Config, dim: int) -> None:
    # Quantized configs keep the originals cold and only the quantized copy pinned.
    memory = models.Memory.COLD if config.quantization else models.Memory.CACHED
    vectors_config = models.VectorParams(
        size=dim, distance=models.Distance.COSINE, datatype=config.datatype, memory=memory
    )
    await client.create_collection(
        COLLECTION, vectors_config=vectors_config, quantization_config=config.quantization
    )


def upload_vectors(client: QdrantClient, index: Vectors) -> None:
    client.upload_collection(
        COLLECTION,
        vectors=index,
        ids=range(len(index)),
        batch_size=UPLOAD_BATCH_SIZE,
        parallel=UPLOAD_PARALLEL,
    )
    client.close()


async def timed_upload(server: Server, index: Vectors) -> float:
    started = time.monotonic()
    await asyncio.to_thread(upload_vectors, server.sync_client(), index)
    return round(time.monotonic() - started, 1)


async def wait_green(client: AsyncQdrantClient) -> float:
    started, greens = time.monotonic(), 0
    while time.monotonic() - started < GREEN_TIMEOUT_S:
        info = await client.get_collection(COLLECTION)
        greens = greens + 1 if info.status == models.CollectionStatus.GREEN else 0
        if greens >= GREEN_CHECKS:
            return round(time.monotonic() - started, 1)
        await asyncio.sleep(2)
    raise TimeoutError(f"{COLLECTION} is not green after {GREEN_TIMEOUT_S}s")


def point_ids(points: list[models.ScoredPoint]) -> set[int]:
    ids = {point.id for point in points}
    if not all(isinstance(point_id, int) for point_id in ids):
        raise TypeError("expected integer point ids")
    return {point_id for point_id in ids if isinstance(point_id, int)}


async def timed_search(
    client: AsyncQdrantClient, vector: list[float], limit: int, variant: SearchVariant
) -> tuple[set[int], float]:
    started = time.perf_counter()
    response = await client.query_points(
        COLLECTION, query=vector, limit=limit, search_params=variant.params()
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return point_ids(response.points), elapsed_ms


async def measure(
    client: AsyncQdrantClient, workload: Workload, variant: SearchVariant, limit: int
) -> SearchStats:
    recalls: list[float] = []
    latencies: list[float] = []
    for query, truth in zip(workload.queries, workload.truth, strict=True):
        ids, elapsed_ms = await timed_search(client, query, limit, variant)
        recalls.append(len(ids & set(truth[:limit])) / limit)
        latencies.append(elapsed_ms)
    return SearchStats(
        recall=round(float(np.mean(recalls)), 4),
        p50_ms=round(float(np.percentile(latencies, 50)), 2),
        p95_ms=round(float(np.percentile(latencies, 95)), 2),
    )


async def warm_up(client: AsyncQdrantClient, workload: Workload, variant: SearchVariant) -> None:
    for query in workload.queries[:WARMUP_QUERIES]:
        await timed_search(client, query, max(workload.limits), variant)


async def sweep(client: AsyncQdrantClient, config: Config, workload: Workload) -> list[Measurement]:
    measurements: list[Measurement] = []
    for variant in config.variants:
        await warm_up(client, workload, variant)
        for limit in workload.limits:
            stats = await measure(client, workload, variant, limit)
            measurements.append(Measurement(variant, limit, stats))
    return measurements


def to_rows(
    config: Config,
    workload: Workload,
    timings: Timings,
    measurements: list[Measurement],
    footprint: Footprint,
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
            idle_rss_mb=footprint.idle.rss_mb,
            idle_rss_anon_mb=footprint.idle.rss_anon_mb,
            idle_rss_file_mb=footprint.idle.rss_file_mb,
            rss_mb=footprint.warm.rss_mb,
            rss_anon_mb=footprint.warm.rss_anon_mb,
            rss_file_mb=footprint.warm.rss_file_mb,
            segments_mb=footprint.disk.segments_mb,
            wal_mb=footprint.disk.wal_mb,
        )
        for m in measurements
    ]


async def run_config(config: Config, workload: Workload, image: str) -> list[ResultRow]:
    async with Server(image, config.slug) as server:
        await create_collection(server.client, config, workload.index.shape[1])
        upload_s = await timed_upload(server, workload.index)
        index_s = await wait_green(server.client)
        print(f"[{config.name}] uploaded in {upload_s}s, green after {index_s}s more", flush=True)
        idle = await server.memory()
        measurements = await sweep(server.client, config, workload)
        # Read after the sweep, once the working set is warm.
        footprint = Footprint(idle, await server.memory(), await server.disk_usage())
    return to_rows(config, workload, Timings(upload_s, index_s), measurements, footprint)


def select_configs(only: list[str] | None) -> list[Config]:
    return [config for config in build_configs() if not only or config.name in only]


def output_path(options: RunOptions, workload: Workload) -> Path:
    path = options.out or RESULTS_DIR / f"{workload.dataset}-{len(workload.index)}.jsonl"
    if path.exists() and not options.append:
        raise SystemExit(f"{path} exists. Delete it or pass --append.")
    path.parent.mkdir(exist_ok=True)
    return path


async def run_benchmark(options: RunOptions) -> None:
    workload = build_workload(options)
    path = output_path(options, workload)
    for config in select_configs(options.only):
        rows = await run_config(config, workload, options.image)
        append_rows(path, rows)
        print(f"[{config.name}] {len(rows)} rows written", flush=True)
    print(f"done, now: uv run bench report {path}")
