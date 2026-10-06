import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset
from numpy.typing import NDArray

type Vectors = NDArray[np.float32]
type Neighbours = NDArray[np.int64]

CACHE_DIR = Path(".cache")
SEED = 42
SCORE_CHUNK = 50_000
PROGRESS_EVERY = 10_000


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    column: str
    dim: int


DATASETS: dict[str, DatasetSpec] = {
    "ada002-1536": DatasetSpec("KShivendu/dbpedia-entities-openai-1M", "openai", 1536),
    "3-small-1536": DatasetSpec(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K",
        "text-embedding-3-small-1536-embedding",
        1536,
    ),
    "3-large-1536": DatasetSpec(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        "text-embedding-3-large-1536-embedding",
        1536,
    ),
    "3-large-3072": DatasetSpec(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
        "text-embedding-3-large-3072-embedding",
        3072,
    ),
}


def normalize(vectors: Vectors) -> Vectors:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def stream_vectors(spec: DatasetSpec, n_points: int) -> Vectors:
    rows = load_dataset(spec.repo, split="train", streaming=True).select_columns([spec.column])
    vectors = np.empty((n_points, spec.dim), dtype=np.float32)
    for i, row in enumerate(itertools.islice(rows, n_points), start=1):
        vectors[i - 1] = row[spec.column]
        if i % PROGRESS_EVERY == 0:
            print(f"  read {i:,}/{n_points:,} vectors", flush=True)
    return normalize(vectors)


def load_vectors(dataset: str, n_points: int) -> Vectors:
    cached = CACHE_DIR / f"{dataset}-{n_points}.npy"
    if cached.exists():
        return np.load(cached)
    CACHE_DIR.mkdir(exist_ok=True)
    vectors = stream_vectors(DATASETS[dataset], n_points)
    np.save(cached, vectors)
    return vectors


def synthetic_vectors(n_points: int, dim: int) -> Vectors:
    rng = np.random.default_rng(SEED)
    return normalize(rng.standard_normal((n_points, dim)).astype(np.float32))


def split_queries(vectors: Vectors, n_queries: int) -> tuple[Vectors, Vectors]:
    """Hold the queries out of the index, so a query never finds itself."""
    rng = np.random.default_rng(SEED)
    picked = rng.choice(len(vectors), size=n_queries, replace=False)
    keep = np.ones(len(vectors), dtype=bool)
    keep[picked] = False
    return vectors[keep], vectors[picked]


def exact_neighbours(index: Vectors, queries: Vectors, k: int) -> Neighbours:
    """Exact cosine top-k. The vectors are normalized, so a dot product is enough."""
    scores = np.empty((len(queries), len(index)), dtype=np.float32)
    for start in range(0, len(index), SCORE_CHUNK):
        scores[:, start : start + SCORE_CHUNK] = queries @ index[start : start + SCORE_CHUNK].T
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)
