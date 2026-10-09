import itertools
from collections.abc import Iterator
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


DATASETS: dict[str, DatasetSpec] = {
    "ada002-1536": DatasetSpec("KShivendu/dbpedia-entities-openai-1M", "openai"),
    "3-small-1536": DatasetSpec(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K",
        "text-embedding-3-small-1536-embedding",
    ),
    "3-large-1536": DatasetSpec(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        "text-embedding-3-large-1536-embedding",
    ),
    "3-large-3072": DatasetSpec(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
        "text-embedding-3-large-3072-embedding",
    ),
    "3-small-512": DatasetSpec(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-512-100K",
        "text-embedding-3-small-512-embedding",
    ),
    "3-small-1024": DatasetSpec(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1024-100K",
        "text-embedding-3-small-1024-embedding",
    ),
    "3-large-1024": DatasetSpec(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1024-100K",
        "text-embedding-3-large-1024-embedding",
    ),
    "arxiv-titles-instructorxl": DatasetSpec(
        "Qdrant/arxiv-titles-instructorxl-embeddings", "vector"
    ),
    "arxiv-abstracts-instructorxl": DatasetSpec(
        "Qdrant/arxiv-abstracts-instructorxl-embeddings", "vector"
    ),
    "bge-m3-ads": DatasetSpec("Qdrant/BGE-m3-1-million-ads", "bgem3_dense_vecs"),
    "gte-ads": DatasetSpec("Qdrant/gte-multilingual-ads-1M", "gte"),
    "gte-product-ads": DatasetSpec("Qdrant/gte-multilingual-product-ads-1M", "gte"),
    "fineweb-10b": DatasetSpec("Qdrant/FineWeb-10B", "dense_embedding"),
    "coyo-ve": DatasetSpec("Qdrant/Coyo-VE", "qwen_multimodal_embedding"),
    "wolt-food-clip": DatasetSpec("Qdrant/wolt-food-clip-ViT-B-32-embeddings", "vector"),
}


def resolve_dataset(
    name: str | None, repo: str | None, column: str | None
) -> tuple[str, DatasetSpec]:
    """A registered name, or any Hugging Face dataset given as a repo and a vector column."""
    if repo and column:
        return repo.split("/")[-1].lower(), DatasetSpec(repo, column)
    if name:
        return name, DATASETS[name]
    raise ValueError("pass a dataset name, or a repo and a column")


def normalize(vectors: Vectors) -> Vectors:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def stream_column(spec: DatasetSpec) -> Iterator[NDArray[np.float32]]:
    rows = load_dataset(spec.repo, split="train", streaming=True).select_columns([spec.column])
    return (np.asarray(row[spec.column], dtype=np.float32) for row in rows)


def stream_vectors(spec: DatasetSpec, n_points: int) -> Vectors:
    stream = stream_column(spec)
    first = next(stream)
    vectors = np.empty((n_points, len(first)), dtype=np.float32)
    count = 0
    for count, vector in enumerate(itertools.islice(itertools.chain([first], stream), n_points), 1):
        vectors[count - 1] = vector
        if count % PROGRESS_EVERY == 0:
            print(f"  read {count:,}/{n_points:,} vectors", flush=True)
    if count < n_points:
        raise ValueError(f"{spec.repo} has only {count:,} rows, {n_points:,} requested")
    return normalize(vectors)


def load_vectors(name: str, spec: DatasetSpec, n_points: int) -> Vectors:
    cached = CACHE_DIR / f"{name}-{n_points}.npy"
    if cached.exists():
        return np.load(cached)
    CACHE_DIR.mkdir(exist_ok=True)
    vectors = stream_vectors(spec, n_points)
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
