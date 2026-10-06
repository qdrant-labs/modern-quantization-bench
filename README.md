# Modern Quantization Benchmarks

A benchmark for Qdrant quantization. It compares scalar, binary, and TurboQuant quantization, plus the `turbo4` datatype, on OpenAI embeddings of DBpedia entities. For each configuration it measures recall against exact search, query latency, and on-disk size.

It is the benchmark behind the [Binary Quantization](https://qdrant.tech/articles/binary-quantization/) article.

## Requirements

- Docker, running. The benchmark starts one Qdrant container per configuration.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 or later and the dependencies.
- Qdrant 1.19 or later. The default image is `qdrant/qdrant:v1.19`.
- A network connection for the first run, to stream the vectors from Hugging Face. They are cached in `.cache/` afterward.

## Quickstart

```bash
uv sync

# A short run on synthetic data. It checks that your setup works.
uv run bench run --smoke

# The full run on the ada-002 dataset. It takes about 10 to 15 minutes.
uv run bench run --dataset ada002-1536 --n-points 100000 --out results/ada002-1536.jsonl

# Print the results as a Markdown table.
uv run bench report results/ada002-1536.jsonl --limit 10
```

`run` refuses to overwrite a results file. Delete it, or pass `--append` to add rows to it.

To run a few configurations only, name them:

```bash
uv run bench run --only float32 binary-1bit turbo-bits4 --out results/subset.jsonl
```

If a run stops early, remove its containers with `docker rm -f $(docker ps -aq --filter name=binquant-)`.

## Datasets

| Name | Hugging Face dataset | Dimensions |
|---|---|---|
| `ada002-1536` | [KShivendu/dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) | 1,536 |
| `3-small-1536` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K) | 1,536 |
| `3-large-1536` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) | 1,536 |
| `3-large-3072` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) | 3,072 |

The embeddings are precomputed, so you do not need an OpenAI key. The benchmark reads the first `--n-points` rows, which is 100,000 by default, and normalizes them.

## Configurations

| Name | Compression |
|---|---|
| `float32` | 1x, the baseline |
| `scalar-int8` | 4x |
| `binary-1bit` | 32x |
| `binary-2bit` | 16x |
| `binary-1_5bit` | about 21x |
| `binary-1bit+query-scalar8` | 32x, with an 8-bit scalar query |
| `turbo-bits4` | 8x |
| `turbo-bits2` | 16x |
| `turbo-bits1_5` | about 21x |
| `turbo-bits1` | 32x |
| `turbo4-datatype` | 8x, with no full-precision copy |
| `turbo4-datatype+turbo-bits1` | 1-bit TurboQuant, rescored against `turbo4` vectors |

Quantized configurations run every combination of rescoring (off, on) and oversampling (1x, 2x, 3x, 4x). `float32` and `turbo4-datatype` run one plain search.

## Method

- **Collections.** One fresh container per configuration, with default HNSW settings (`m=16`, `ef_construct=100`) and cosine distance. For quantized configurations the original vectors are `cold` and the quantized vectors are `pinned`.
- **Queries.** `--n-queries` vectors, 100 by default, are picked at random and held out of the index, so a query never finds itself.
- **Ground truth.** Exact cosine search over the indexed vectors with NumPy, independent of Qdrant.
- **Recall.** The share of the exact top-k that a search returns, for each limit in `--limits` (10 and 100 by default).
- **Latency.** One client sends the queries one at a time after 10 warm-up queries. The report shows the median (p50) and the 95th percentile (p95).
- **Disk.** `segments_mb` is the size of the collection's segments, measured as blocks in use, because Qdrant preallocates sparse files. `wal_mb` is the write-ahead log.
- **Memory.** The results include resident memory of the Qdrant process, right after indexing (`idle`) and after the searches (`warm`), split into heap (`anon`) and file-backed pages (`file`). File-backed pages include the operating system's page cache, which fills with original vectors during rescoring. Use them with care, and do not read them as the RAM a configuration needs. For sizing, combine the computed size of the quantized copy (points x dimensions x bits / 8) with `segments_mb`.

## Results format

`run` appends one JSON object per configuration, search variant, and limit to a `.jsonl` file:

| Field | Meaning |
|---|---|
| `dataset`, `n_points`, `n_queries`, `dim` | The workload |
| `config`, `notes` | The configuration and its compression |
| `rescore`, `oversampling` | The search variant. Both are `null` for a plain search. |
| `limit`, `recall`, `p50_ms`, `p95_ms` | Recall and latency at that limit |
| `upload_s`, `index_s` | Seconds to upload, and then to reach a green collection |
| `idle_rss_*`, `rss_*` | Process memory in MB, after indexing and after the searches |
| `segments_mb`, `wal_mb` | Disk usage in MB |

## Caveats

- A single run varies, especially latency. On a shared machine, differences of a few tenths of a millisecond are noise. Repeat the run before you compare close results.
- Results depend on hardware and on the Qdrant version. Record both when you publish numbers.
- The benchmark uses the first rows of each dataset, not a random sample.

## Development

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check src
```
