# Modern Quantization Benchmarks

A benchmark for Qdrant quantization, run against a Qdrant Cloud cluster. It compares scalar, binary, and TurboQuant quantization, plus the `turbo4` datatype, on pre-embedded datasets from Hugging Face. For each configuration it measures recall against exact search, query latency, and on-disk size.

It is the benchmark behind the [Binary Quantization](https://qdrant.tech/articles/binary-quantization/) article.

## Requirements

- A Qdrant Cloud cluster on Qdrant 1.19 or later, with its URL and an API key. The benchmark uses the `memory` settings and the TurboQuant datatype, which need 1.19. A single-node cluster works best, because telemetry reports one node at a time.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 or later and the dependencies.
- A network connection to Hugging Face for the first run, to stream the vectors. They are cached in `.cache/` afterward.

Give the cluster enough RAM for the `float32` baseline, as a rule of thumb about twice the size of the `float32` vectors (points x dimensions x 4 bytes). If the cluster is smaller, the baseline reads from disk and the results are not comparable.

The benchmark reads memory, size, and hardware usage from the cluster's `/telemetry` endpoint. Hardware reporting is on by default on Qdrant Cloud. A local server has it off. To try the benchmark against one, start it with `QDRANT__SERVICE__HARDWARE_REPORTING=true`.

## Credentials

The benchmark reads the cluster from two environment variables. It does not accept them as arguments.

```bash
export QDRANT_URL="https://xyz-example.eu-central.aws.cloud.qdrant.io:6333"
export QDRANT_API_KEY="..."
```

If you keep them in an env file, `uv run --env-file .env bench ...` loads it.

## Quickstart

```bash
uv sync

# A short run on synthetic data. It checks that your credentials and setup work.
uv run bench run --smoke

# The full run on the ada-002 dataset.
uv run bench run --dataset ada002-1536 --n-points 100000 --out results/ada002-1536.jsonl

# Print the results as a Markdown table.
uv run bench report results/ada002-1536.jsonl --limit 10
```

`run` refuses to overwrite a results file. Delete it, or pass `--append` to add rows to it.

To run a few configurations only, name them:

```bash
uv run bench run --only float32 binary-1bit turbo-bits4 --out results/subset.jsonl
```

Each configuration gets its own collection, named `binquant-<dataset>-<configuration>`, and the benchmark deletes it when the configuration finishes. If a run stops early, `uv run bench cleanup` deletes every collection on the cluster whose name starts with `binquant-`.

Each configuration uploads the full set of vectors again. For 100,000 vectors with 1,536 dimensions that is about 600 MB per configuration, and about 7 GB for a full run. Plan for the upload time, and run from a machine with a fast connection.

## Run on a Kubernetes cluster with kubench

The `Dockerfile` builds an image with `bench` on the `PATH`, so it works as a [kubench](https://github.com/qdrant-labs/kubench) job. Build it once from the project root. `--platform` goes through `docker buildx`, which honors `.dockerignore`. Without it, kubench sends the whole directory, including the multi-gigabyte `.cache/`, as the build context.

```bash
kubench image . --name binquant-bench --registry REGISTRY --platform linux/amd64,linux/arm64
```

Put the cluster credentials in a Kubernetes Secret with the same names as the environment variables:

```bash
kubectl create secret generic qdrant-creds \
  --from-literal=QDRANT_URL=https://xyz-example.cloud.qdrant.io:6333 \
  --from-literal=QDRANT_API_KEY=...
```

Generate a job for each dataset and apply it. Mount shared directories over `/app/.cache`, for the downloaded vectors, and `/app/results`. Set `-b 0`, so a failed job does not retry on top of its own leftover collections.

```bash
kubench job --image REGISTRY/binquant-bench:latest -m binquant-ada002 -b 0 -e qdrant-creds \
  -v /mnt/shared/binquant/cache:/app/.cache -v /mnt/shared/binquant/results:/app/results \
  -o job.yaml -- bench run --dataset ada002-1536
kubectl apply -f job.yaml
```

Jobs that share one Qdrant cluster disturb each other's latency and heap numbers, because those are measured on the whole process. Run one job at a time per cluster, or give each concurrent job its own cluster and its own Secret. If a job fails, run `bench cleanup` the same way to delete its collections.

## Datasets

List the registered datasets with `uv run bench datasets`, and pass one with `--dataset`.

| Name | Hugging Face dataset | Dimensions |
|---|---|---|
| `ada002-1536` | [KShivendu/dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) | 1,536 |
| `3-small-512` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-small-512-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-512-100K) | 512 |
| `3-small-1024` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1024-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1024-100K) | 1,024 |
| `3-small-1536` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K) | 1,536 |
| `3-large-1024` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1024-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1024-100K) | 1,024 |
| `3-large-1536` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) | 1,536 |
| `3-large-3072` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) | 3,072 |
| `arxiv-titles-instructorxl` | [Qdrant/arxiv-titles-instructorxl-embeddings](https://huggingface.co/datasets/Qdrant/arxiv-titles-instructorxl-embeddings) | 768 |
| `arxiv-abstracts-instructorxl` | [Qdrant/arxiv-abstracts-instructorxl-embeddings](https://huggingface.co/datasets/Qdrant/arxiv-abstracts-instructorxl-embeddings) | 768 |
| `bge-m3-ads` | [Qdrant/BGE-m3-1-million-ads](https://huggingface.co/datasets/Qdrant/BGE-m3-1-million-ads) | 1,024 |
| `gte-ads` | [Qdrant/gte-multilingual-ads-1M](https://huggingface.co/datasets/Qdrant/gte-multilingual-ads-1M) | 768 |
| `gte-product-ads` | [Qdrant/gte-multilingual-product-ads-1M](https://huggingface.co/datasets/Qdrant/gte-multilingual-product-ads-1M) | 768 |
| `fineweb-10b` | [Qdrant/FineWeb-10B](https://huggingface.co/datasets/Qdrant/FineWeb-10B) | 768 |
| `coyo-ve` | [Qdrant/Coyo-VE](https://huggingface.co/datasets/Qdrant/Coyo-VE) | 2,048 |
| `wolt-food-clip` | [Qdrant/wolt-food-clip-ViT-B-32-embeddings](https://huggingface.co/datasets/Qdrant/wolt-food-clip-ViT-B-32-embeddings) | 512 |

The embeddings are precomputed, so you do not need a model or an API key. The benchmark reads the first `--n-points` rows, which is 100,000 by default, and fails if the dataset has fewer. It normalizes the vectors and uses cosine distance, whatever distance the dataset was built for.

### Other datasets

Any dataset on the Hub with one dense vector per row works. Pass its repository and the vector column. The dimension comes from the data.

```bash
uv run bench run --repo Qdrant/arxiv-titles-instructorxl-embeddings --column vector \
    --out results/arxiv-titles.jsonl
```

The column must hold a flat list of numbers. Multivector and sparse columns are not supported. Results are named after the repository. Quantization quality depends on the embedding model, and binary quantization works best on high-dimensional, centered vectors, so expect different results on different datasets.

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

- **Collections.** One fresh collection per configuration, with default HNSW settings (`m=16`, `ef_construct=100`) and cosine distance. For quantized configurations the original vectors are `cold` and the quantized vectors are `pinned`.
- **Queries.** `--n-queries` vectors, 100 by default, are picked at random and held out of the index, so a query never finds itself.
- **Ground truth.** Exact cosine search over the indexed vectors with NumPy, independent of Qdrant.
- **Recall.** The share of the exact top-k that a search returns, for each limit in `--limits` (10 and 100 by default).
- **Latency.** One client sends the queries one at a time after 10 warm-up queries, over the REST API. The report shows the median (p50) and the 95th percentile (p95) in two forms: measured by the client, which includes the network, and reported by the server (`server_*`), which does not. Over the internet the network dominates the client figures, so compare configurations with the server figures, or run the benchmark from a machine in the cluster's region.
- **Memory.** `heap_mb` is the live heap that the collection adds to the Qdrant process, read from telemetry after the searches. It is the change from before the collection was created, so it isolates the collection from the rest of the cluster. It counts heap only. Memory-mapped file pages, such as original vectors read back during rescoring, are not in it, which makes it a better measure of the RAM a configuration needs than process memory. Changes of a few MB are noise.
- **Size.** `vectors_mb` is the storage of the collection's original vectors, from telemetry. It does not include the quantized copy or the index. Compute the quantized copy as points x dimensions x bits / 8.
- **Hardware usage.** `cpu_per_query` and `vector_io_read_kb_per_query` come from the cluster's per-collection hardware counters, as the change around each measured batch of queries, divided by the number of queries. The counters use Qdrant's own units, so compare configurations with each other, not with other systems. Vector reads count disk reads of cold vectors, so they are 0 when the vectors sit in RAM.

## Results format

`run` appends one JSON object per configuration, search variant, and limit to a `.jsonl` file:

| Field | Meaning |
|---|---|
| `dataset`, `n_points`, `n_queries`, `dim` | The workload |
| `config`, `notes` | The configuration and its compression |
| `rescore`, `oversampling` | The search variant. Both are `null` for a plain search. |
| `limit`, `recall`, `p50_ms`, `p95_ms` | Recall and client-side latency at that limit |
| `upload_s`, `index_s` | Seconds to upload, and then to reach a green collection |
| `server_p50_ms`, `server_p95_ms` | Latency reported by the server, without the network |
| `heap_mb`, `vectors_mb` | Heap added by the collection, and the storage of its original vectors, in MB |
| `cpu_per_query`, `vector_io_read_kb_per_query` | Hardware counters per query, in Qdrant's units and in KB |

## Caveats

- A single run varies, especially latency. On a shared cluster, differences of a few tenths of a millisecond are noise. Repeat the run before you compare close results.
- Result files from earlier versions of the benchmark, which ran a local server, still print. Columns they do not have show `n/a`.
- Telemetry shows one node. On a multi-node cluster, memory and size cover only the shards on the node that answers.
- The benchmark creates and deletes collections, and it uploads a lot of data. Use a cluster you can spend it on.
- Results depend on hardware and on the Qdrant version. Record both when you publish numbers.
- The benchmark uses the first rows of each dataset, not a random sample.

## Development

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check src
```
