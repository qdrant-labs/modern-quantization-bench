from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from binquant_bench.retry import with_retry

MIB = 2**20
KIB = 2**10
# Level 3 has everything read here. Level 4 adds search statistics that the client cannot parse.
DETAILS_LEVEL = 3


@dataclass(frozen=True)
class Usage:
    cpu: int
    vector_io_read: int


@with_retry
async def fetch(client: AsyncQdrantClient) -> models.TelemetryData:
    response = await client.http.service_api.telemetry(
        details_level=DETAILS_LEVEL, per_collection=True
    )
    if response.result is None:
        raise RuntimeError("telemetry returned no result")
    return response.result


def heap_bytes(data: models.TelemetryData) -> int:
    """Live heap of the whole Qdrant process. It excludes memory-mapped file pages."""
    if data.memory is None:
        raise RuntimeError("telemetry has no memory section. Turn on hardware reporting.")
    return data.memory.allocated_bytes


def vectors_bytes(data: models.TelemetryData, collection: str) -> int:
    """Storage of the original vectors, not of the quantized copy or the index."""
    if data.collections is None:
        raise RuntimeError("telemetry has no collections section")
    return sum(
        shard.local.vectors_size_bytes or 0
        for found in data.collections.collections or []
        if isinstance(found, models.CollectionTelemetry) and found.id == collection
        for shard in found.shards or []
        if shard.local is not None
    )


def usage(data: models.TelemetryData, collection: str) -> Usage:
    """Cumulative hardware counters of one collection, in Qdrant's own units."""
    if data.hardware is None or collection not in data.hardware.collection_data:
        raise RuntimeError("telemetry has no hardware data. Turn on hardware reporting.")
    counters = data.hardware.collection_data[collection]
    return Usage(cpu=counters.cpu, vector_io_read=counters.vector_io_read)
