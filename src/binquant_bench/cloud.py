import os
from dataclasses import dataclass
from typing import Self

from qdrant_client import AsyncQdrantClient, QdrantClient

from binquant_bench.retry import with_retry

COLLECTION_PREFIX = "binquant-"
CLIENT_TIMEOUT_S = 600
ENV_VARS = ("QDRANT_URL", "QDRANT_API_KEY")


@dataclass(frozen=True)
class CloudSettings:
    url: str
    api_key: str

    @classmethod
    def from_env(cls) -> Self:
        missing = [name for name in ENV_VARS if name not in os.environ]
        if missing:
            raise SystemExit(f"Set {' and '.join(missing)} in the environment.")
        return cls(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

    def async_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url=self.url, api_key=self.api_key, prefer_grpc=True, timeout=CLIENT_TIMEOUT_S
        )

    def sync_client(self) -> QdrantClient:
        return QdrantClient(
            url=self.url, api_key=self.api_key, prefer_grpc=True, timeout=CLIENT_TIMEOUT_S
        )


def collection_name(dataset: str, config_slug: str) -> str:
    return f"{COLLECTION_PREFIX}{dataset}-{config_slug}"


@with_retry
async def bench_collections(client: AsyncQdrantClient) -> list[str]:
    response = await client.get_collections()
    return [c.name for c in response.collections if c.name.startswith(COLLECTION_PREFIX)]


@with_retry
async def delete_collection(client: AsyncQdrantClient, name: str) -> None:
    await client.delete_collection(name)


async def delete_bench_collections(settings: CloudSettings) -> list[str]:
    client = settings.async_client()
    try:
        names = await bench_collections(client)
        for name in names:
            await delete_collection(client, name)
    finally:
        await client.close()
    return names
