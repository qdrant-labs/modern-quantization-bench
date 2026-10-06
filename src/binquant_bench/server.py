import asyncio
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import httpx
from qdrant_client import AsyncQdrantClient, QdrantClient

HOST = "127.0.0.1"
COLLECTION = "bench"
READY_TIMEOUT_S = 60
CLIENT_TIMEOUT_S = 600
STORAGE = f"/qdrant/storage/collections/{COLLECTION}"
PROC_FIELDS = ("VmRSS", "RssAnon", "RssFile")


@dataclass(frozen=True)
class Memory:
    rss_mb: float
    rss_anon_mb: float
    rss_file_mb: float


@dataclass(frozen=True)
class DiskUsage:
    segments_mb: float
    wal_mb: float


async def docker(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {stderr.decode().strip()}")
    return stdout.decode().strip()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def read_proc_status(pid: int) -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in PROC_FIELDS:
            fields[key] = int(value.split()[0])
    return fields


def kb_to_mb(kb: int) -> float:
    return round(kb / 1024, 1)


class Server:
    """A throwaway Qdrant container with an async client."""

    def __init__(self, image: str, label: str) -> None:
        self.image = image
        self.rest_port = free_port()
        self.grpc_port = free_port()
        self.name = f"binquant-{label}-{self.rest_port}"
        self.client = AsyncQdrantClient(
            host=HOST,
            port=self.rest_port,
            grpc_port=self.grpc_port,
            prefer_grpc=True,
            timeout=CLIENT_TIMEOUT_S,
        )

    async def __aenter__(self) -> Self:
        await docker(
            "run", "-d", "--rm", "--name", self.name,
            "-p", f"{HOST}:{self.rest_port}:6333", "-p", f"{HOST}:{self.grpc_port}:6334",
            self.image,
        )  # fmt: skip
        await self.wait_ready()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.client.close()
        await docker("rm", "-f", self.name)

    def sync_client(self) -> QdrantClient:
        return QdrantClient(
            host=HOST,
            port=self.rest_port,
            grpc_port=self.grpc_port,
            prefer_grpc=True,
            timeout=CLIENT_TIMEOUT_S,
        )

    async def wait_ready(self) -> None:
        async with httpx.AsyncClient() as http:
            for _ in range(READY_TIMEOUT_S):
                if await self.is_up(http):
                    return
                await asyncio.sleep(1)
        raise TimeoutError(f"{self.name} did not start in {READY_TIMEOUT_S}s")

    async def is_up(self, http: httpx.AsyncClient) -> bool:
        try:
            await http.get(f"http://{HOST}:{self.rest_port}/")
        except httpx.TransportError:
            return False
        return True

    async def qdrant_pid(self) -> int:
        """Docker reports `bash entrypoint.sh` as the container process. Qdrant is its child."""
        main_pid = int(await docker("inspect", "-f", "{{.State.Pid}}", self.name))
        children = Path(f"/proc/{main_pid}/task/{main_pid}/children").read_text().split()
        return next(
            int(pid) for pid in children if b"qdrant" in Path(f"/proc/{pid}/cmdline").read_bytes()
        )

    async def memory(self) -> Memory:
        """RssFile counts resident pages of memory-mapped files, RssAnon is heap."""
        status = read_proc_status(await self.qdrant_pid())
        return Memory(
            rss_mb=kb_to_mb(status["VmRSS"]),
            rss_anon_mb=kb_to_mb(status["RssAnon"]),
            rss_file_mb=kb_to_mb(status["RssFile"]),
        )

    async def disk_usage(self) -> DiskUsage:
        return DiskUsage(
            segments_mb=await self.used_mb(f"{STORAGE}/*/segments"),
            wal_mb=await self.used_mb(f"{STORAGE}/*/wal"),
        )

    async def used_mb(self, pattern: str) -> float:
        """Blocks in use, not apparent size, because Qdrant preallocates sparse files."""
        out = await docker("exec", self.name, "sh", "-c", f"du -skc {pattern} | tail -1")
        return kb_to_mb(int(out.split()[0]))
