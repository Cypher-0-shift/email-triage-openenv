"""
Client for the Email Triage OpenEnv environment.

Wraps HTTP calls to the FastAPI server in a clean Python API.
Supports both async (default) and sync (via .sync()) usage patterns
consistent with the OpenEnv spec.
"""

import asyncio
import os
from typing import Optional, Tuple

import httpx

try:
    from models import TriageAction, TriageObservation, TriageState
except ImportError:
    from email_triage_env.models import TriageAction, TriageObservation, TriageState


class _SyncEmailTriageEnv:
    """Synchronous wrapper — use EmailTriageEnv(...).sync()"""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def reset(self, task_name: str = "basic_triage") -> TriageObservation:
        r = self._client.post(f"{self._base_url}/reset", json={"task_name": task_name})
        r.raise_for_status()
        return TriageObservation(**r.json())

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, dict]:
        r = self._client.post(
            f"{self._base_url}/step",
            json={"action": action.model_dump()},
        )
        r.raise_for_status()
        data = r.json()
        return (
            TriageObservation(**data["observation"]),
            data["reward"],
            data["done"],
            data.get("info", {}),
        )

    def state(self) -> TriageState:
        r = self._client.get(f"{self._base_url}/state")
        r.raise_for_status()
        return TriageState(**r.json())

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class EmailTriageEnv:
    """
    Async-first client for the Email Triage environment.

    Usage (async):
        async with EmailTriageEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset("basic_triage")
            obs, reward, done, info = await env.step(TriageAction(label="spam"))

    Usage (sync):
        with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset("basic_triage")
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def sync(self) -> _SyncEmailTriageEnv:
        return _SyncEmailTriageEnv(self._base_url, self._timeout)

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def reset(self, task_name: str = "basic_triage") -> TriageObservation:
        assert self._client, "Use 'async with EmailTriageEnv(...) as env:'"
        r = await self._client.post(
            f"{self._base_url}/reset", json={"task_name": task_name}
        )
        r.raise_for_status()
        return TriageObservation(**r.json())

    async def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, dict]:
        assert self._client, "Use 'async with EmailTriageEnv(...) as env:'"
        r = await self._client.post(
            f"{self._base_url}/step",
            json={"action": action.model_dump()},
        )
        r.raise_for_status()
        data = r.json()
        return (
            TriageObservation(**data["observation"]),
            data["reward"],
            data["done"],
            data.get("info", {}),
        )

    async def state(self) -> TriageState:
        assert self._client, "Use 'async with EmailTriageEnv(...) as env:'"
        r = await self._client.get(f"{self._base_url}/state")
        r.raise_for_status()
        return TriageState(**r.json())

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Docker-launch helper (mirrors OpenEnv convention) ──

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 8000, **kwargs) -> "EmailTriageEnv":
        """
        Start a Docker container from `image_name` and return a connected client.
        Requires Docker to be running locally.
        """
        import subprocess, time
        container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", f"{port}:8000", image_name],
            text=True,
        ).strip()

        base_url = f"http://localhost:{port}"
        client = cls(base_url=base_url, **kwargs)
        client._container_id = container_id

        # Wait for server to be ready
        for _ in range(30):
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{base_url}/health", timeout=2.0)
                    if r.status_code == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(1.0)

        return client