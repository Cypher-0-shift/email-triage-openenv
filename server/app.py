"""
FastAPI server for the Email Triage OpenEnv environment.

Endpoints:
  POST /reset   – start a new episode
  POST /step    – take a triage action
  GET  /state   – get current episode state
  GET  /health  – liveness probe
"""

import os
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from models import TriageAction, TriageObservation, TriageState
    from server.email_triage_environment import EmailTriageEnvironment
except ImportError:
    from email_triage_env.models import TriageAction, TriageObservation, TriageState
    from email_triage_env.server.email_triage_environment import EmailTriageEnvironment


# ─────────────────────────────────────────────
# Request / Response wrappers
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = "basic_triage"


class StepRequest(BaseModel):
    action: TriageAction


class StepResponse(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ─────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────

def create_app(max_concurrent_envs: int = 1) -> FastAPI:
    app = FastAPI(
        title="Email Triage OpenEnv",
        description="An RL environment for email triage with 3 tasks of increasing difficulty.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Single-session environment (one episode at a time)
    env = EmailTriageEnvironment()

    @app.get("/health")
    async def health():
        return {"status": "healthy", "env": "email_triage_env"}

    @app.post("/reset", response_model=TriageObservation)
    async def reset(req: ResetRequest = ResetRequest()):
        try:
            obs = env.reset(task_name=req.task_name)
            return obs
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/step", response_model=StepResponse)
    async def step(req: StepRequest):
        try:
            obs, reward, done, info = env.step(req.action)
            return StepResponse(observation=obs, reward=reward, done=done, info=info)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/state", response_model=TriageState)
    async def state():
        return env.state()

    @app.get("/tasks")
    async def list_tasks():
        return {"tasks": EmailTriageEnvironment.VALID_TASKS}

    @app.get("/metadata")
    async def metadata():
        return {
            "name": "email_triage_env",
            "description": "A real-world email triage environment where agents classify, reply to, flag, and delegate emails."
        }

    @app.get("/schema")
    async def schema():
        return {
            "action": TriageAction.model_json_schema(),
            "observation": TriageObservation.model_json_schema(),
            "state": TriageState.model_json_schema()
        }

    @app.post("/mcp")
    async def mcp(request: Request):
        try:
            body = await request.json()
            req_id = body.get("id")
        except Exception:
            req_id = None
            
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"status": "ok"}
        }

    return app


def main():
    import uvicorn
    uvicorn.run("email_triage_env.server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()