"""
AEGIS Security Environment — FastAPI Server
OpenEnv-compatible server wrapping the AEGIS cascade.
"""

import sys
sys.path.insert(0, "..")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aegis_env import AEGISSecurityEnv, AEGISAction
from models import AEGISObservation, AEGISState, ThreatCategory

app = FastAPI(
    title="AEGIS Security Environment",
    description="OpenEnv RL Environment — 42-layer adversarial defense cascade",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (per-session in production)
env = AEGISSecurityEnv(max_steps=100)


class ResetResponse(BaseModel):
    state: AEGISState


class StepRequest(BaseModel):
    prompt: str
    expected_category: str = "BENIGN"
    stellschrauben_adjustments: dict[str, float] | None = None


class StepResponse(BaseModel):
    observation: AEGISObservation
    state: AEGISState


@app.post("/reset", response_model=ResetResponse)
async def reset():
    state = env.reset()
    return ResetResponse(state=state)


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    action = AEGISAction(
        prompt=req.prompt,
        expected_category=ThreatCategory(req.expected_category),
        stellschrauben_adjustments=req.stellschrauben_adjustments,
    )
    obs = env.step(action)
    state = env.state()
    return StepResponse(observation=obs, state=state)


@app.get("/state", response_model=AEGISState)
async def get_state():
    return env.state()


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "AEGIS Security", "layers": 42}
