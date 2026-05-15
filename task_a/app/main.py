from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.simulator import simulate_review
from core.persona_builder import (
    get_user_persona,
    get_business_context,
    get_random_user_id,
    get_random_business_id,
    df_users,
    df_biz
)

app = FastAPI(
    title="BCT LLM Challenge - Task A",
    description="User Modeling and Review Simulation Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ReviewRequest(BaseModel):
    user_id: Optional[str] = None
    business_id: Optional[str] = None
    nigerian_mode: Optional[bool] = True

class PersonaRequest(BaseModel):
    user_id: str

class BusinessRequest(BaseModel):
    business_id: str

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/random-ids")
def random_ids():
    return {
        "user_id":     get_random_user_id(),
        "business_id": get_random_business_id()
    }

@app.post("/simulate-review")
def simulate(req: ReviewRequest):
    user_id     = req.user_id     or get_random_user_id()
    business_id = req.business_id or get_random_business_id()
    result = simulate_review(user_id, business_id, nigerian_mode=req.nigerian_mode)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/get-persona")
def get_persona(req: PersonaRequest):
    persona = get_user_persona(req.user_id)
    if "error" in persona:
        raise HTTPException(status_code=404, detail=persona["error"])
    return persona

@app.post("/get-business")
def get_business(req: BusinessRequest):
    biz = get_business_context(req.business_id)
    if "error" in biz:
        raise HTTPException(status_code=404, detail=biz["error"])
    return biz

@app.get("/stats")
def stats():
    return {
        "total_users":      len(df_users),
        "total_businesses": len(df_biz),
        "model":            "llama-3.1-8b-instant via Groq",
        "nigerian_mode":    "enabled by default"
    }
