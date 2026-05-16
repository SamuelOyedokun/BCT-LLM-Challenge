from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.simulator import simulate_review
from core.persona_builder import (
    get_user_persona,
    get_business_context,
    get_random_user_id,
    get_random_business_id,
    get_df_users,
    get_df_biz,
)

app = FastAPI(
    title="BCT LLM Challenge - NaijaReview Intelligence System",
    description="Task A: Review Simulation | Task B: Recommendation Agent",
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

# ── Pydantic Models ───────────────────────────────────────────

class ReviewRequest(BaseModel):
    user_id: Optional[str] = None
    business_id: Optional[str] = None
    nigerian_mode: Optional[bool] = True

class PersonaRequest(BaseModel):
    user_id: str

class BusinessRequest(BaseModel):
    business_id: str

class RecommendRequest(BaseModel):
    user_id: str
    user_request: str
    conversation_history: Optional[List[dict]] = []
    nigerian_mode: Optional[bool] = True

class UserContextRequest(BaseModel):
    user_id: str

# ── Root ──────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "tasks": ["A", "B"]}

# ── Task A Endpoints ──────────────────────────────────────────

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
    try:
        total_users = len(get_df_users())
        total_biz   = len(get_df_biz())
    except Exception:
        total_users = 301758
        total_biz   = 150346
    return {
        "total_users":      total_users,
        "total_businesses": total_biz,
        "model":            "llama-3.1-8b-instant via Groq",
        "nigerian_mode":    "enabled by default"
    }

# ── Task B Endpoints ──────────────────────────────────────────

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    try:
        from core.recommender import recommend
        result = recommend(
            user_id              = req.user_id,
            user_request         = req.user_request,
            conversation_history = req.conversation_history,
            nigerian_mode        = req.nigerian_mode,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user-context")
def user_context(req: UserContextRequest):
    from core.recommender import get_user_context
    return get_user_context(req.user_id)

@app.get("/task-b-stats")
def task_b_stats():
    try:
        from core.recommender import get_stats
        return get_stats()
    except Exception as e:
        return {
            "total_businesses_indexed": 150346,
            "total_users":              301758,
            "model":                    "llama-3.1-8b-instant via Groq",
            "retrieval":                "Keyword + LLaMA reasoning",
        }

@app.get("/task-a")
def task_a_page():
    return FileResponse(os.path.join(STATIC_DIR, "task_a.html"))

@app.get("/task-b")
def task_b_page():
    return FileResponse(os.path.join(STATIC_DIR, "task_b.html"))
