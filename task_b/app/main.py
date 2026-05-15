main_b_fixed = '''from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.recommender import recommend, get_user_context
from core.recommender import collection, df_users

app = FastAPI(
    title="BCT LLM Challenge - Task B: Recommendation Agent",
    description="Personalised recommendations powered by semantic search + LLaMA reasoning.",
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

class RecommendRequest(BaseModel):
    user_id: str
    user_request: str
    conversation_history: Optional[List[dict]] = []
    nigerian_mode: Optional[bool] = True

class UserContextRequest(BaseModel):
    user_id: str

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "service": "Task B - Recommendation Agent"}

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    try:
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
    return get_user_context(req.user_id)

@app.get("/stats")
def stats():
    return {
        "total_businesses_indexed": collection.count(),
        "total_users":              len(df_users),
        "model":                    "llama-3.1-8b-instant via Groq",
        "retrieval":                "ChromaDB + sentence-transformers",
        "nigerian_mode":            "enabled by default",
    }
'''

with open("../task_b/app/main.py", "w", encoding="utf-8") as f:
    f.write(main_b_fixed)
print("✅ Task B main.py fixed")