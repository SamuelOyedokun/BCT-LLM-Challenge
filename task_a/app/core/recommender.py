import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR))

from startup import download_all
from core.config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import random

df_users   = None
collection = None
model      = None
groq_client = None

def load_data():
    global df_users, collection, model, groq_client
    if df_users is not None:
        return
    download_all()
    print("Loading recommender components...")

    DB_PATH = str(BASE_DIR.parent / "vectordb")

    df_users = pd.read_csv(DATA_DIR / "user_profiles.csv")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client_db  = chromadb.PersistentClient(path=DB_PATH)
    collection = client_db.get_collection("businesses")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"   Recommender ready — {collection.count():,} businesses indexed")

NIGERIAN_FLAVOUR = [
    "Respond like a friendly Nigerian concierge who knows the city well.",
    "Use warm Nigerian English. Say things like 'this place no go disappoint you'.",
    "Sound like a knowledgeable Nigerian friend giving genuine advice.",
]

def get_user_context(user_id: str) -> dict:
    load_data()
    row = df_users[df_users["user_id"] == user_id]
    if row.empty:
        return {
            "user_id":             user_id,
            "is_cold_start":       True,
            "favorite_categories": "Restaurants, Food",
            "most_common_city":    "Lagos",
            "avg_biz_stars":       4.0,
            "tip_count":           0,
        }
    u = row.iloc[0]
    return {
        "user_id":             user_id,
        "is_cold_start":       False,
        "favorite_categories": str(u["favorite_categories"]),
        "most_common_city":    str(u["most_common_city"]),
        "avg_biz_stars":       float(u["avg_biz_stars"]),
        "tip_count":           int(u["tip_count"]),
    }

def retrieve_candidates(query: str, category_hint: str = "", n: int = 10) -> list:
    load_data()
    search_query = f"{query} {category_hint}".strip()
    embedding    = model.encode([search_query]).tolist()
    results      = collection.query(query_embeddings=embedding, n_results=n)
    candidates = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        candidates.append({
            "business_id": results["ids"][0][i],
            "name":        meta.get("name", ""),
            "city":        meta.get("city", ""),
            "state":       meta.get("state", ""),
            "categories":  meta.get("categories", ""),
            "stars":       meta.get("stars", 0),
            "review_count":meta.get("review_count", 0),
        })
    return candidates

def rank_and_explain(user_context, candidates, user_request,
                     conversation_history=[], nigerian_mode=True) -> dict:
    load_data()
    candidate_text = ""
    for i, c in enumerate(candidates, 1):
        candidate_text += (
            f"{i}. {c['name']} ({c['categories']}) — "
            f"{c['city']}, {c['state']} — "
            f"{c['stars']} stars\n"
        )

    history_text = ""
    if conversation_history:
        history_text = "CONVERSATION HISTORY:\n"
        for turn in conversation_history[-4:]:
            history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    nigerian_instruction = random.choice(NIGERIAN_FLAVOUR) if nigerian_mode else ""
    cold_start_note = "Note: New user — use popular highly-rated options." if user_context["is_cold_start"] else ""

    prompt = f"""You are an intelligent recommendation agent.

USER PROFILE:
- Favourite categories: {user_context["favorite_categories"]}
- Usually visits: {user_context["most_common_city"]}
- Average rating they give: {user_context["avg_biz_stars"]} stars
{cold_start_note}

{history_text}

USER REQUEST: {user_request}

CANDIDATE BUSINESSES:
{candidate_text}

TASK:
1. Select TOP 5 most relevant businesses
2. Write a short personalised reason for each
3. {nigerian_instruction}

Respond in this EXACT JSON format:
{{
  "reasoning": "Brief reasoning 2-3 sentences",
  "recommendations": [
    {{
      "rank": 1,
      "name": "Business Name",
      "city": "City",
      "state": "ST",
      "categories": "Category",
      "stars": 4.5,
      "why": "Personalised reason"
    }}
  ],
  "follow_up": "One follow-up question"
}}

Return ONLY valid JSON.
"""

    import json
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=TEMPERATURE,
    )
    raw  = response.choices[0].message.content.strip()
    raw  = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def recommend(user_id, user_request, conversation_history=[], nigerian_mode=True) -> dict:
    user_context = get_user_context(user_id)
    candidates   = retrieve_candidates(user_request, user_context["favorite_categories"], n=15)
    result       = rank_and_explain(user_context, candidates, user_request,
                                    conversation_history, nigerian_mode)
    result["user_id"]       = user_id
    result["is_cold_start"] = user_context["is_cold_start"]
    result["user_city"]     = user_context["most_common_city"]
    result["request"]       = user_request
    return result
