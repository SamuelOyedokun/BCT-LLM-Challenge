import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR))

from startup import download_all
from core.config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE

import pandas as pd
import numpy as np
from groq import Groq
import random
import json

df_users  = None
df_biz    = None
groq_client = None

def load_data():
    global df_users, df_biz, groq_client
    if df_users is not None:
        return
    download_all()
    print("Loading Task B components...")
    df_users = pd.read_csv(DATA_DIR / "user_profiles.csv")
    df_biz   = pd.read_csv(DATA_DIR / "businesses.csv")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"   Task B ready — {len(df_biz):,} businesses | {len(df_users):,} users")

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

def retrieve_candidates(user_request: str, n: int = 15,
                        conversation_history: list = [],
                        user_context: dict = {}) -> list:
    load_data()
    # Build search query from request + recent conversation context
    full_query = user_request

    # If request is vague (cheaper, better, another), use last assistant topic
    vague_words = {"cheaper","better","another","different","similar",
                   "else","more","other","again","instead"}
    request_words = set(user_request.lower().split())

    if request_words & vague_words and conversation_history:
        # Pull topic from last user message
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                full_query = turn["content"] + " " + user_request
                break

    # Add user favourite category as soft hint
    fav_cats = user_context.get("favorite_categories", "")
    if fav_cats and fav_cats != "Unknown":
        full_query = full_query + " " + fav_cats[:50]

    keywords = full_query.lower().split()
    stop_words = {"a","an","the","me","my","i","want","need","good","great",
                  "best","find","looking","for","some","place","to","and","or",
                  "cheaper","better","another","different","similar","time",
                  "this","that","these","those","something","anything"}
    keywords = [k for k in keywords if k not in stop_words and len(k) > 2]

    if not keywords:
        keywords = ["restaurant", "food"]

    def score_row(row):
        text = (str(row.get("categories", "")) + " " + 
                str(row.get("name", ""))).lower()
        return sum(1 for kw in keywords if kw in text)

    df_sample = df_biz.sample(min(8000, len(df_biz)), random_state=42).copy()
    df_sample["score"] = df_sample.apply(score_row, axis=1)
    df_top = df_sample[df_sample["score"] > 0]

    if len(df_top) == 0:
        df_top = df_sample.nlargest(n, "stars")
    else:
        df_top = df_top.nlargest(n, ["score", "stars"])

    candidates = []
    for _, row in df_top.iterrows():
        candidates.append({
            "name":        str(row.get("name", "")),
            "city":        str(row.get("city", "")),
            "state":       str(row.get("state", "")),
            "categories":  str(row.get("categories", "")),
            "stars":       float(row.get("stars", 0)),
            "review_count":int(row.get("review_count", 0)),
        })
    return candidates

def get_stats() -> dict:
    load_data()
    return {
        "total_businesses_indexed": len(df_biz),
        "total_users":              len(df_users),
        "model":                    "llama-3.1-8b-instant via Groq",
        "retrieval":                "Keyword + LLaMA reasoning",
        "nigerian_mode":            "enabled by default",
    }

def recommend(user_id: str, user_request: str,
              conversation_history: list = [],
              nigerian_mode: bool = True) -> dict:
    load_data()
    user_context = get_user_context(user_id)
    candidates   = retrieve_candidates(user_request, n=15, conversation_history=conversation_history, user_context=user_context)

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

CANDIDATE BUSINESSES (matched to user request):
{candidate_text}

TASK:
1. Select TOP 5 businesses that best match the USER REQUEST
2. Write a short personalised reason for each considering user profile
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

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=TEMPERATURE,
    )
    raw  = response.choices[0].message.content.strip()
    raw  = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    result["user_id"]       = user_id
    result["is_cold_start"] = user_context["is_cold_start"]
    result["user_city"]     = user_context["most_common_city"]
    result["request"]       = user_request
    return result
