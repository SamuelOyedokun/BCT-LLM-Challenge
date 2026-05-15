import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from pathlib import Path
import pandas as pd

BASE    = Path(__file__).parent.parent.parent.parent
SHARED  = BASE / "shared"
DB_PATH = str(BASE / "task_b" / "vectordb")

# Load once at startup
print("Loading recommender components...")
model      = SentenceTransformer("all-MiniLM-L6-v2")
client_db  = chromadb.PersistentClient(path=DB_PATH)
collection = client_db.get_collection("businesses")
df_users   = pd.read_csv(SHARED / "user_profiles.csv")
print("   Recommender ready")

from core.config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
groq_client = Groq(api_key=GROQ_API_KEY)

NIGERIAN_FLAVOUR = [
    "Respond like a friendly Nigerian concierge who knows the city well.",
    "Use warm Nigerian English. Occasionally say things like 'this place no go disappoint you' or 'e dey sweet'.",
    "Sound like a knowledgeable Nigerian friend giving genuine advice.",
]

import random

def get_user_context(user_id: str) -> dict:
    """Get user profile or return cold-start defaults."""
    row = df_users[df_users["user_id"] == user_id]
    if row.empty:
        return {
            "user_id":             user_id,
            "is_cold_start":       True,
            "favorite_categories": "Restaurants, Food",
            "most_common_city":    "Las Vegas",
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
    """Semantic search in ChromaDB."""
    search_query = f"{query} {category_hint}".strip()
    embedding    = model.encode([search_query]).tolist()
    results      = collection.query(
        query_embeddings = embedding,
        n_results        = n,
    )
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
            "score":       round(1 - results["distances"][0][i], 3),
        })
    return candidates


def rank_and_explain(
    user_context: dict,
    candidates: list,
    user_request: str,
    conversation_history: list = [],
    nigerian_mode: bool = True,
) -> dict:
    """Use LLaMA to reason about candidates and return ranked recommendations."""

    candidate_text = ""
    for i, c in enumerate(candidates, 1):
        candidate_text += (
            f"{i}. {c['name']} ({c['categories']}) — "
            f"{c['city']}, {c['state']} — "
            f"{c['stars']} stars, {c['review_count']} reviews\n"
        )

    history_text = ""
    if conversation_history:
        history_text = "CONVERSATION HISTORY:\n"
        for turn in conversation_history[-4:]:
            history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    nigerian_instruction = random.choice(NIGERIAN_FLAVOUR) if nigerian_mode else ""

    cold_start_note = ""
    if user_context["is_cold_start"]:
        cold_start_note = "Note: This is a new user with no history. Use popular, highly-rated options."

    prompt = f"""You are an intelligent recommendation agent.

USER PROFILE:
- Favourite categories: {user_context["favorite_categories"]}
- Usually visits: {user_context["most_common_city"]}
- Average rating they give: {user_context["avg_biz_stars"]} stars
- Total reviews written: {user_context["tip_count"]}
{cold_start_note}

{history_text}

USER REQUEST: {user_request}

CANDIDATE BUSINESSES (from semantic search):
{candidate_text}

TASK:
1. Reason about which businesses best match this user's preferences and request
2. Select the TOP 5 most relevant ones
3. For each, write a short 1-2 sentence personalised reason why this user would like it
4. {nigerian_instruction}

Respond in this EXACT JSON format:
{{
  "reasoning": "Your brief reasoning about the user's needs (2-3 sentences)",
  "recommendations": [
    {{
      "rank": 1,
      "name": "Business Name",
      "city": "City",
      "state": "ST",
      "categories": "Category",
      "stars": 4.5,
      "why": "Personalised reason for this user"
    }}
  ],
  "follow_up": "One natural follow-up question to refine recommendations further"
}}

Return ONLY valid JSON. No preamble, no explanation outside the JSON.
"""

    response = groq_client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 1024,
        temperature = TEMPERATURE,
    )

    import json
    raw  = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw  = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    return data


def recommend(
    user_id: str,
    user_request: str,
    conversation_history: list = [],
    nigerian_mode: bool = True,
) -> dict:
    """Full recommendation pipeline."""

    # Step 1: Get user context
    user_context = get_user_context(user_id)

    # Step 2: Semantic retrieval
    category_hint = user_context["favorite_categories"]
    candidates    = retrieve_candidates(user_request, category_hint, n=15)

    # Step 3: LLM ranking + explanation
    result = rank_and_explain(
        user_context         = user_context,
        candidates           = candidates,
        user_request         = user_request,
        conversation_history = conversation_history,
        nigerian_mode        = nigerian_mode,
    )

    # Step 4: Attach metadata
    result["user_id"]       = user_id
    result["is_cold_start"] = user_context["is_cold_start"]
    result["user_city"]     = user_context["most_common_city"]
    result["request"]       = user_request

    return result
