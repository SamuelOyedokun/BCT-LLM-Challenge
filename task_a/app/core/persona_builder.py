import pandas as pd
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
sys.path.insert(0, str(BASE_DIR))

from startup import download_all
download_all()

print("Loading data files...")
df_users = pd.read_csv(DATA_DIR / "user_profiles.csv")
df_tips  = pd.read_csv(DATA_DIR / "tips_with_business.csv")
df_biz   = pd.read_csv(DATA_DIR / "businesses.csv")
print(f"   users: {len(df_users):,} | businesses: {len(df_biz):,} | tips: {len(df_tips):,}")

def get_user_persona(user_id: str) -> dict:
    user_row = df_users[df_users["user_id"] == user_id]
    if user_row.empty:
        return {"error": f"User {user_id} not found"}
    u = user_row.iloc[0]
    user_tips = df_tips[df_tips["user_id"] == user_id].tail(5)
    recent_tips       = user_tips["text"].tolist()
    recent_businesses = user_tips["name"].tolist() if "name" in user_tips.columns else []
    avg_stars = float(u["avg_biz_stars"])
    if avg_stars >= 4.0:
        tone = "enthusiastic and positive"
    elif avg_stars >= 3.0:
        tone = "balanced and moderate"
    else:
        tone = "critical and demanding"
    tip_count = int(u["tip_count"])
    if tip_count >= 20:
        reviewer_type = "power reviewer"
    elif tip_count >= 5:
        reviewer_type = "regular reviewer"
    else:
        reviewer_type = "occasional reviewer"
    return {
        "user_id":             str(user_id),
        "tip_count":           tip_count,
        "avg_stars":           round(avg_stars, 2),
        "businesses_visited":  int(u["businesses_visited"]),
        "favorite_categories": str(u["favorite_categories"]),
        "most_common_city":    str(u["most_common_city"]),
        "last_active":         str(u["last_active"]),
        "tone":                tone,
        "reviewer_type":       reviewer_type,
        "recent_tips":         recent_tips,
        "recent_businesses":   recent_businesses,
    }

def get_business_context(business_id: str) -> dict:
    biz_row = df_biz[df_biz["business_id"] == business_id]
    if biz_row.empty:
        return {"error": f"Business {business_id} not found"}
    b = biz_row.iloc[0]
    return {
        "business_id":  str(business_id),
        "name":         str(b["name"]),
        "city":         str(b["city"]),
        "state":        str(b["state"]),
        "categories":   str(b["categories"]),
        "avg_stars":    float(b["stars"]),
        "review_count": int(b["review_count"]),
    }

def get_random_user_id() -> str:
    return str(df_users.sample(1).iloc[0]["user_id"])

def get_random_business_id() -> str:
    return str(df_biz.sample(1).iloc[0]["business_id"])
