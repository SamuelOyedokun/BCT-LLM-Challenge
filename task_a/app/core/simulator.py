from groq import Groq
from core.config import GROQ_API_KEY, MODEL_NAME, MAX_TOKENS, TEMPERATURE
from core.persona_builder import get_user_persona, get_business_context
import random

client = Groq(api_key=GROQ_API_KEY)

NIGERIAN_FLAVOUR = [
    "Use a warm, expressive Nigerian English tone occasionally.",
    "Sprinkle in mild Pidgin English phrases like 'e dey sweet', 'no be small thing', 'I no go lie'.",
    "Sound like a Nigerian who has visited this type of place before.",
    "Use expressions like 'chai!', 'e be like say', 'this one na correct', 'to God be the glory' where natural.",
]

def simulate_review(user_id: str, business_id: str, nigerian_mode: bool = True) -> dict:
    """
    Simulate a review for a given user and business.
    Returns star rating + written review.
    """
    persona   = get_user_persona(user_id)
    business  = get_business_context(business_id)

    if "error" in persona:
        return persona
    if "error" in business:
        return business

    # Determine star rating based on user tendency + business quality
    base_stars = (persona["avg_stars"] + business["avg_stars"]) / 2
    noise      = random.uniform(-0.5, 0.5)
    raw        = base_stars + noise
    star_rating = max(1, min(5, round(raw)))

    # Build Nigerian flavour instruction
    nigerian_instruction = random.choice(NIGERIAN_FLAVOUR) if nigerian_mode else ""

    # Recent tips as writing style examples
    style_examples = ""
    if persona["recent_tips"]:
        examples = "\n".join([f"- {t}" for t in persona["recent_tips"][:3]])
        style_examples = f"""
Here are some of this user\'s past reviews to help you match their writing style:
{examples}
"""

    prompt = f"""You are simulating a Yelp review written by a real person with the following profile:

USER PROFILE:
- Reviewer type: {persona["reviewer_type"]}
- Tone: {persona["tone"]}
- Favourite categories: {persona["favorite_categories"]}
- City: {persona["most_common_city"]}
- Average rating they give: {persona["avg_stars"]} stars
- Number of reviews written: {persona["tip_count"]}
{style_examples}

BUSINESS BEING REVIEWED:
- Name: {business["name"]}
- Category: {business["categories"]}
- Location: {business["city"]}, {business["state"]}
- Overall rating: {business["avg_stars"]} stars

TASK:
Write a realistic Yelp review for this business as this user would write it.
The review should reflect a {star_rating}-star experience.
Be specific, mention the business type and location naturally.
Length: 3-5 sentences. Sound human, not robotic.
{nigerian_instruction}

Return ONLY the review text. No preamble, no explanation.
"""

    response = client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
    )

    review_text = response.choices[0].message.content.strip()

    return {
        "user_id":      user_id,
        "business_id":  business_id,
        "star_rating":  star_rating,
        "review_text":  review_text,
        "persona_tone": persona["tone"],
        "business_name": business["name"],
        "nigerian_mode": nigerian_mode,
    }
