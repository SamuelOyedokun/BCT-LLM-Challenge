# NaijaReview Intelligence System
### DSN × BCT LLM Agent Challenge — Hackathon 3.0

A dual-agent LLM system for Nigerian-contextualised user modelling and personalised recommendation.

---

## 🏗️ Architecture

| Component | Technology |
|-----------|-----------|
| LLM Backbone | LLaMA 3.1-8b-instant via Groq |
| Semantic Retrieval | ChromaDB + sentence-transformers |
| API Framework | FastAPI |
| Containerisation | Docker |
| Dataset | Yelp Open Dataset |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed
- Groq API key

### Run Both Agents

```bash
# Clone the repo
git clone https://github.com/samueloyedokun/bct-llm-challenge
cd bct-llm-challenge

# Add your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Build and run
docker-compose up --build
```

Task A available at: http://localhost:8000  
Task B available at: http://localhost:8001

---

## 📡 API Endpoints

### Task A — User Modelling
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /simulate-review | Generate review for user+business |
| POST | /get-persona | Get user behavioural profile |
| GET | /stats | Dataset statistics |

**Example request:**
```json
POST /simulate-review
{
  "user_id": "abc123",
  "business_id": "xyz789",
  "nigerian_mode": true
}
```

### Task B — Recommendation Agent
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /recommend | Get personalised recommendations |
| POST | /user-context | Get user context profile |
| GET | /stats | Index statistics |

**Example request:**
```json
POST /recommend
{
  "user_id": "abc123",
  "user_request": "I want a good suya spot near me",
  "conversation_history": [],
  "nigerian_mode": true
}
```

---

## 🇳🇬 Nigerian Contextualisation Layer

A key innovation of this system is the Nigerian cultural adaptation layer, which:

- Maps users to Nigerian consumer archetypes (Lagos Hustler, Abuja Elite, Student Budget, SME Owner, Food Enthusiast)
- Adapts linguistic tone to reflect Nigerian English and Pidgin patterns
- Encodes price sensitivity signals relevant to Nigerian economic context
- Weights peer trust and social proof signals appropriately

---

## 📁 Project Structure