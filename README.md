# NaijaReview Intelligence System
### DSN × BCT LLM Agent Challenge — Hackathon 3.0

A culturally-grounded dual-agent LLM framework for Nigerian user modelling and personalised recommendation.

🔴 **Live Demo:** https://bct-task-a.onrender.com

---

## 🏗️ Architecture

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM Backbone | LLaMA 3.1-8b-instant via Groq | Review generation + recommendation reasoning |
| Semantic Retrieval | ChromaDB + sentence-transformers | Business semantic search |
| API Framework | FastAPI + Uvicorn | REST endpoint exposure |
| Data Pipeline | Pandas + Python | Persona extraction + behavioural modelling |
| Deployment | Render (cloud) | Live at bct-task-a.onrender.com |
| Frontend | Vanilla JS + HTML/CSS | Interactive agent dashboards |
| Dataset | Yelp Open Dataset | 908,915 interactions, 150,346 businesses |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API key (free at console.groq.com)

### Run Locally

```bash
# Clone the repo
git clone https://github.com/SamuelOyedokun/BCT-LLM-Challenge
cd BCT-LLM-Challenge

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r task_a/requirements.txt

# Add your API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here

# Run Task A
cd task_a
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run Task B (new terminal)
cd task_b
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Or use the Live Deployment
Both Task A and Task B are accessible at:
https://bct-task-a.onrender.com
---

## 📡 API Endpoints

### Task A — User Modelling & Review Simulation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /simulate-review | Generate review + rating for user+business |
| POST | /get-persona | Get user behavioural profile |
| GET | /random-ids | Get random user_id and business_id |
| GET | /stats | Dataset statistics |

**Example:**
```json
POST /simulate-review
{
  "user_id": "user001",
  "business_id": "biz003",
  "nigerian_mode": true
}
```

### Task B — Conversational Recommendation Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /recommend | Get personalised recommendations |
| POST | /user-context | Get user context profile |
| GET | /stats | Index statistics |

**Example:**
```json
POST /recommend
{
  "user_id": "user001",
  "user_request": "I want a good suya spot near me",
  "conversation_history": [],
  "nigerian_mode": true
}
```

---

## 🇳🇬 Nigerian Contextualisation Layer

The defining innovation of this system — a structured cultural adaptation framework operating across both agents:

- **5 Behavioural Archetypes:** Lagos Hustler, Abuja Elite, Student Budget, SME Owner, Food Enthusiast
- **Linguistic Adaptation:** Pidgin-English code-switching ("e dey sweet", "price don too much")
- **Price Sensitivity Modelling:** Explicit affordability signals calibrated to Nigerian economic context
- **Trust & Social Proof:** Peer validation, review density, and community endorsement weighting

---

## 📊 Evaluation Results

| Metric | Value | Task |
|--------|-------|------|
| RMSE (Rating Accuracy) | 0.84 | A |
| BERTScore F1 | 0.87 | A |
| ROUGE-1 | 0.48 | A |
| NDCG@10 | 0.71 | B |
| Hit Rate@10 | 0.77 | B |
| Precision@5 | 0.64 | B |

---

## 🧪 Test Results

All endpoints validated before submission:

**Task A: ✅ Health  ✅ Stats  ✅ Simulate Review  ✅ Get Persona  ✅ Random IDs
Task B: ✅ Health  ✅ Stats  ✅ Known User  ✅ Cold Start  ✅ Multi-turn**
---

## 📄 Solution Paper

Full methodology, ablation studies, and evaluation available in:
[`NaijaReview_Solution_Paper.pdf`](./NaijaReview_Solution_Paper.pdf)

---

## 👤 Author

**Samuel Oyedokun**  
 thesamueloyedokun@gmail.com 
