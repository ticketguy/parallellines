# ParallelLines — IntuOne Perception Engine

Multi-layer internet signal analysis that produces real-time intelligence briefings on prediction markets. Raw signals from Polymarket are processed through six analytical layers and synthesised by a fine-tuned Llama 3.1 8B model (IntuOne).

---

## How it works

```
Polymarket / Web
      │
      ▼
 ┌─────────────┐
 │  Connectors  │  pull market + page data
 └──────┬──────┘
        │
        ▼
 ┌──────────────────────────────────────────────────┐
 │                  Six Layers                       │
 │  market ✓  · news ✓  · sentiment ✓              │
 │  social ~  · geopolitical ~  · synthesis ✓      │
 └──────────────────────┬───────────────────────────┘
      ✓ live   ~ stub (connector not yet wired)
                        │
                        ▼
              ┌─────────────────┐
              │    IntuOne LLM   │  fine-tuned Llama 3.1 8B (QLoRA)
              └────────┬────────┘
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
        Chat UI    Dashboard    REST API
```

IntuOne is trained with a teacher→student loop: Claude generates gold-standard briefings from live signals, and those examples fine-tune the local model via QLoRA so it runs entirely on your own hardware.

---

## Quick start

**Windows**
```bat
run.bat
```

**Linux / macOS**
```bash
chmod +x run.sh && ./run.sh
```

Both scripts create a virtual environment, install dependencies, set up the database, and start the server. After startup, open:

| URL | What |
|-----|------|
| `http://localhost:8000/app/chat.html` | Chat with IntuOne |
| `http://localhost:8000/app/dashboard.html` | Signal dashboard |
| `http://localhost:8000/docs` | Interactive API docs |
| `http://localhost:8000/health` | Health + model status |

The server runs immediately without a GPU. To enable the fine-tuned model, follow the training steps in [SETUP.md](SETUP.md).

---

## Project structure

```
parallellines/
├── app/
│   ├── api/v1/          # REST endpoints (chat, signals, markets, training, …)
│   ├── connectors/      # Data sources (Polymarket, web crawler)
│   ├── layers/          # Six analytical layers + synthesis
│   ├── inference/       # Model loading & inference pipeline
│   ├── training/        # Dataset formatter, generator (Claude), QLoRA trainer
│   ├── memory/          # Extraction & retrieval
│   ├── models/          # SQLAlchemy ORM models
│   ├── schemas/         # Pydantic schemas
│   ├── services/        # Business logic (ingestion, IntuOne service)
│   ├── config.py        # Settings (loaded from .env)
│   └── main.py          # FastAPI app + lifespan
├── alembic/             # Database migrations
├── static/              # Frontend (chat.html, dashboard.html)
├── tests/
├── .env.example
├── pyproject.toml
├── run.bat              # Windows launcher
├── run.sh               # Linux/macOS launcher
└── SETUP.md             # Full setup guide (database, training, GPU)
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Database | PostgreSQL (production) / SQLite (dev) |
| ORM | SQLAlchemy async + Alembic migrations |
| Data validation | Pydantic v2 |
| Training data | Anthropic Claude (teacher model) |
| Fine-tuning | QLoRA via PEFT + TRL (SFTTrainer) |
| Base model | Meta Llama 3.1 8B Instruct |
| HTTP client | httpx |
| Frontend | Vanilla HTML/CSS/JS |

---

## API overview

All endpoints are under `/api/v1`. Full interactive docs at `/docs`.

| Endpoint | Description |
|----------|-------------|
| `POST /ingest/run` | Trigger a manual data pull from all connectors |
| `GET  /signals` | List ingested signals |
| `GET  /markets` | List tracked Polymarket markets |
| `GET  /layers/{layer_name}` | Get latest output for a specific layer |
| `POST /chat` | Chat with IntuOne |
| `POST /training/generate` | Generate training examples (requires Anthropic key) |
| `GET  /training/export` | Export dataset to disk for fine-tuning |
| `GET  /reports` | List generated briefings |
| `GET  /health` | Server + model status |

---

## Configuration

Copy `.env.example` to `.env` and fill in your values. See [SETUP.md](SETUP.md) for the full reference.

The minimum required to run the server is a valid `DATABASE_URL`. An `ANTHROPIC_API_KEY` is only needed for training data generation. The `HUGGING_FACE_HUB_TOKEN` is only needed to download the base model for training.

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=app
```
