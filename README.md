# ParallelLines — IntuOne Perception Engine

A perception engine built on two primary layers — **Submind** and **IntuOne** — that turns raw internet signals into real-time intelligence briefings on prediction markets.

---

## Architecture

The engine has two primary layers:

```
External sources
(Polymarket, Twitter, news sites, …)
          │
          ▼
╔═════════════════════════════════════╗
║          SUBMIND LAYER              ║  app/agents/
║                                     ║
║  PolymarketSubmind  ✓               ║  Each submind owns one data source.
║  TwitterSubmind     ~  (planned)    ║  It fetches raw data and normalises
║  NewsSubmind        ~  (planned)    ║  it into typed Signal objects.
╚══════════════════╤══════════════════╝
                   │  normalised signals
                   ▼
        ┌──────────────────────────┐
        │   Signal processing      │  app/layers/
        │   market · news ·        │  Six scorers reduce signals to
        │   sentiment · social ·   │  numeric layer scores (-1 to +1).
        │   geopolitical ·         │
        │   synthesis              │
        └──────────┬───────────────┘
                   │  layer scores
                   ▼
╔═════════════════════════════════════╗
║          INTUONE LAYER              ║  app/inference/
║                                     ║
║  Fine-tuned Llama 3.1 8B (QLoRA)   ║  Reads all layer scores and
║  LoRA adapter trained on Claude-   ║  generates a structured
║  labelled briefings                 ║  intelligence briefing.
╚══════════════════╤══════════════════╝
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    Chat UI    Dashboard    REST API
```

**Submind layer** — each submind is an autonomous agent responsible for exactly one external source. It knows the source's API, normalises the data into the common `SignalCreate` schema, and assigns it to the correct analytical layer. Adding a new data source means writing a new submind.

**IntuOne layer** — the fine-tuned local model. It receives a structured context of layer scores and signal excerpts and produces a briefing. It is trained via a teacher→student loop: Claude generates gold-standard briefings, and QLoRA fine-tuning bakes that reasoning into a model that runs entirely on your own hardware.

The six signal-processing layers (market, news, sentiment, social, geopolitical, synthesis) sit between the two primary layers as a normalisation pipeline, not as top-level architecture.

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
│   ├── agents/          # SUBMIND LAYER — one agent per data source
│   │   ├── base.py      #   SubmindBase abstract class
│   │   └── polymarket.py#   PolymarketSubmind (live)
│   ├── connectors/      # Low-level HTTP fetchers (used by agents)
│   ├── layers/          # Signal processing — market, news, sentiment,
│   │                    #   social, geopolitical, synthesis scorers
│   ├── inference/       # INTUONE LAYER — model loading & inference pipeline
│   ├── training/        # Dataset formatter, generator (Claude), QLoRA trainer
│   ├── memory/          # Extraction & retrieval
│   ├── api/v1/          # REST endpoints (chat, signals, markets, training, …)
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
└── SETUP.md             # Full setup guide (database, subminds, training, GPU)
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
