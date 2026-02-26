# ParallelLines — IntuOne Perception Engine

A perception framework that maps how belief, sentiment, and conviction form, evolve, and persist. Built on two primary layers — **Submind** and **IntuOne** — it does not attempt to predict outcomes. It observes how humans relate to uncertainty, meaning, and trust.

---

## Architecture

```
External sources
(Polymarket, Twitter, news sites, …)
          │
          ▼
╔═════════════════════════════════════════════════════════════╗
║                     SUBMIND LAYER                           ║  app/agents/
║                                                             ║
║  PolymarketSubmind  ✓    Silent observer. No interpretation.║
║  TwitterSubmind     ~    Each submind owns one source and   ║
║  NewsSubmind        ~    records raw presence as Signals.   ║
╚══╤══════════════════════════════════════════════════════════╝
   │  signals fan out to all layers simultaneously
   │
   ├──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
   ▼              ▼              ▼              ▼              ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│PROBABILITY│ │CONVICTION│ │   ECHO   │ │  MEMORY  │ │  SHADOW  │ │FRACTURE ~│
│  market  │ │sentiment │ │  social  │ │   news   │ │geopolit. │ │(planned) │
│          │ │          │ │          │ │          │ │          │ │          │
│ What does│ │How deeply│ │How is    │ │How is    │ │What unspo│ │Where does│
│ the crowd│ │is belief │ │belief    │ │belief    │ │ken forces│ │belief    │
│ price as │ │held?     │ │amplified?│ │preserved?│ │drive it? │ │detach    │
│ likely?  │ │          │ │          │ │          │ │          │ │from fact?│
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────────┘
     │             │            │             │            │
     └─────────────┴────────────┴─────────────┴────────────┘
                                    │
                             PERCEPTION INDEX
                        (non-linear composite — app/layers/synthesis.py)
                                    │
                                    ▼
╔═════════════════════════════════════════════════════════════╗
║                     INTUONE LAYER                           ║  app/inference/
║                                                             ║
║  Fine-tuned Llama 3.1 8B (QLoRA)   Interpreter. Reads the  ║
║  LoRA adapter trained on Claude-   Perception Index and     ║
║  generated perception analyses     translates it into       ║
║                                    natural-language output. ║
╚══════════════════╤══════════════════════════════════════════╝
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    Chat UI    Dashboard    REST API
```

**Submind layer** — the silent observer. Each submind owns exactly one external source. It fetches raw data, records presence without interpretation, and normalises into typed `Signal` objects. Adding a new data source means writing a new submind.

**The six perception layers** run simultaneously and independently — each answers a different question about the same reality. They do not form a pipeline. No layer overrides another. The Perception Index is their non-linear composite; compressing it to a single score hides instability (high Conviction + high Fracture = instability, not certainty).

**Fracture Layer** — not yet implemented. Will detect divergence between crowd probability and minority conviction: where belief is detaching from data.

**IntuOne layer** — the interpreter. It reads resonance, not correctness. It receives the full Perception Index and translates it into natural-language analysis. Trained via a teacher→student loop: Claude generates gold-standard perception analyses, QLoRA bakes that reasoning into a local model that runs entirely on your own hardware.

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
