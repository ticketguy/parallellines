# Setup Guide

This guide covers four paths, in order of complexity:

1. **[Run the server](#1-run-the-server-no-gpu-needed)** — UI, API, and signal ingestion. No GPU, no model weights required.
2. **[Activate the other layers](#2-activate-the-other-layers)** — Wire up news, sentiment, social, and geopolitical data sources.
3. **[Generate training data](#3-generate-training-data)** — Use Claude to label live signals.
4. **[Train and run IntuOne](#4-train-intuone)** — Fine-tune Llama 3.1 8B on your data, serve it locally.

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | [python.org/downloads](https://www.python.org/downloads/) |
| Git | [git-scm.com](https://git-scm.com) |
| PostgreSQL **or** SQLite | SQLite needs no installation — good for local testing |
| Anthropic API key | Required for training data generation only |
| Hugging Face token + Meta Llama license | Required for downloading the base model |
| NVIDIA GPU with 16 GB+ VRAM | Required for training and local model inference |

---

## 1. Run the server (no GPU needed)

### 1a. Clone and enter the repo

```bash
git clone <your-repo-url>
cd parallellines
```

### 1b. Configure the environment

```bash
cp .env.example .env
```

Open `.env` and set your database. For a quick start with no PostgreSQL:

```dotenv
DATABASE_URL=sqlite+aiosqlite:///./parallellines.db
```

For PostgreSQL (recommended for production):

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:yourpassword@localhost/parallellines
```

Make sure PostgreSQL is running and the database exists before proceeding:

```sql
-- in psql
CREATE DATABASE parallellines;
```

### 1c. Start the server

**Windows**
```bat
run.bat
```

**Linux / macOS**
```bash
chmod +x run.sh && ./run.sh
```

The script will:
- Create a Python virtual environment (`.venv/`)
- Install all dependencies
- Run Alembic migrations to create the database schema
- Start the server on port 8000

Open [http://localhost:8000/app/chat.html](http://localhost:8000/app/chat.html).

The server is fully functional at this point. Signal ingestion pulls from Polymarket every 5 minutes automatically. The chat endpoint returns signal-layer summaries. The model status at `/health` will show `"model_loaded": false` until you complete step 3.

### 1d. Manual install (alternative to run scripts)

If you prefer to manage things yourself:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -e "."
cp .env.example .env   # then edit .env
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2. Activate the other layers

Out of the box only the **Market** layer has live data (Polymarket). Here is the status of all six layers and how to activate each one.

### Layer status overview

| Layer | Status | What feeds it |
|-------|--------|---------------|
| Market | Live | Polymarket connector (auto-polls every 5 min) |
| News | Ready — needs sources | Web crawler connector |
| Sentiment | Ready — needs sources | Web crawler connector (same signals) |
| Social | Stub | Twitter/Reddit connector (not yet wired) |
| Geopolitical | Stub | Custom connector needed |
| Synthesis | Live | Aggregates whichever layers have scores |

### News and sentiment layers — add crawl sources

The web crawler connector is already running. It just needs URLs to fetch. Add any news page, RSS feed, or blog:

```bash
# Add a news source
curl -X POST http://localhost:8000/api/v1/ingest/sources \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://reuters.com/world",
    "topic_tags": ["geopolitics", "finance"],
    "layer": "news",
    "label": "Reuters World"
  }'

# Add another for a specific topic
curl -X POST http://localhost:8000/api/v1/ingest/sources \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://coindesk.com",
    "topic_tags": ["bitcoin", "crypto"],
    "layer": "news",
    "label": "CoinDesk"
  }'
```

Sources are stored in the database and loaded automatically on the next ingestion cycle (every 5 minutes). Trigger an immediate pull with:

```bash
curl -X POST http://localhost:8000/api/v1/ingest/run
```

The same signals flow into both the **News** layer (headline/volume scoring) and the **Sentiment** layer (NLP sentiment scoring), so both activate with the same sources.

To route a source to the sentiment layer specifically:

```bash
curl -X POST http://localhost:8000/api/v1/ingest/sources \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/opinion",
    "topic_tags": ["politics"],
    "layer": "sentiment",
    "label": "Opinion feed"
  }'
```

### Social layer — Twitter / Reddit

The social layer scorer is implemented and ready. It needs a connector that produces signals with a `sentiment_score` field. Two future connectors are planned (keys already in `.env.example`):

```dotenv
# .env — uncomment when you wire up the connector
# TWITTER_BEARER_TOKEN=your-bearer-token-here
# NEWS_API_KEY=your-newsapi-key-here
```

Until a social connector is built, you can feed it manually via the signals API:

```bash
curl -X POST http://localhost:8000/api/v1/signals \
  -H "Content-Type: application/json" \
  -d '{
    "source": "manual",
    "layer": "social",
    "topic_tags": ["bitcoin"],
    "signal_strength": 0.8,
    "processed_data": {"sentiment_score": 0.6},
    "confidence": 0.7
  }'
```

### Geopolitical layer

The geopolitical layer reads a `direction_score` field (range -1.0 to +1.0) from signals. It is a stub pending a connector that parses government statements, regulatory filings, or similar sources. You can feed it manually with the same approach as social above, using `"layer": "geopolitical"` and a `direction_score` in `processed_data`.

### Check layer scores

After ingestion, query the current score for any layer and topic:

```bash
GET /api/v1/layers/{layer_name}?topic=bitcoin
```

Where `layer_name` is one of: `market`, `news`, `sentiment`, `social`, `geopolitical`, `synthesis`.

---

## 3. Generate training data

Training data generation uses Claude to produce gold-standard briefings from live signals. You need:

- The server from step 1 running
- An Anthropic API key in `.env`

```dotenv
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

### Ingest signals first

```bash
curl -X POST http://localhost:8000/api/v1/ingest/run
```

This pulls the latest markets and signals from Polymarket into the database.

### Generate labelled examples

```bash
# Generate examples for one or more topics (repeat for different topics)
curl -X POST "http://localhost:8000/api/v1/training/generate?topic=US+election&split=train"
curl -X POST "http://localhost:8000/api/v1/training/generate?topic=Bitcoin+price&split=train"
curl -X POST "http://localhost:8000/api/v1/training/generate?topic=AI+regulation&split=train"

# Generate some validation examples
curl -X POST "http://localhost:8000/api/v1/training/generate?topic=Federal+Reserve&split=val"
```

Each call runs the six-layer pipeline on current signals, sends the result to Claude, and stores the Claude-generated briefing as a labelled training example.

Aim for **at least 50–100 training examples** before fine-tuning. The more diverse the topics, the better.

### Export the dataset to disk

```bash
# Downloads as a HuggingFace Dataset directory
curl http://localhost:8000/api/v1/training/export --output data/intuone-v1.zip
# Or just use the path returned — the API saves it to ./data/intuone-v1 by default
```

---

## 4. Train IntuOne

Training requires a GPU machine. If you do not have a local GPU, see [Cloud GPU options](#cloud-gpu-options) below.

### 3a. Install the ML stack

```bash
# Activate the same venv from step 1
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install -e ".[ml]"
```

This installs: `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`.

### 3b. Accept the Meta Llama license

1. Go to [huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
2. Sign in and accept the license
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Add it to `.env`:

```dotenv
HUGGING_FACE_HUB_TOKEN=hf_...
```

Or export it as an environment variable:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...
```

### 3c. Run training

```bash
python -m app.training.trainer \
    --dataset ./data/intuone-v1 \
    --output  ./checkpoints/intuone-v1 \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Training arguments** (all optional, defaults shown):

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `--dataset` | `./data/intuone-v1` | Path to the exported dataset |
| `--output` | `./checkpoints/intuone-v1` | Where to save the LoRA adapter |
| `--epochs` | `3` | Number of training epochs |
| `--lora-r` | `32` | LoRA rank (higher = more parameters) |
| `--lr` | `2e-4` | Learning rate |
| `--report-to` | `none` | Set to `wandb` for W&B logging |

**Estimated training time:**

| Hardware | Time (3 epochs, ~100 examples) |
|----------|-------------------------------|
| RTX 3090 / 4090 (24 GB) | ~2–3 hours |
| A100 (40/80 GB) | ~30–45 minutes |
| RTX 3080 (10 GB) | Not recommended (VRAM too low for 4-bit) |

Training uses QLoRA (4-bit NF4 quantisation + LoRA adapters), so the base model weights are not modified. The output is a small adapter directory (~300 MB) at `./checkpoints/intuone-v1/final_adapter/`.

### 3d. Enable the model

Add to `.env`:

```dotenv
LOAD_MODEL_ON_STARTUP=true
ADAPTER_PATH=./checkpoints/intuone-v1/final_adapter
BASE_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

Restart the server (`run.bat` / `run.sh`). On startup it will load the base model and apply the LoRA adapter. The `/health` endpoint will return `"model_loaded": true` when ready.

Chat responses will now come from IntuOne instead of the layer-summary fallback.

---

## Environment variable reference

All variables go in `.env` (copy from `.env.example`).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL URL | Use `sqlite+aiosqlite:///./parallellines.db` for SQLite |
| `DEBUG` | No | `false` | Enable debug logging |
| `SECRET_KEY` | No | dev key | Change in production |
| `ANTHROPIC_API_KEY` | Training only | — | For generating training data |
| `POLYMARKET_API_URL` | No | `https://clob.polymarket.com` | Polymarket CLOB endpoint |
| `GAMMA_API_URL` | No | `https://gamma-api.polymarket.com` | Polymarket Gamma endpoint |
| `INGEST_INTERVAL_SECONDS` | No | `300` | How often the background loop polls connectors |
| `BASE_MODEL_NAME` | Training/inference | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `ADAPTER_PATH` | Inference only | `./checkpoints/intuone-v1/final_adapter` | Path to the trained LoRA adapter |
| `LOAD_MODEL_ON_STARTUP` | No | `false` | Set `true` to load the model when the server starts |
| `HUGGING_FACE_HUB_TOKEN` | Training/inference | — | HF token for gated model downloads |

---

## Cloud GPU options

If you don't have a local GPU, you can run training on a rented machine:

| Provider | Recommended instance | Notes |
|----------|---------------------|-------|
| [RunPod](https://runpod.io) | RTX 4090 or A100 pod | Good hourly pricing |
| [Vast.ai](https://vast.ai) | Any 24 GB+ GPU | Cheapest option |
| [Lambda Labs](https://lambdalabs.com) | A100 instance | Reliable, good disk I/O |
| Google Colab Pro+ | A100 | Easiest if you already use Colab |

Workflow:
1. Start a GPU instance with at least 24 GB VRAM
2. Clone the repo and `pip install -e ".[ml]"`
3. Copy your `.env` (with HF token) to the instance
4. Run the training command
5. Download `./checkpoints/intuone-v1/final_adapter/` back to your machine (it's ~300 MB)
6. Point `ADAPTER_PATH` at it and restart your local server

---

## Troubleshooting

**Migrations fail with PostgreSQL**
- Confirm PostgreSQL is running: `pg_isready`
- Confirm the database exists: `psql -U postgres -c "\l"`
- Check the `DATABASE_URL` in `.env` matches your PostgreSQL credentials

**`ImportError` during training**
- Run `pip install -e ".[ml]"` — the ML stack is not installed by default

**CUDA out of memory**
- The trainer uses 4-bit quantisation, which requires ~10 GB VRAM minimum
- Reduce `per_device_train_batch_size` to `1` in `app/training/trainer.py`
- If your GPU has less than 16 GB, use a cloud instance instead

**Model loads but chat responses are slow**
- Inference on CPU is very slow. Make sure your GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`

**`401 Unauthorized` from Hugging Face**
- Accept the Meta Llama license at huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- Set `HUGGING_FACE_HUB_TOKEN` in `.env`

**Port 8000 already in use**
- Change the port: `uvicorn app.main:app --port 8001 --reload`
