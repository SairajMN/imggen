# imggen

Minimal black-and-white thumbnail generation studio powered by **FastAPI**, **OpenRouter**, and a single-file vanilla JS UI (`static/index.html`), with strict pre-call budget/call-count guardrails, multi-model dropdown selection, IP-agnostic template extraction, and a cross-session Generation History & Model Cost dashboard.

---

## Architecture

```mermaid
flowchart TD
    UI["static/index.html (Dark UI + History Modal)"] -->|POST /api/sessions/sid/generate| API["app.py (FastAPI)"]
    API --> Guard{"Budget & Call Guardrails"}
    Guard -->|calls >= max_calls| E429["429 Call limit reached"]
    Guard -->|spent + est > budget| E402["402 Budget cap reached"]
    Guard -->|OK| Prompt["prompt.py (build_prompt + extract_rgb_colors)"]

    Prompt --> Route{"Model in IMAGES_API_HINTS?"}
    Route -->|Yes| ImgAPI["POST openrouter.ai/api/v1/images"]
    Route -->|No| ChatAPI["POST openrouter.ai/api/v1/chat/completions"]
    ChatAPI -->|404: use /api/v1/images| ImgAPI

    ImgAPI -->|400 moderation + ref| RetryNoRef["Retry once with frame only"]
    ImgAPI -->|200 OK| Save["Save PNG + session.json in data/sessions/sid/"]
    RetryNoRef -->|200 OK| Save
    ChatAPI -->|200 OK| Save
```

### Request Sequence

```mermaid
sequenceDiagram
    participant U as Browser UI
    participant F as FastAPI (app.py)
    participant P as prompt.py
    participant O as OpenRouter API

    U->>F: POST /api/sessions/{sid}/generate (frame, ref, context, model)
    F->>F: Check max_calls (429) & budget_usd (402)
    F->>P: extract_rgb_colors(ref) + build_prompt(...)
    alt Model in IMAGES_API_HINTS
        F->>O: POST /api/v1/images (prompt, aspect_ratio, input_references)
        opt 400 moderation with ref attached
            F->>O: Retry POST /api/v1/images (frame only)
        end
    else Chat-completions image model
        F->>O: POST /api/v1/chat/completions (modalities: ["image"])
        opt 404 directing to /api/v1/images
            F->>O: Fallback POST /api/v1/images
        end
    end
    O-->>F: Image (b64_json / data URL / HTTP URL) + usage.cost
    F->>F: Save output-N.png & update session.json
    F-->>U: Updated session JSON
```

---

## Setup & Run

Requires **Python >= 3.11** and [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env   # add OPENROUTER_API_KEY and optional defaults
uv sync
uv run uvicorn app:app --port 8400
```

Open `http://127.0.0.1:8400`:
1. Attach **Frame Image** (subject/presenter) and **Reference Image** (layout blueprint).
2. Paste **Video Context** and optional **Instruction**.
3. Pick a model from the dropdown and **Aspect Ratio**, then click **Generate**.
4. Click **History** in the header to view total spend, per-model average/total costs, filterable output gallery, and resume any past session.

---

## Configuration (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | `""` | Required OpenRouter API key |
| `IMAGE_MODEL` | `recraft/recraft-v4.1-flash-v2` | Default model selected on startup |
| `ESTIMATED_COST_PER_IMAGE` | `0.007` | Pre-call budget check & fallback cost when `usage.cost` is absent |
| `BUDGET_USD` | `0.10` | Per-session hard spend cap (`spent + est <= BUDGET_USD`) |
| `MAX_CALLS` | `10` | Per-session hard call limit |
| `DATA_DIR` | `./data` | Directory where `sessions/<sid>/` artifacts are stored |

---

## API Endpoints

| Method & Path | Purpose |
| :--- | :--- |
| `GET /` | Serves `static/index.html` |
| `GET /api/config` | Returns default model, `IMAGE_MODELS` list, budget/call caps, key status, and aspect ratios |
| `POST /api/sessions` | Creates a new session under `data/sessions/<sid>/` |
| `GET /api/sessions/{sid}` | Returns session state with computed `spent_usd` and `calls_used` |
| `POST /api/sessions/{sid}/generate` | Validates caps, builds prompt, calls OpenRouter, saves PNG, updates session |
| `GET /api/history` | Aggregates `totals`, `by_model` costs, and newest-first `items` across all sessions |
| `GET /api/sessions/{sid}/outputs/{name}` | Serves generated PNG (`?download=1` for attachment) |
| `GET /api/sessions/{sid}/inputs/{name}` | Serves uploaded `frame.png` or `ref.png` |

---

## Tests

All tests mock `httpx.post` and isolate `DATA_DIR` in a temporary directory (zero live API spend):

```bash
uv run pytest -q
```
