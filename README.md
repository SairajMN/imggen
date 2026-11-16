# imggen

Minimal black-and-white chatbot UI for thumbnail generation via **OpenRouter** (`recraft/recraft-v4.1-flash-v2` by default) with strict pre-call budget and call-count guardrails.

## Setup & Run

```bash
cd /Users/sahil/Desktop/imggen
cp .env.example .env   # add your OPENROUTER_API_KEY
uv sync
uv run uvicorn app:app --port 8400
```

Open `http://127.0.0.1:8400`, attach your **Frame Image** and **Reference Image**, paste **Video Context**, and click **Generate**.

## Cost & Call Guardrails

- **1 click = 1 API call** (no automatic retries or background calls).
- **`BUDGET_USD`** (default `0.10`): checked before every call (`spent + ESTIMATED_COST_PER_IMAGE <= BUDGET_USD`).
- **`MAX_CALLS`** (default `10`): hard per-session limit.
- **Swap model for multimodal image-in**: `recraft/recraft-v4.1-flash-v2` is text-to-image only (uses `rgb_colors` extracted locally in `prompt.py` at `$0.00`). To pass input image pixels directly, set `IMAGE_MODEL` in `.env` to a multimodal OpenRouter image model.

## Tests

```bash
uv run pytest -q
```
