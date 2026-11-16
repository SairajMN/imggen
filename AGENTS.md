# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Antigravity) working in `imggen`.

## Project Overview

`imggen` is a lightweight FastAPI + vanilla HTML/JS thumbnail generation app backed by OpenRouter. Users upload a subject **Frame Image** (`IMAGE 1`), a layout **Reference Image** (`IMAGE 2`), and **Video Context**, then generate 9:16 (or other aspect ratio) thumbnails across multiple OpenRouter image models with strict per-session cost/call guardrails.

## Repository Map

- `app.py` — FastAPI application, `Settings` (`pydantic-settings`), `IMAGE_MODELS` catalog, `IMAGES_API_HINTS` routing, OpenRouter `/api/v1/chat/completions` + `/api/v1/images` clients, session persistence, and `/api/history` aggregation.
- `prompt.py` — `BASE_SYSTEM_PROMPT` (IP-agnostic template extraction rules), `extract_rgb_colors()` (local Pillow median-cut quantization, \$0.00 cost), and `build_prompt()`.
- `static/index.html` — Single-file dark UI (drag/drop upload slots, model dropdown, live preview, cost/call guardrail table, and History & Model Costs modal).
- `tests/test_app.py` — Pytest suite using `FastAPI.TestClient` and `monkeypatch` on `httpx.post`.
- `pyproject.toml` / `uv.lock` — Python >=3.11 dependencies managed with `uv`.
- `data/` — Runtime session storage (`data/sessions/<32-hex-id>/session.json`, `inputs/`, `outputs/`). Gitignored.

## Commands

```bash
# Install dependencies
uv sync

# Run dev server
uv run uvicorn app:app --port 8400

# Run test suite (always run after edits)
uv run pytest -q
```

## Architecture & Invariants

1. **Strict Guardrails Before Billing:**
   - Check `len(s["calls"]) >= settings.max_calls` (`429`) and `_spent(s) + settings.estimated_cost_per_image > settings.budget_usd` (`402`) **before** making any OpenRouter HTTP request.
   - Never add background retries that could incur unexpected charges; the only automatic retry in `_call_openrouter_images` is on a `400` content-moderation rejection (unbilled) when `ref_bytes` is attached, retrying once with `frame_bytes` only.

2. **OpenRouter Dual-Endpoint Routing (`app.py`):**
   - Models matching `IMAGES_API_HINTS` (`gpt-image`, `seedream`, `flux`, `qwen-image`, `riverflow`, `minimax`, `mai-image`, `krea`, `grok-imagine`) go directly to `POST https://openrouter.ai/api/v1/images` using `input_references`.
   - Other models (e.g. `google/gemini-*-image`, `recraft/recraft-v4.1-flash-v2`) go to `POST https://openrouter.ai/api/v1/chat/completions` with `"modalities": ["image"]`, and automatically fall back to `/api/v1/images` if OpenRouter returns `404` mentioning `/api/v1/images`.
   - `recraft-v4.1-flash` is treated as text-only (`rgb_colors` in `image_config`, no `image_url` parts).
   - Provider moderation passthroughs: `gpt-image` sets `"moderation": "low"`; `flux` sets `"safety_tolerance": 5`.
   - `_decode_image_output()` must support both `/api/v1/images` (`data[0].b64_json` / `data[0].url`) and `/api/v1/chat/completions` (`choices[0].message.images`).

3. **IP-Agnostic Prompt Design (`prompt.py`):**
   - `IMAGE 2` is strictly a layout/design blueprint. `BASE_SYSTEM_PROMPT` must instruct the model to extract spatial grid, layering, and container geometry while excluding `IMAGE 2`'s franchise logos, numerals, fonts, and scenery.
   - Avoid aggressive/violent phrasing in `BASE_SYSTEM_PROMPT` (e.g., use `CARRYOVER`, `EXCLUDE`, `presenter`, `facial features`) so strict upstream moderation filters (Seedream, BFL FLUX) are not unnecessarily triggered.

4. **Testing & Security Rules:**
   - Never make live paid OpenRouter calls in tests. Isolate `settings.data_dir`, `settings.openrouter_api_key`, and `settings.image_model` in fixtures and mock `httpx.post`.
   - Never commit `.env` or `data/`.
