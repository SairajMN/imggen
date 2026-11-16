"""FastAPI backend for imggen: OpenRouter image generation with strict budget & call caps."""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic_settings import BaseSettings, SettingsConfigDict

from prompt import build_prompt, extract_rgb_colors

ROOT = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"
VALID_RATIOS = ("9:16", "1:1", "16:9", "3:4", "4:3")
DEFAULT_IMAGE_MODEL = "openai/gpt-image-2"
FALLBACK_MODELS = ("minimax/minimax-h3-image", "openai/gpt-image-2.5-sunburst")
IMAGE_MODELS = [
    {"id": "openai/gpt-image-2", "label": "OpenAI GPT Image 2 (Default · Frame + Ref)"},
    {"id": "google/gemini-3.1-flash-image", "label": "Google Gemini 3.1 Flash Image / Nano Banana 2 (Frame + Ref)"},
    {"id": "google/gemini-2.5-flash-image", "label": "Google Gemini 2.5 Flash Image / Nano Banana (Frame + Ref)"},
    {"id": "google/gemini-nano-banana-2.1", "label": "Google Nano Banana 2.1 (Frame + Ref)"},
    {"id": "google/gemini-3-pro-image", "label": "Google Gemini 3 Pro Image / Nano Banana Pro"},
    {"id": "openai/gpt-image-2.6-sunburst", "label": "OpenAI GPT Image 2.6 Sunburst (Frame + Ref)"},
    {"id": "openai/gpt-image-2.5-sunburst", "label": "OpenAI GPT Image 2.5 Sunburst (Fallback · Frame + Ref)"},
    {"id": "openai/gpt-image-2.5-flare", "label": "OpenAI GPT Image 2.5 Flare (Fast Frame + Ref)"},
    {"id": "x-ai/grok-imagine-image-2.0", "label": "xAI Grok Imagine Image 2.0 (Frame + Ref)"},
    {"id": "black-forest-labs/flux-3-image", "label": "BFL FLUX 3 Image (Frame + Ref · $0.048)"},
    {"id": "black-forest-labs/flux-3-image-klein", "label": "BFL FLUX 3 Image Klein (Fast · $0.009)"},
    {"id": "qwen/qwen-image-3-pro", "label": "Qwen Image 3 Pro (Frame + Ref · $0.04)"},
    {"id": "qwen/qwen-image-3", "label": "Qwen Image 3 (Frame + Ref)"},
    {"id": "qwen/qwen-image-2.1-turbo", "label": "Qwen Image 2.1 Turbo (Fast Frame + Ref)"},
    {"id": "minimax/minimax-h3-image", "label": "MiniMax H3 Image (Fallback · Frame + Ref)"},
    {"id": "krea/k3-edit", "label": "Krea K3 Edit (Frame + Ref)"},
    {"id": "krea/krea-2-medium-turbo", "label": "Krea 2 Medium Turbo (Fast Frame + Ref)"},
    {"id": "sourceful/riverflow-v2.5-fast", "label": "Sourceful Riverflow V2.5 Fast (Frame + Ref)"},
    {"id": "recraft/recraft-v4.1-flash-v2", "label": "Recraft V4.1 Flash V2 (Text + Palette · $0.007)"},
    {"id": "bytedance-seed/seedream-5-5", "label": "ByteDance Seedream 5.5 (Strict Face/IP Filter · $0.02)"},
    {"id": "bytedance-seed/seedream-5-0-flash", "label": "ByteDance Seedream 5.0 Flash (Strict Face/IP Filter)"},
    {"id": "bytedance-seed/seedream-5-0-pro", "label": "ByteDance Seedream 5.0 Pro (Strict Face/IP Filter · $0.045)"},
]
IMAGES_API_HINTS = ("gpt-image", "seedream", "flux", "qwen-image", "riverflow", "minimax", "mai-image", "krea", "grok-imagine")
MAX_IMAGE_MB = 20
LOCK = threading.Lock()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openrouter_api_key: str = ""
    image_model: str = "openai/gpt-image-2"
    estimated_cost_per_image: float = 0.007
    budget_usd: float = 0.10
    max_calls: int = 10
    data_dir: Path = ROOT / "data"


settings = Settings()
app = FastAPI(title="imggen", version="0.1.0")


def _session_dir(sid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", sid):
        raise HTTPException(404, "Session not found")
    return settings.data_dir / "sessions" / sid


def _load(sid: str) -> dict[str, Any]:
    p = _session_dir(sid) / "session.json"
    if not p.exists():
        raise HTTPException(404, "Session not found")
    return json.loads(p.read_text())


def _save(s: dict[str, Any]) -> None:
    d = _session_dir(s["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps(s, indent=2))


def _spent(s: dict[str, Any]) -> float:
    return round(sum(float(c.get("cost_usd", 0.0)) for c in s.get("calls", [])), 6)


def _save_png(raw: bytes, dest: Path) -> None:
    if len(raw) > MAX_IMAGE_MB << 20:
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_MB} MB limit")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:
        raise HTTPException(422, "Invalid image file (use PNG, JPEG, or WebP)") from e
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(dest, format="PNG")


def _decode_image_output(body: dict[str, Any]) -> bytes:
    if "data" in body:
        items = body.get("data") or []
        if not items:
            raise RuntimeError("Model returned no image in response")
        b64 = items[0].get("b64_json")
        if b64:
            return base64.b64decode(b64)
        url = items[0].get("url") or ""
    else:
        msg = body["choices"][0]["message"]
        images = msg.get("images") or []
        if not images:
            raise RuntimeError("Model returned no image in response")
        url = images[0].get("image_url", {}).get("url") or images[0].get("url") or ""
    if url.startswith("data:image/"):
        b64 = url.split(",", 1)[1]
        return base64.b64decode(b64)
    if url.startswith("http://") or url.startswith("https://"):
        r = httpx.get(url, timeout=60)
        r.raise_for_status()
        return r.content
    raise RuntimeError("Unsupported image output format from OpenRouter")


def _call_openrouter_images(
    headers: dict[str, str],
    model: str,
    prompt_text: str,
    aspect_ratio: str,
    frame_bytes: bytes | None,
    ref_bytes: bytes | None,
) -> httpx.Response:
    def _post(f_bytes: bytes | None, r_bytes: bytes | None) -> httpx.Response:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt_text,
            "aspect_ratio": aspect_ratio,
        }
        if "gpt-image" in model:
            payload["moderation"] = "low"
        elif "flux" in model:
            payload["safety_tolerance"] = 5
        refs = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(raw).decode()}"}}
            for raw in (f_bytes, r_bytes)
            if raw
        ]
        if refs:
            payload["input_references"] = refs
        return httpx.post(OPENROUTER_IMAGES_URL, headers=headers, json=payload, timeout=120)

    r = _post(frame_bytes, ref_bytes)
    if r.status_code == 400 and "moderation" in r.text.lower() and ref_bytes:
        # Reference thumbnail (e.g. GTA artwork/gestures) often trips provider image filters; retry with frame only
        r = _post(frame_bytes, None)
    return r


def _request_openrouter_once(
    headers: dict[str, str],
    model: str,
    prompt_text: str,
    aspect_ratio: str,
    frame_bytes: bytes | None,
    ref_bytes: bytes | None,
    ref_colors: list[list[int]] | None,
) -> tuple[bytes, float, str]:
    is_recraft_text_only = "recraft-v4.1-flash" in model

    if any(k in model for k in IMAGES_API_HINTS):
        r = _call_openrouter_images(headers, model, prompt_text, aspect_ratio, frame_bytes, ref_bytes)
    else:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        if not is_recraft_text_only:
            if frame_bytes:
                b64 = base64.b64encode(frame_bytes).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            if ref_bytes:
                b64 = base64.b64encode(ref_bytes).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        image_config: dict[str, Any] = {"aspect_ratio": aspect_ratio}
        if is_recraft_text_only and ref_colors:
            image_config["rgb_colors"] = ref_colors

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content if len(content) > 1 else prompt_text}],
            "modalities": ["image"],
            "image_config": image_config,
        }
        r = httpx.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        if r.status_code == 404 and "/api/v1/images" in r.text:
            r = _call_openrouter_images(headers, model, prompt_text, aspect_ratio, frame_bytes, ref_bytes)

    if r.status_code >= 400:
        try:
            err_msg = r.json().get("error", {}).get("message") or r.text[:300]
        except Exception:
            err_msg = r.text[:300]
        if "moderation" in err_msg.lower():
            raise HTTPException(
                422,
                f"{model} blocked by upstream content moderation ({err_msg}). "
                "This provider's safety filter rejects real faces or branded game IP—switch to Google Gemini 3.1 Flash Image, Google Nano Banana 2.1, or OpenAI GPT Image 2.5.",
            )
        raise HTTPException(502, f"OpenRouter error ({r.status_code}): {err_msg}")
    body = r.json()
    img_bytes = _decode_image_output(body)
    usage = body.get("usage") or {}
    raw_cost = usage.get("cost") if usage.get("cost") is not None else body.get("total_cost")
    cost = round(float(raw_cost), 6) if raw_cost is not None else settings.estimated_cost_per_image
    note = (
        "Generated with Recraft V4.1 Flash V2 (text + extracted reference palette)."
        if is_recraft_text_only
        else f"Generated with {model}."
    )
    return img_bytes, cost, note


def _call_openrouter(
    prompt_text: str,
    *,
    model: str | None = None,
    aspect_ratio: str,
    frame_bytes: bytes | None,
    ref_bytes: bytes | None,
    ref_colors: list[list[int]] | None,
) -> tuple[bytes, float, str, str]:
    if not settings.openrouter_api_key:
        raise HTTPException(400, "OPENROUTER_API_KEY is not set in .env")

    chosen = (model or "").strip() or settings.image_model
    candidates = [chosen, *(m for m in FALLBACK_MODELS if m != chosen)] if chosen == settings.image_model else [chosen]
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    last_err: HTTPException | None = None
    for idx, candidate in enumerate(candidates):
        try:
            img_bytes, cost, note = _request_openrouter_once(
                headers, candidate, prompt_text, aspect_ratio, frame_bytes, ref_bytes, ref_colors
            )
            if idx > 0:
                note = f"{note} (fallback from {chosen})"
            return img_bytes, cost, note, candidate
        except HTTPException as exc:
            last_err = exc
    assert last_err is not None
    raise last_err


def _out(s: dict[str, Any]) -> dict[str, Any]:
    return {
        **s,
        "spent_usd": _spent(s),
        "budget_usd": settings.budget_usd,
        "max_calls": settings.max_calls,
        "calls_used": len(s.get("calls", [])),
        "model": settings.image_model,
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    models = list(IMAGE_MODELS)
    if settings.image_model and all(m["id"] != settings.image_model for m in models):
        models.insert(0, {"id": settings.image_model, "label": f"{settings.image_model} (.env default)"})
    return {
        "model": settings.image_model,
        "models": models,
        "budget_usd": settings.budget_usd,
        "max_calls": settings.max_calls,
        "estimated_cost_per_image": settings.estimated_cost_per_image,
        "has_key": bool(settings.openrouter_api_key),
        "aspect_ratios": list(VALID_RATIOS),
    }


@app.post("/api/sessions", status_code=201)
def create_session() -> dict[str, Any]:
    sid = uuid.uuid4().hex
    s = {
        "id": sid,
        "created_at": datetime.now(UTC).isoformat(),
        "calls": [],
        "messages": [],
        "outputs": [],
        "has_frame": False,
        "has_ref": False,
    }
    with LOCK:
        _save(s)
    return _out(s)


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict[str, Any]:
    return _out(_load(sid))


@app.get("/api/history")
def get_history() -> dict[str, Any]:
    sessions_dir = settings.data_dir / "sessions"
    items: list[dict[str, Any]] = []
    by_model_map: dict[str, dict[str, Any]] = {}
    session_count = 0

    if sessions_dir.exists():
        for p in sessions_dir.glob("*/session.json"):
            try:
                s = json.loads(p.read_text())
            except Exception:
                continue
            session_count += 1
            sid = s.get("id", p.parent.name)
            outputs = s.get("outputs") or []
            calls = s.get("calls") or []
            user_msgs = [m for m in (s.get("messages") or []) if m.get("role") == "user"]

            out_idx = 0
            for idx, c in enumerate(calls):
                model = c.get("model") or settings.image_model
                cost = round(float(c.get("cost_usd", 0.0)), 6)
                ok = c.get("ok", True)
                um = user_msgs[out_idx] if out_idx < len(user_msgs) else (user_msgs[-1] if user_msgs else {})
                output_url = outputs[out_idx] if ok and out_idx < len(outputs) else None
                if ok:
                    out_idx += 1

                items.append({
                    "session_id": sid,
                    "call_index": c.get("index", idx + 1),
                    "model": model,
                    "aspect_ratio": c.get("aspect_ratio", ""),
                    "cost_usd": cost,
                    "at": c.get("at") or s.get("created_at", ""),
                    "output_url": output_url,
                    "frame_url": um.get("frame_url"),
                    "ref_url": um.get("ref_url"),
                    "prompt_summary": (um.get("text") or "").strip(),
                })

                agg = by_model_map.setdefault(model, {"model": model, "count": 0, "total_cost_usd": 0.0})
                agg["count"] += 1
                agg["total_cost_usd"] = round(agg["total_cost_usd"] + cost, 6)

    items.sort(key=lambda x: x["at"], reverse=True)
    by_model = sorted(
        (
            {
                **m,
                "avg_cost_usd": round(m["total_cost_usd"] / m["count"], 6) if m["count"] else 0.0,
            }
            for m in by_model_map.values()
        ),
        key=lambda x: (-x["total_cost_usd"], -x["count"]),
    )
    total_spent = round(sum(m["total_cost_usd"] for m in by_model), 6)
    return {
        "totals": {
            "total_spent_usd": total_spent,
            "total_generations": len(items),
            "total_sessions": session_count,
        },
        "by_model": by_model,
        "items": items,
    }


@app.post("/api/sessions/{sid}/generate")
async def generate(
    sid: str,
    context: str = Form(""),
    message: str = Form(""),
    aspect_ratio: str = Form("9:16"),
    model: str = Form(""),
    frame: UploadFile | None = File(None),
    ref: UploadFile | None = File(None),
) -> dict[str, Any]:
    if aspect_ratio not in VALID_RATIOS:
        raise HTTPException(422, f"aspect_ratio must be one of {VALID_RATIOS}")
    chosen_model = model.strip() or settings.image_model

    s = _load(sid)
    d = _session_dir(sid)

    if len(s["calls"]) >= settings.max_calls:
        raise HTTPException(429, f"Call limit reached ({settings.max_calls}/{settings.max_calls} calls used).")
    if _spent(s) + settings.estimated_cost_per_image > settings.budget_usd:
        raise HTTPException(
            402,
            f"Budget cap reached (${_spent(s):.4f} used of ${settings.budget_usd:.2f}; next call est. ${settings.estimated_cost_per_image:.4f}).",
        )

    frame_path, ref_path = d / "inputs" / "frame.png", d / "inputs" / "ref.png"
    if frame is not None:
        _save_png(await frame.read(MAX_IMAGE_MB * (1 << 20) + 1), frame_path)
        s["has_frame"] = True
    if ref is not None:
        _save_png(await ref.read(MAX_IMAGE_MB * (1 << 20) + 1), ref_path)
        s["has_ref"] = True

    frame_bytes = frame_path.read_bytes() if frame_path.exists() else None
    ref_bytes = ref_path.read_bytes() if ref_path.exists() else None
    ref_colors = extract_rgb_colors(ref_bytes) if ref_bytes else None

    if not (context.strip() or message.strip() or frame_bytes or ref_bytes):
        raise HTTPException(422, "Provide video context, a message, or at least one image.")

    prompt_text = build_prompt(
        context,
        message,
        has_frame=frame_bytes is not None,
        has_ref=ref_bytes is not None,
        ref_colors=ref_colors,
    )

    user_text = message.strip() or context.strip() or "Generate thumbnail from attached images."
    s["messages"].append({
        "role": "user",
        "text": user_text,
        "frame_url": f"/api/sessions/{sid}/inputs/frame.png" if frame_bytes else None,
        "ref_url": f"/api/sessions/{sid}/inputs/ref.png" if ref_bytes else None,
        "at": datetime.now(UTC).isoformat(),
    })

    img_bytes, cost_usd, note, used_model = _call_openrouter(
        prompt_text,
        model=chosen_model,
        aspect_ratio=aspect_ratio,
        frame_bytes=frame_bytes,
        ref_bytes=ref_bytes,
        ref_colors=ref_colors,
    )

    out_name = f"output-{len(s['outputs']) + 1}.png"
    out_path = d / "outputs" / out_name
    _save_png(img_bytes, out_path)
    out_url = f"/api/sessions/{sid}/outputs/{out_name}"

    s["outputs"].append(out_url)
    s["calls"].append({
        "index": len(s["calls"]) + 1,
        "model": used_model,
        "aspect_ratio": aspect_ratio,
        "cost_usd": cost_usd,
        "ok": True,
        "at": datetime.now(UTC).isoformat(),
    })
    s["messages"].append({
        "role": "assistant",
        "text": note,
        "image_url": out_url,
        "cost_usd": cost_usd,
        "model": used_model,
        "at": datetime.now(UTC).isoformat(),
    })
    with LOCK:
        _save(s)
    return _out(s)


@app.get("/api/sessions/{sid}/outputs/{name}")
def get_output(sid: str, name: str, download: bool = False) -> FileResponse:
    if not re.fullmatch(r"output-[0-9]+\.png", name):
        raise HTTPException(404, "Not found")
    p = _session_dir(sid) / "outputs" / name
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(
        p,
        media_type="image/png",
        filename=f"imggen-{sid[:8]}-{name}",
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/api/sessions/{sid}/inputs/{name}")
def get_input(sid: str, name: str) -> FileResponse:
    if name not in ("frame.png", "ref.png"):
        raise HTTPException(404, "Not found")
    p = _session_dir(sid) / "inputs" / name
    if not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="image/png")
