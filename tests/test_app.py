import base64
import io
from pathlib import Path
import tempfile

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as imggen_app


def _png_bytes(color: tuple[int, int, int] = (20, 40, 120), size: tuple[int, int] = (128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(imggen_app.settings, "data_dir", Path(tempfile.mkdtemp(prefix="imggen-test-")))
    monkeypatch.setattr(imggen_app.settings, "openrouter_api_key", "sk-or-test")
    monkeypatch.setattr(imggen_app.settings, "image_model", "recraft/recraft-v4.1-flash-v2")
    monkeypatch.setattr(imggen_app.settings, "budget_usd", 0.015)
    monkeypatch.setattr(imggen_app.settings, "estimated_cost_per_image", 0.007)
    monkeypatch.setattr(imggen_app.settings, "max_calls", 5)


def _mock_openrouter(monkeypatch, out_png: bytes, cost: float = 0.007, captured: list[dict] | None = None):
    b64 = base64.b64encode(out_png).decode()

    def fake_post(url, headers=None, json=None, timeout=None):
        if captured is not None and json is not None:
            captured.append(json)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"images": [{"image_url": {"url": f"data:image/png;base64,{b64}"}}]}}],
                "usage": {"cost": cost},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)


def test_generate_download_and_budget_guardrail(monkeypatch):
    out_png = _png_bytes((255, 255, 255), (270, 480))
    captured: list[dict] = []
    _mock_openrouter(monkeypatch, out_png, cost=0.008, captured=captured)

    client = TestClient(imggen_app.app)
    s = client.post("/api/sessions").json()
    sid = s["id"]

    r1 = client.post(
        f"/api/sessions/{sid}/generate",
        data={"context": "Saros PS5 launch", "message": "make it like ref", "aspect_ratio": "9:16"},
        files={
            "frame": ("frame.png", _png_bytes((10, 20, 30)), "image/png"),
            "ref": ("ref.png", _png_bytes((220, 230, 250)), "image/png"),
        },
    )
    assert r1.status_code == 200
    assert captured[0]["modalities"] == ["image"]
    body = r1.json()
    assert body["calls_used"] == 1
    assert body["spent_usd"] == pytest.approx(0.008)
    assert len(body["outputs"]) == 1

    dl = client.get(f"{body['outputs'][0]}?download=1")
    assert dl.status_code == 200
    assert dl.headers["content-type"] == "image/png"
    assert "attachment" in dl.headers["content-disposition"]

    # Budget is 0.015, spent is 0.008, estimated next is 0.007 -> 0.015 <= 0.015 allowed once more
    r2 = client.post(f"/api/sessions/{sid}/generate", data={"context": "second try", "aspect_ratio": "9:16"})
    assert r2.status_code == 200

    # 3rd call exceeds 0.015 budget -> blocked before calling OpenRouter
    r3 = client.post(f"/api/sessions/{sid}/generate", data={"context": "third try", "aspect_ratio": "9:16"})
    assert r3.status_code == 402
    assert "Budget cap reached" in r3.json()["detail"]


def test_max_calls_guardrail(monkeypatch):
    monkeypatch.setattr(imggen_app.settings, "budget_usd", 1.0)
    monkeypatch.setattr(imggen_app.settings, "max_calls", 1)
    _mock_openrouter(monkeypatch, _png_bytes(), cost=0.007)

    client = TestClient(imggen_app.app)
    sid = client.post("/api/sessions").json()["id"]

    assert client.post(f"/api/sessions/{sid}/generate", data={"context": "one"}).status_code == 200
    r2 = client.post(f"/api/sessions/{sid}/generate", data={"context": "two"})
    assert r2.status_code == 429
    assert "Call limit reached" in r2.json()["detail"]


def test_fallback_order_when_default_fails(monkeypatch):
    monkeypatch.setattr(imggen_app.settings, "image_model", "openai/gpt-image-2")
    monkeypatch.setattr(imggen_app.settings, "budget_usd", 1.0)
    monkeypatch.setattr(imggen_app.settings, "max_calls", 5)

    tried = []
    b64 = base64.b64encode(_png_bytes((200, 100, 50))).decode()

    def fake_post(url, headers=None, json=None, timeout=None):
        model = json["model"]
        tried.append(model)
        if model in ("openai/gpt-image-2", "minimax/minimax-h3-image"):
            return httpx.Response(502, json={"error": {"message": f"{model} unavailable"}})
        return httpx.Response(200, json={"data": [{"b64_json": b64}], "usage": {"cost": 0.01}})

    monkeypatch.setattr(httpx, "post", fake_post)

    client = TestClient(imggen_app.app)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(
        f"/api/sessions/{sid}/generate",
        data={"context": "Saros launch", "message": "make it look like this"},
        files={"ref": ("ref.png", _png_bytes((240, 240, 240)), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert tried == ["openai/gpt-image-2", "minimax/minimax-h3-image", "openai/gpt-image-2.5-sunburst"]
    assert body["calls_used"] == 1
    assert body["calls"][-1]["model"] == "openai/gpt-image-2.5-sunburst"
    assert body["calls"][-1]["ok"] is True
    assert "fallback" in body["messages"][-1]["text"]

    # Explicit non-default model does not trigger fallback
    tried.clear()
    r2 = client.post(
        f"/api/sessions/{sid}/generate",
        data={"context": "Saros launch", "model": "minimax/minimax-h3-image"},
    )
    assert r2.status_code == 502
    assert tried == ["minimax/minimax-h3-image"]


def test_fallback_order_when_default_fails(monkeypatch):
    monkeypatch.setattr(imggen_app.settings, "image_model", "openai/gpt-image-2")
    monkeypatch.setattr(imggen_app.settings, "budget_usd", 1.0)
    monkeypatch.setattr(imggen_app.settings, "max_calls", 5)

    tried = []
    b64 = base64.b64encode(_png_bytes((200, 100, 50))).decode()

    def fake_post(url, headers=None, json=None, timeout=None):
        model = json["model"]
        tried.append(model)
        if model in ("openai/gpt-image-2", "minimax/minimax-h3-image"):
            return httpx.Response(502, json={"error": {"message": f"{model} unavailable"}})
        return httpx.Response(200, json={"data": [{"b64_json": b64}], "usage": {"cost": 0.01}})

    monkeypatch.setattr(httpx, "post", fake_post)

    client = TestClient(imggen_app.app)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(
        f"/api/sessions/{sid}/generate",
        data={"context": "Saros launch", "message": "make it look like this"},
        files={"ref": ("ref.png", _png_bytes((240, 240, 240)), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert tried == ["openai/gpt-image-2", "minimax/minimax-h3-image", "openai/gpt-image-2.5-sunburst"]
    assert body["calls_used"] == 1
    assert body["calls"][-1]["model"] == "openai/gpt-image-2.5-sunburst"
    assert body["calls"][-1]["ok"] is True
    assert "fallback" in body["messages"][-1]["text"]


def test_images_endpoint_for_gpt_image_and_fallback(monkeypatch):
    out_png = _png_bytes((12, 34, 56), (128, 128))
    b64 = base64.b64encode(out_png).decode()
    calls: list[tuple[str, dict]] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        if url == imggen_app.OPENROUTER_URL:
            return httpx.Response(
                404,
                json={"error": {"message": "Use the /api/v1/images endpoint instead.", "code": 404}},
            )
        return httpx.Response(
            200,
            json={"data": [{"b64_json": b64, "media_type": "image/png"}], "usage": {"cost": 0.01}},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(imggen_app.settings, "budget_usd", 1.0)
    monkeypatch.setattr(imggen_app.settings, "image_model", "openai/gpt-image-2.5-sunburst")

    client = TestClient(imggen_app.app)
    sid = client.post("/api/sessions").json()["id"]

    r1 = client.post(
        f"/api/sessions/{sid}/generate",
        data={"context": "sunset", "aspect_ratio": "9:16"},
        files={"frame": ("frame.png", _png_bytes((10, 20, 30)), "image/png")},
    )
    assert r1.status_code == 200
    assert calls[0][0] == imggen_app.OPENROUTER_IMAGES_URL
    assert len(calls[0][1]["input_references"]) == 1

    # Non-gpt-image model that 404s on chat/completions falls back to /api/v1/images
    calls.clear()
    monkeypatch.setattr(imggen_app.settings, "image_model", "some/images-only-model")
    r2 = client.post(f"/api/sessions/{sid}/generate", data={"context": "sunset", "aspect_ratio": "1:1"})
    assert r2.status_code == 200
    assert [c[0] for c in calls] == [imggen_app.OPENROUTER_URL, imggen_app.OPENROUTER_IMAGES_URL]


def test_prompt_strips_reference_ip():
    from prompt import build_prompt

    p = build_prompt("Street Fighter 6 launch at Sarjapur store", has_frame=True, has_ref=True, ref_colors=[[10, 20, 30]])
    assert "ZERO REFERENCE IP CARRYOVER" in p
    assert "EXCLUDE ALL REFERENCE IP, LOGOS & SYMBOLS" in p
    assert "REFERENCE PALETTE HINT" in p


def test_model_dropdown_selection(monkeypatch):
    out_png = _png_bytes((12, 34, 56), (128, 128))
    b64 = base64.b64encode(out_png).decode()
    calls: list[tuple[str, dict]] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return httpx.Response(
            200,
            json={"data": [{"b64_json": b64, "media_type": "image/png"}], "usage": {"cost": 0.02}},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(imggen_app.settings, "budget_usd", 1.0)

    client = TestClient(imggen_app.app)
    cfg = client.get("/api/config").json()
    model_ids = [m["id"] for m in cfg["models"]]
    assert "bytedance-seed/seedream-5-5" in model_ids
    assert "bytedance-seed/seedream-5-0-flash" in model_ids
    assert "minimax/minimax-h3-image" in model_ids
    assert "qwen/qwen-image-2.1-turbo" in model_ids
    assert "krea/k3-edit" in model_ids
    for expected in (
        "x-ai/grok-imagine-image-2.0",
        "qwen/qwen-image-3",
        "krea/krea-2-medium-turbo",
        "openai/gpt-image-2",
        "sourceful/riverflow-v2.5-fast",
        "google/gemini-2.5-flash-image",
    ):
        assert expected in model_ids

    sid = client.post("/api/sessions").json()["id"]
    r = client.post(
        f"/api/sessions/{sid}/generate",
        data={"context": "Street Fighter 6", "model": "bytedance-seed/seedream-5-5", "aspect_ratio": "9:16"},
    )
    assert r.status_code == 200
    assert calls[0][0] == imggen_app.OPENROUTER_IMAGES_URL
    assert calls[0][1]["model"] == "bytedance-seed/seedream-5-5"
    assert r.json()["calls"][0]["model"] == "bytedance-seed/seedream-5-5"


def test_history_aggregates_models_and_costs(monkeypatch):
    _mock_openrouter(monkeypatch, _png_bytes((20, 30, 40)), cost=0.015)
    monkeypatch.setattr(imggen_app.settings, "budget_usd", 1.0)

    client = TestClient(imggen_app.app)
    s1 = client.post("/api/sessions").json()["id"]
    s2 = client.post("/api/sessions").json()["id"]

    assert client.post(
        f"/api/sessions/{s1}/generate",
        data={"context": "Street Fighter 6", "model": "recraft/recraft-v4.1-flash-v2"},
    ).status_code == 200
    assert client.post(
        f"/api/sessions/{s2}/generate",
        data={"context": "Tekken 8", "model": "google/gemini-3.1-flash-image"},
    ).status_code == 200

    hist = client.get("/api/history").json()
    assert hist["totals"]["total_sessions"] == 2
    assert hist["totals"]["total_generations"] == 2
    assert hist["totals"]["total_spent_usd"] == pytest.approx(0.03)
    assert len(hist["by_model"]) == 2
    assert len(hist["items"]) == 2
    assert hist["items"][0]["output_url"] is not None


def test_default_model_fallback_order(monkeypatch):
    out_png = _png_bytes((40, 50, 60), (128, 128))
    b64 = base64.b64encode(out_png).decode()
    tried: list[str] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        m = json["model"]
        tried.append(m)
        if m in ("openai/gpt-image-2", "minimax/minimax-h3-image"):
            return httpx.Response(502, json={"error": {"message": f"{m} unavailable"}})
        return httpx.Response(200, json={"data": [{"b64_json": b64}], "usage": {"cost": 0.03}})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(imggen_app.settings, "budget_usd", 1.0)
    monkeypatch.setattr(imggen_app.settings, "image_model", "openai/gpt-image-2")

    client = TestClient(imggen_app.app)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/generate", data={"context": "Street Fighter 6"})
    assert r.status_code == 200
    assert tried == ["openai/gpt-image-2", "minimax/minimax-h3-image", "openai/gpt-image-2.5-sunburst"]
    assert r.json()["calls"][-1]["model"] == "openai/gpt-image-2.5-sunburst"

    # Explicit non-default model does not trigger fallback
    tried.clear()
    r2 = client.post(
        f"/api/sessions/{sid}/generate",
        data={"context": "Street Fighter 6", "model": "minimax/minimax-h3-image"},
    )
    assert r2.status_code == 502
    assert tried == ["minimax/minimax-h3-image"]
