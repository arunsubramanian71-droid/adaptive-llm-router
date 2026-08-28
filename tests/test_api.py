from __future__ import annotations

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def client():
    from router.api.app import app

    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_exposes_disclaimer_and_role_models(client):
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert "DEMO ONLY" in body["disclaimer"]
    assert body["strong_model_id"] == "claude-sonnet-5"
    assert body["cheap_model_id"] == "claude-haiku-4-5"
    assert body["tau"] == 0.5


def test_route_hard_prompt_selects_strong(client):
    response = client.post("/route", json={"prompt": "Prove that this algorithm terminates for every input."})
    assert response.status_code == 200
    body = response.json()
    assert body["selected_role"] == "strong"
    assert body["selected_model_id"] == "claude-sonnet-5"
    assert 0.0 <= body["probability"] <= 1.0
    assert body["estimated_cost_usd"] > 0
    assert "DEMO ONLY" in body["disclaimer"]


def test_route_easy_prompt_selects_cheap(client):
    response = client.post("/route", json={"prompt": "What is the capital of France?"})
    assert response.status_code == 200
    body = response.json()
    assert body["selected_role"] == "cheap"
    assert body["selected_model_id"] == "claude-haiku-4-5"


def test_route_rejects_empty_prompt(client):
    response = client.post("/route", json={"prompt": "   "})
    assert response.status_code == 400


def test_route_rejects_missing_prompt_field(client):
    response = client.post("/route", json={})
    assert response.status_code == 422


def test_strong_route_costs_more_than_cheap_route(client):
    strong = client.post("/route", json={"prompt": "Prove that this algorithm terminates."}).json()
    cheap = client.post("/route", json={"prompt": "What is the capital of France?"}).json()
    if strong["selected_role"] == "strong" and cheap["selected_role"] == "cheap":
        assert strong["estimated_cost_usd"] > cheap["estimated_cost_usd"]
