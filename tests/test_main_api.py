"""Tests fuer API-Flow-Paritaet und Empty-Result-Robustheit."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _base_payload() -> dict:
    return {
        "grunddaten": {
            "vertical": "retail",
            "stadt_plz": "Berlin",
            "radius_km": 10,
            "isochrone_minuten": 10,
            "kandidaten_anzahl": 50,
            "top_n": 5,
            "seed": 42,
        },
        "optionen": {
            "preset": "balanced",
            "explainability": True,
            "sensitivitaetsanalyse": True,
            "zeige_baseline_vergleich": True,
        },
    }


def test_analyze_json_includes_optional_outputs_when_enabled():
    payload = _base_payload()

    response = client.post("/analyze/json", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "sensitivitaet" in data
    assert data["sensitivitaet"] is not None
    assert "baseline_vergleich" in data
    assert data["baseline_vergleich"] is not None


def test_analyze_json_handles_empty_results_cleanly():
    payload = _base_payload()
    payload["ziele"] = {
        "optimierungsziel": "roi_max",
        "mindest_roi_prozent": 95,
        "max_payback_monate": 6,
        "risikoappetit": "neutral",
    }

    response = client.post("/analyze/json", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["standorte"], list)
