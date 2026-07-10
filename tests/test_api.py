from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE_REQUEST = {
    "koi_period": 9.48803557,
    "koi_duration": 2.9575,
    "koi_depth": 615.8,
    "koi_prad": 2.26,
    "koi_teq": 793.0,
    "koi_insol": 93.59,
    "koi_model_snr": 35.8,
    "koi_steff": 5455.0,
    "koi_slogg": 4.467,
    "koi_srad": 0.927,
    "koi_kepmag": 15.347,
    "koi_impact": 0.146,
}


def test_read_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_returns_prediction_and_confidence():
    response = client.post("/predict", json=SAMPLE_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}
    assert 0.0 <= body["confidence"] <= 1.0
