BASE = {
    "network_packet_size": 599,
    "protocol_type": "TCP",
    "login_attempts": 4,
    "session_duration": 492.98,
    "encryption_used": "DES",
    "ip_reputation_score": 0.607,
    "failed_logins": 1,
    "browser_type": "Edge",
    "unusual_time_access": 0,
}


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid_input(api_client):
    r = api_client.post("/predict", json=BASE)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"prediction", "label", "probability"}
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_label_matches_prediction(api_client):
    r = api_client.post("/predict", json=BASE)
    body = r.json()
    expected = {0: "Normal session", 1: "Possible attack session"}
    assert body["label"] == expected[body["prediction"]]


def test_predict_session_id_optional(api_client):
    r = api_client.post("/predict", json={"session_id": "SID_001", **BASE})
    assert r.status_code == 200


def test_predict_encryption_used_optional(api_client):
    payload = {k: v for k, v in BASE.items() if k != "encryption_used"}
    r = api_client.post("/predict", json=payload)
    assert r.status_code == 200


def test_predict_invalid_protocol_type(api_client):
    r = api_client.post("/predict", json={**BASE, "protocol_type": "FTP"})
    assert r.status_code == 422


def test_predict_missing_required_field(api_client):
    payload = {k: v for k, v in BASE.items() if k != "network_packet_size"}
    r = api_client.post("/predict", json=payload)
    assert r.status_code == 422