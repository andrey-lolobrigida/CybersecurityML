import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.config import load_config
from src.main import build_pipeline


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def pipeline():
    return build_pipeline()


@pytest.fixture(scope="session")
def api_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def raw_row():
    return {
        "session_id": "SID_TEST",
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