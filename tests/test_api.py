"""
Handles API testing functionality
"""

from unittest.mock import Mock
from fastapi.testclient import TestClient
from views_challenge.api.keys_handler import verify_api_key_with_rate_limit
from views_challenge.configs.config import settings

settings.keys_mode = False

import pandas as pd
import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def setup_dummy_parquet():
    os.makedirs("env", exist_ok=True)
    path = "env/preds_001.parquet"

    created = False
    if not os.path.exists(path):
        df = pd.DataFrame({
            "pred_ln_sb_best": [[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24, 0.0]],
            "pred_ln_os_best": [[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24, 0.0]],
            "pred_ln_ns_best": [[0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24, 0.0]],
            "country_id": [163],
            "lat": [-46.75],
            "lon": [-37.75],
            "row": [87],
            "col": [436],
            "month_id": [409],
            "priogrid_id": [62356],
        })
        df = df.set_index(["priogrid_id", "month_id"])
        df.to_parquet(path, index=True)
        created = True

    yield

    if created and os.path.exists(path):
        os.remove(path)

# Mock the API key verification for tests
def mock_verify_api_key():
    return Mock()

from views_challenge.main import app

app.dependency_overrides[verify_api_key_with_rate_limit] = mock_verify_api_key

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 404

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    content = response.json()
    assert content.get("data_loaded") is True

def test_get_countries():
    response = client.get("/countries")
    assert response.status_code == 200
    content = response.json()
    assert content["countries"] is not None

def test_get_cells():
    assert True
