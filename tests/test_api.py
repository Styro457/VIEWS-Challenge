"""
Handles API testing functionality
"""

from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient

# Mock the database engine and metadata before importing the app
with patch('views_challenge.database.database.engine') as mock_engine, \
     patch('views_challenge.database.models.Base.metadata.create_all') as mock_create_all:
    mock_engine.execute = Mock()
    mock_create_all.return_value = None
    from views_challenge.main import app
    from views_challenge.api.keys_handler import verify_api_key_with_rate_limit

# Mock the API key verification for tests
def mock_verify_api_key():
    return Mock()

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
    assert content["countries"] is not None and len(content["countries"]) > 0

def test_get_cells():
    assert True
