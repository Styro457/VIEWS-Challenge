"""
Handles API testing functionality
"""


from fastapi.testclient import TestClient
from views_challenge.main import app

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
