from app import app

def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "visual-product-search"


def test_home_page_loads():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_evaluation_page_loads():
    client = app.test_client()
    response = client.get("/evaluation")
    assert response.status_code == 200