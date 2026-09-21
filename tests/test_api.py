import pytest
from fastapi.testclient import TestClient

from app.main import app 

client = TestClient(app)

 #Test 1
def test_prediction_correcte():
    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
   
    assert data["predictions"] == [2.0, 4.0, 6.0]

#Test 2
def test_prediction_incorrecte():
    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
  
    assert data["predictions"] != [99.0, 99.0, 99.0]

#Test 3 
def test_prediction_json_incorrect():
    # Missing the "features" key entirely, passing raw numbers instead
    payload = {"wrong_key": [3.5, 1.2, 4.9]} 
    response = client.post("/predict", json=payload)
    
    
    assert response.status_code == 422


