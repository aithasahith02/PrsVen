import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from .conftest import ( 
    mock_rsf_model_g,
    mock_scaler_g,
    mock_gender_encoder_g,
    mock_smoker_encoder_g
)

from scripts.api.app import app

@pytest.fixture(scope="function")
def app_client(): 
    mock_rsf_model_g.reset_mock()
    mock_scaler_g.reset_mock()
    mock_gender_encoder_g.reset_mock()
    mock_smoker_encoder_g.reset_mock()

    mock_scaler_g.transform.return_value = np.array([[0.05] * 76])
    mock_gender_encoder_g.transform.return_value = np.array([0])
    mock_smoker_encoder_g.transform.return_value = np.array([0])
    fresh_mock_step_function = MagicMock()
    fresh_mock_step_function.x = np.array([1.0, 2.0, 3.0, 120.0])
    fresh_mock_step_function.y = np.array([0.99, 0.9, 0.8, 0.1])
    mock_rsf_model_g.predict_survival_function.return_value = [fresh_mock_step_function]
    return TestClient(app)

# Input Data
VALID_PATIENT_DATA = {
    "gender": "Female",
    "smoker_status": "No",
    "age": 60.0,
    "features": [0.1] * 75
}

# Test function - Survival Success prediction
def test_predict_survival_success(app_client):
    response = app_client.post("/predict", json=VALID_PATIENT_DATA)
    assert response.status_code == 200
    data = response.json()
    assert "survival_curve" in data
    assert len(data["survival_curve"]) == len(mock_rsf_model_g.predict_survival_function.return_value[0].x)
    assert mock_scaler_g.transform.call_count == 1
    call_args_df = mock_scaler_g.transform.call_args[0][0]
    assert isinstance(call_args_df, pd.DataFrame)
    expected_scaler_cols = [f"feature_{i+1}" for i in range(75)] + ["age"]
    assert list(call_args_df.columns) == expected_scaler_cols

# Test function - Input shape
def test_predict_survival_invalid_feature_length(app_client):
    invalid_data = VALID_PATIENT_DATA.copy()
    invalid_data["features"] = [0.1] * 70
    response = app_client.post("/predict", json=invalid_data)
    assert response.status_code == 400
    assert "Expected 75 features, but got 70" in response.json()["detail"]

# Test function - "Gender" features encoding
def test_predict_survival_invalid_gender_value(app_client):
    def side_effect_gender_transform(values_list):
        if values_list[0] == "UnknownGender":
            raise ValueError(f"y contains new labels: {set(values_list) - set(mock_gender_encoder_g.classes_)}")
        idx = np.where(mock_gender_encoder_g.classes_ == values_list[0])[0]
        return np.array([idx[0]]) if len(idx) > 0 else np.array([-1])
    mock_gender_encoder_g.transform.side_effect = side_effect_gender_transform

    invalid_data = VALID_PATIENT_DATA.copy()
    invalid_data["gender"] = "UnknownGender"
    response = app_client.post("/predict", json=invalid_data)
    assert response.status_code == 400
    data = response.json()
    assert "Invalid categorical value provided" in data["detail"]
    assert "UnknownGender" in data["detail"]
    for cls_name in mock_gender_encoder_g.classes_:
        assert cls_name in data["detail"]

# Test function - "Smoker" features encoding
def test_predict_survival_invalid_smoker_status_value(app_client):
    def side_effect_smoker_transform(values_list):
        if values_list[0] == "UnknownSmokerStatus":
            raise ValueError(f"y contains new labels: {set(values_list) - set(mock_smoker_encoder_g.classes_)}")
        idx = np.where(mock_smoker_encoder_g.classes_ == values_list[0])[0]
        return np.array([idx[0]]) if len(idx) > 0 else np.array([-1])
    mock_smoker_encoder_g.transform.side_effect = side_effect_smoker_transform

    invalid_data = VALID_PATIENT_DATA.copy()
    invalid_data["smoker_status"] = "UnknownSmokerStatus"
    response = app_client.post("/predict", json=invalid_data)
    assert response.status_code == 400
    data = response.json()
    assert "Invalid categorical value provided" in data["detail"]
    assert "UnknownSmokerStatus" in data["detail"]
    for cls_name in mock_smoker_encoder_g.classes_:
        assert cls_name in data["detail"]