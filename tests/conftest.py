import pytest
from unittest.mock import MagicMock
import numpy as np
import joblib
import importlib
import scripts.api.app 

mock_rsf_model_g = MagicMock()
mock_scaler_g = MagicMock()
mock_gender_encoder_g = MagicMock()
mock_smoker_encoder_g = MagicMock()

mock_scaler_g.transform.return_value = np.array([[0.05] * 76])
mock_gender_encoder_g.transform.return_value = np.array([0])
mock_gender_encoder_g.classes_ = np.array(["Female", "Male"])
mock_smoker_encoder_g.transform.return_value = np.array([0])
mock_smoker_encoder_g.classes_ = np.array(["No", "Yes"])
mock_step_function_g = MagicMock()
mock_step_function_g.x = np.array([1.0, 2.0, 3.0, 120.0])
mock_step_function_g.y = np.array([0.99, 0.9, 0.8, 0.1])
mock_rsf_model_g.predict_survival_function.return_value = [mock_step_function_g]

def _mock_joblib_load_router(filepath_arg):
    filepath_str = str(filepath_arg)
    if "rsf_model.pkl" in filepath_str: return mock_rsf_model_g
    elif "scaler.joblib" in filepath_str: return mock_scaler_g
    elif "gender_encoder.joblib" in filepath_str: return mock_gender_encoder_g
    elif "smoker_encoder.joblib" in filepath_str: return mock_smoker_encoder_g
    raise FileNotFoundError(f"Conftest mocked joblib.load called with unexpected filepath: {filepath_str}")

@pytest.fixture(scope="session", autouse=True)
def patch_joblib_globally(session_mocker): 
    """Patches 'joblib.load' for the entire test session."""
    session_mocker.patch('joblib.load', side_effect=_mock_joblib_load_router)
    importlib.reload(scripts.api.app) 