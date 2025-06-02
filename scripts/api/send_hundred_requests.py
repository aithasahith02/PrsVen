import requests
import random
import time
import json

URL = "http://127.0.0.1:8000/predict"

BASE_PAYLOAD = {
    "gender": "Female",
    "smoker_status": "No",
    "age": 60.0,
    "features": [
        0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.6, -0.4, 0.25, -0.3,
        0.15, -0.6, 0.12, -0.1, 0.8, 0.5, -0.5, 0.9, 0.3, -0.2,
        0.1, 0.0, -0.3, 0.7, 0.6, -0.1, -0.2, 0.05, -0.6, 0.1,
        0.2, -0.4, 0.5, 0.3, 0.25, -0.15, 0.4, 0.6, -0.2, 0.7,
        0.8, -0.5, 0.6, 0.2, 0.1, -0.2, 0.3, -0.1, 0.5, 0.6,
        -0.6, 0.2, 0.0, -0.4, 0.1, 0.3, -0.1, 0.8, 0.5, -0.5,
        0.9, -0.3, 0.4, -0.2, 0.7, 0.2, -0.3, 0.5, 0.1, 0.2,
        -0.6, 0.3, 0.4, -0.1, 0.3
    ]
}

for i in range(100):
    payload = BASE_PAYLOAD.copy()
    # Optionally randomize age slightly to simulate variation
    payload["age"] = round(random.uniform(10, 80), 1)
    # Optional: add jitter to feature values
    payload["features"] = [round(x + random.uniform(-0.05, 0.05), 3) for x in BASE_PAYLOAD["features"]]

    response = requests.post(URL, json=payload)
    print(f"Request {i+1}: Status {response.status_code} | Response: {response.json()}")

    time.sleep(0.1)  # Slight delay to avoid overwhelming the server
