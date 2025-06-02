#!/bin/bash
# This script can be used to start the feature drift monitoring pipeline

cd /Users/sahithaitha/Documents/projects/PrsVen || exit

echo "Starting the Drift Monitoring Pipeline"

# Start FastAPI server in background
echo "Starting FastAPI server.."
PYTHONPATH=. uvicorn scripts.api.app:app --reload &
SERVER_PID=$!

sleep 5

# Step 1: Save cleaned training data
echo "Saving cleaned training data to data-lake.."
python scripts/training/save_training_data.py

# Step 2: Send 100 synthetic inputs to the API
echo "Sending 100 synthetic inputs to FastAPI..."
python scripts/api/send_hundred_inputs.py

sleep 2

# Step 3: Process inference logs into aggregate format
echo "Processing inference logs..."
python scripts/etl/process_inference_logs.py

# Step 4: Generate feature distribution report
echo "Generating drift report..."
python scripts/monitoring/feature_distribution_report.py

# Stop FastAPI server
echo "Stopping FastAPI server..."
kill $SERVER_PID

echo "Drift Monitoring Pipeline Completed!"
echo "Please navigate to feature_distribution_reports directory to view the latest reports!"
