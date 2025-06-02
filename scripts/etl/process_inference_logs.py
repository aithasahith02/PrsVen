# PRSVEN/scripts/etl/process_inference_logs.py
import pandas as pd
import json
import os
from datetime import datetime

RAW_LOG_FILE = "api_raw_inference_logs.jsonl" 
DATA_LAKE_BASE_DIR = "./data-lake"
INFERENCE_DATA_DIR = os.path.join(DATA_LAKE_BASE_DIR, "inference")
EXPECTED_FEATURE_COUNT = 75 # Exactly equal to the count in app.py

os.makedirs(INFERENCE_DATA_DIR, exist_ok=True)

def process_logs():
    if not os.path.exists(RAW_LOG_FILE):
        print(f"Raw log file '{RAW_LOG_FILE}' not found. No inference data to process.")
        return

    processed_count = 0
    with open(RAW_LOG_FILE, 'r') as f_raw:
        for line_number, line in enumerate(f_raw):
            try:
                log_entry = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {line_number + 1}. Skipping.")
                continue

            request_id = log_entry.get("request_id", f"unknown_request_{line_number}")
            timestamp_str = log_entry.get("timestamp", datetime.utcnow().isoformat())
            
            try:
                event_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                filename_ts = event_dt.strftime('%Y%m%d_%H%M%S_%f')
            except ValueError:
                filename_ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f') 

            input_data = log_entry.get("input_features_raw", {})
            
            data_for_df = {
                'request_id': request_id,
                'api_request_timestamp': timestamp_str,
                'gender': input_data.get('gender'),
                'smoker_status': input_data.get('smoker_status'),
                'age': input_data.get('age')
            }
            
            features_list = input_data.get('features', [])
            for i in range(EXPECTED_FEATURE_COUNT):
                data_for_df[f'feature_{i+1}'] = features_list[i] if i < len(features_list) else None
            
            df_inference_single = pd.DataFrame([data_for_df])
            
            output_filename = f"inference_log_{filename_ts}_{request_id}.parquet"
            output_path = os.path.join(INFERENCE_DATA_DIR, output_filename)
            
            try:
                df_inference_single.to_parquet(output_path, index=False, engine='pyarrow')
                processed_count += 1
            except Exception as e:
                print(f"Error saving Parquet file {output_path}: {e}")

    if processed_count > 0:
        print(f"Successfully processed {processed_count} inference logs into Parquet files in '{INFERENCE_DATA_DIR}'.")
    else:
        print("No new inference logs were processed (or input file was empty/corrupt).")

if __name__ == "__main__":
    print("Processing raw API inference logs...")
    process_logs()
    print("Processing complete.")