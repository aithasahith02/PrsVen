from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    'survival_prediction_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',
    catchup=False
) as dag:

    clean_data = BashOperator(
        task_id='clean_data',
        bash_command='cd /app && python3 scripts/preprocessing/clean_data.py'
    )

    train_rsf = BashOperator(
        task_id='train_rsf',
        bash_command='cd /app && python3 scripts/training/rsf_model.py'
    )

    train_cox = BashOperator(
        task_id='train_cox',
        bash_command='cd /app && python3 scripts/training/train_cox.py'
    )

    save_training_data = BashOperator(
        task_id='save_training_data',
        bash_command='cd /app && python3 scripts/training/save_training_data.py'
    )

    process_logs = BashOperator(
        task_id='process_inference_logs',
        bash_command='cd /app && python3 scripts/etl/process_inference_logs.py'
    )

    report_feature_dist = BashOperator(
        task_id='feature_distribution_report',
        bash_command='cd /app && python3 scripts/monitoring/feature_distribution_report.py'
    )

    drift_report = BashOperator(
        task_id='drift_report',
        bash_command='cd /app && python3 scripts/monitoring/drift_report.py'
    )

    # Set dependencies
    clean_data >> [train_cox, train_rsf] >> save_training_data >> process_logs >> report_feature_dist >> drift_report
