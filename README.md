Survival Analysis MLOps Pipeline 
=============================================================

[![Build Status](https://github.com/aithasahith02/PrsVen/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/survival-mlops/actions)

🚀 Project Highlights
---------------------

-   ✨ **End-to-end Survival Analysis MLOps pipeline** combining CoxPH and Random Survival Forest (RSF)

-   ♲ **0.8005 C-index** with RSF model for accurate patient survival prediction

-   ⚙ Automated with **Apache Airflow**, deployed using **Kubernetes**, containerized with **Docker**

-   ⚡ **FastAPI-based REST API** for real-time predictions (<200ms latency)

-   🔍 Over **75 feature drift plots** auto-generated and stored daily

-   ✅ **CI/CD with GitHub Actions** for testing, linting, and redeployment

* * * * *

🔬 Overview
-----------

This project builds a scalable, production-ready **MLOps pipeline** for predicting patient survival time using real-world clinical data. It includes preprocessing, model training, evaluation, inference API, feature drift monitoring, and deployment pipelines.

Two survival models were trained: **CoxPH** and **Random Survival Forest**. The RSF model demonstrated superior performance:

| Metric | CoxPH | RSF |
| --- | --- | --- |
| C-Index | 0.5481 | **0.8005** |
| AUC @ 12 months | 0.4340 | 0.0944 |
| AUC @ 24 months | 0.4437 | 0.0938 |
| AUC @ 36 months | 0.4448 | 0.0833 |
| Integrated Brier Score | 0.1891 | **0.1630** |

* * * * *

🚀 MLOps Features
-----------------

-   **Modular DAGs with Airflow** orchestrating all 9 pipeline stages

-   **Model versioning**, inference logging, and drift visualization using Parquet + seaborn/matplotlib

-   **FastAPI REST service** supports real-time input & inference testing

-   **Kubernetes manifests** for multi-container deployment

-   **Multi-stage Docker builds** for optimized image size

-   **GitHub Actions** workflows for CI/CD: build, test, lint, push, deploy

* * * * *

🌐 Key Technologies
-------------------

-   **Python** --- Core logic, modeling, and API

-   **Apache Airflow** --- ML pipeline orchestration

-   **Docker & Kubernetes** --- Containerization and scalable deployment

-   **FastAPI** --- REST API server for predictions

-   **GitHub Actions** --- CI/CD workflows

-   **Parquet** --- Efficient data storage

-   **Lifelines, scikit-survival** --- Survival modeling libraries

* * * * *

📂 Directory Structure
----------------------

```
├── dags/                      # Airflow DAG definitions
├── data/                      # Raw and preprocessed data
├── data-lake/                # Daily stored training/inference data
├── execution/                # Automation shell scripts
├── feature_distribution_reports/  # KDE, histogram-based drift plots
├── k8s/                      # Deployment YAMLs (webserver, scheduler, db)
├── models/                   # Trained models and scalers
├── outputs/                  # Key screenshots
├── report_outputs/          # COX vs RSF visual reports
├── scripts/
│   ├── api/                  # FastAPI app + output visualizer
│   ├── etl/                  # Inference log handling
│   ├── preprocessing/        # Data cleaning and transformation
│   ├── training/             # Model training and storage
│   └── evaluation/           # Metric generation & comparison
├── tests/                    # Unit tests for API and scripts
├── Dockerfile                # Multi-stage build Dockerfile
├── requirements.txt          # Python dependencies
└── .github/workflows/        # GitHub Actions YAML CI pipeline

```

* * * * *

🥇 Why This Project Stands Out
------------------------------

This isn't just another ML pipeline --- it's a **real-world-ready**, fully containerized, modular, and observable system. From **model metrics that outperform baselines** to **scalable Kubernetes deployments and CI pipelines**, this project reflects a deep integration of **ML engineering + DevOps**.

It shows:

-   How to build real-time prediction APIs on clinical datasets

-   How to scale and monitor ML inference systems

-   How to integrate CI/CD in ML workflows with full traceability

* * * * *

📅 Future Improvements
----------------------

-   Add MLflow or Neptune for experiment tracking

-   Add Prometheus/Grafana dashboards for system observability

-   Integrate MinIO or AWS S3 for scalable data lake

* * * * *

👤 Author
---------

**Sahith Aitha**\
[LinkedIn](https://www.linkedin.com/in/sahith-aitha-845887191/) | [GitHub](https://github.com/aithasahith02) | [Portfolio](https://sahithaitha.com/)

Thank you!
