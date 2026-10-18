# Distributed Smart Grid Peak Load Forecasting and Regional Demand Analytics Using Hadoop and Spark

A big-data pipeline that forecasts daily peak electricity demand for Indian states and regions, with dashboards for viewing forecasts and managing the pipeline.

## Overview

Daily demand data is stored in HDFS, cleaned with Apache Pig, organised in Hive, and used to train forecasting models in Spark MLlib. Predictions are presented in Streamlit dashboards for end users and operators.

```
CSV  →  HDFS  →  Pig (cleaning)  →  Hive (partitioned)  →  Spark MLlib  →  Streamlit
```

## Technology Stack

| Purpose | Technology |
|---|---|
| Distributed storage | Hadoop HDFS |
| Data cleaning | Apache Pig |
| Data warehouse | Apache Hive (Parquet, partitioned by state) |
| Processing and modelling | PySpark, Spark MLlib |
| Dashboards | Streamlit, Plotly |
| Deployment | Docker Compose |

## Dataset

Indian Power Demand & Shortage Data (2020–2025), daily state-wise records from Grid-India (POSOCO), available on Kaggle. Place the file at `data/grid_demand.csv`.

## Repository Structure

```
docker-compose.yml       Cluster configuration
hadoop.env               Shared Hadoop/Hive settings
data/clean_grid_data.pig Data cleaning script
sql/create_tables.hql    Hive table definitions
scripts/train_model.py   Feature engineering and model training
user_interface.py        User dashboard
operator_interface.py    Operator console
dashboard.py             Basic dashboard
requirements.txt         Python dependencies
```

## Getting Started

**Prerequisites:** Docker Desktop (about 8 GB RAM allocated) and Python 3.

1. Start the cluster:
   ```
   docker compose up -d
   ```
2. Load the data into HDFS, run the Pig cleaning script, and create the Hive tables (`sql/create_tables.hql`).
3. Train the models:
   ```
   docker exec -it -u root -e HOME=/tmp spark-master spark-submit --master spark://spark-master:7077 --conf spark.jars.ivy=/tmp/.ivy2 /scripts/train_model.py
   ```
   Predictions are written to `data/predictions/predictions.csv`.
4. Launch the dashboards:
   ```
   pip install -r requirements.txt
   streamlit run user_interface.py
   streamlit run operator_interface.py --server.port 8502
   ```

## Approach

- **Cleaning:** invalid and missing records are removed (69,012 → 62,972 rows).
- **Features:** previous-day and previous-week demand, 7-day and 30-day rolling statistics, calendar features, and state.
- **Models:** Linear Regression and Random Forest, trained on the first 80% of each state's history and tested on the most recent 20%.
- **Metrics:** RMSE and MAPE.

## Results

| Model | RMSE | MAPE |
|---|---|---|
| Linear Regression | 422 MW | 6.21% |
| Random Forest | 635 MW | 5.24% |

Most states are forecast within 5% error. Smaller north-eastern states and Telangana show higher error.

## Limitations

- The dataset contains days with no data, which are excluded rather than estimated.
- Accuracy figures are likely optimistic for true day-ahead forecasting, because some features use same-day values.
- Only daily data is available, so the time of day at which peak demand occurs cannot be predicted.

## Data Source

Grid-India (POSOCO) daily reports, via the Kaggle dataset "Indian Power Demand & Shortage Data (2020–2025)".
