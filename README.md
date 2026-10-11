# Distributed Smart Grid Peak Load Forecasting and Regional Demand Analytics Using Hadoop and Spark

Big-data pipeline that forecasts daily peak electricity demand for Indian states/regions.

**Pipeline:** CSV → HDFS → Pig (cleaning) → Hive (Parquet, partitioned by state) → PySpark window features → Spark MLlib regression → CSV → Streamlit dashboards

| Layer | Tool |
|---|---|
| Storage | HDFS |
| Cleaning | Apache Pig 0.17 (local mode) |
| Warehouse | Hive, Parquet partitioned by `state` |
| Processing / ML | PySpark window functions, Spark MLlib (Linear Regression, Random Forest) |
| Serving | Streamlit + Plotly (user view and operator console) |
| Runtime | Docker Desktop (WSL2), `bde2020` Hadoop/Hive images, `bitnamilegacy/spark:3.5` |

## Dataset
"Indian Power Demand & Shortage Data (2020–2025)" from Kaggle (daily, state-wise, from Grid-India/POSOCO).
Save it as `data/grid_demand.csv` (9 columns, 36 entities x 1,917 days = 69,012 rows).

## Repository layout
```
docker-compose.yml        cluster definition (7 containers)
hadoop.env                shared Hadoop/Hive config
data/clean_grid_data.pig  Pig cleaning script
sql/create_tables.hql     Hive tables
scripts/train_model.py    PySpark feature engineering + MLlib training
dashboard.py              basic dashboard
user_interface.py         viewer app
operator_interface.py     operator console (data view, training, logs)
requirements.txt          Python packages for the Streamlit apps
```

## Setup (Windows 11, Docker Desktop)
1. Install Docker Desktop with the WSL2 backend (give it about 8 GB RAM, 4+ CPUs).
2. Start the cluster:
   ```powershell
   docker compose up -d
   docker compose ps
   ```
   HDFS UI: http://localhost:9870, Spark UI: http://localhost:8080
3. Install Python packages inside the Spark containers (repeat after any container recreation):
   ```powershell
   docker exec -it -u root spark-master pip install numpy pandas
   docker exec -it -u root spark-worker pip install numpy
   ```

## Run the pipeline
1. **Load raw data into HDFS**
   ```powershell
   docker exec -it namenode bash
   ```
   ```bash
   hdfs dfs -mkdir -p /data/grid/raw /data/grid/clean
   hdfs dfs -put -f /data/grid_demand.csv /data/grid/raw/
   ```
2. **Clean with Pig** (Pig must be installed in the namenode container, e.g. extracted to `/opt/pig-0.17.0`; it is lost if the container is recreated)
   ```bash
   /opt/pig-0.17.0/bin/pig -x local /data/clean_grid_data.pig
   hdfs dfs -put -f /tmp/grid_demand_clean/part-m-00000 /data/grid/clean/grid_demand_clean.csv
   ```
   The script drops the header row, rows with NULL `max_demand_met`/`energy_met`/`drawl_schedule`, and rows with `max_demand_met <= 0`. Result: 69,012 → 62,972 rows.
3. **Create the Hive tables**
   ```powershell
   docker cp sql/create_tables.hql hive-server:/tmp/create_tables.hql
   docker exec -it hive-server beeline -u jdbc:hive2://localhost:10000 -f /tmp/create_tables.hql
   ```
4. **Train and predict**
   ```powershell
   docker exec -it -u root -e HOME=/tmp spark-master spark-submit --master spark://spark-master:7077 --conf spark.jars.ivy=/tmp/.ivy2 /scripts/train_model.py
   ```
   Writes `data/predictions/predictions.csv`.
5. **Dashboards**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run user_interface.py
   streamlit run operator_interface.py --server.port 8502
   ```

## Method
- Features per state (PySpark `Window`): `lag_1`, `lag_7`, 7-day and 30-day rolling average, 7-day rolling max, plus day of week, month, day of year, `energy_met`, `drawl_schedule` and a one-hot state encoding.
- Chronological 80/20 train/test split per state (no random split, because this is a time series).
- Models: Linear Regression and Random Forest (50 trees, depth 8); metrics RMSE and MAPE.
- Essar steel (a single industrial consumer whose data ends 2020-04-15) is excluded.

## Results (all states, 12,547 test rows)
| Model | RMSE | MAPE |
|---|---|---|
| Linear Regression | 422.44 MW | 6.21% |
| Random Forest | 634.92 MW | 5.24% |

The script keeps the model with the lower RMSE (Linear Regression). Most states are under 5% MAPE (Kerala 2.8%, DVC 1.9%). Small north-eastern states (Sikkim 22.8%, Mizoram 16.9%, Nagaland, Arunachal, Manipur) and Telangana (15.3%) are weakest.

## Known limitations
- **Data gaps:** 29 days are empty for all 36 entities; Pig removes them rather than imputing, so lags near those days are shifted by a day or so. DD, DNH and Jharkhand have shorter histories.
- **Possible leakage:** the rolling windows include the current day and `energy_met` is a same-day value, so reported accuracy is probably optimistic for a true day-ahead forecast.
- **Daily data only:** the dataset has no time-of-day information, so the time of the daily peak cannot be predicted.
- **Spark Parquet output:** Spark's Parquet writer produced no data files on the Docker/WSL2/Windows bind mount, so predictions are collected with `toPandas()` and written as CSV.

## Data source
Kaggle: Indian Power Demand & Shortage Data (2020–2025), originally from Grid-India/POSOCO daily reports.
