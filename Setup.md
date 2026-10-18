# Setup Guide

Step-by-step instructions for running the pipeline on Windows 11 with Docker Desktop. For a project overview, see [README.md](README.md).

## 1. Prerequisites

- Windows 11 with virtualization enabled in BIOS
- Windows features **Virtual Machine Platform** and **Windows Subsystem for Linux** turned on, then in PowerShell:
  ```powershell
  wsl --update
  wsl --set-default-version 2
  ```
- Docker Desktop (WSL2 backend), with about 8 GB RAM and 4+ CPUs allocated under Settings → Resources
- Python 3 and Git

## 2. Start the Cluster

From the project folder:
```powershell
docker compose up -d
docker compose ps
```
All seven containers (namenode, datanode, hive-metastore-postgresql, hive-metastore, hive-server, spark-master, spark-worker) should show `Up`. Allow a few minutes for Hive to initialise.

- HDFS UI: http://localhost:9870
- Spark UI: http://localhost:8080 (one worker should be registered)

If a container-name conflict appears, run `docker compose down` in any older project folder first.

## 3. Install Python Packages in the Spark Containers

Repeat this after any container is recreated:
```powershell
docker exec -it -u root spark-master pip install numpy pandas
docker exec -it -u root spark-worker pip install numpy
```

## 4. Load Data into HDFS

Copy `grid_demand.csv` and `clean_grid_data.pig` into the project's `data\` folder (mounted at `/data` in the namenode and Spark containers), then:
```powershell
docker exec -it namenode bash
```
```bash
hdfs dfs -mkdir -p /data/grid/raw /data/grid/clean
hdfs dfs -put -f /data/grid_demand.csv /data/grid/raw/
hdfs dfs -ls /data/grid/raw
```

## 5. Clean the Data with Pig

Pig runs in local mode inside the namenode container. Install it once:

1. On Windows, download `pig-0.17.0.tar.gz` from the Apache archive into the project's `data\` folder.
2. In the namenode container:
   ```bash
   tar -xzf /data/pig-0.17.0.tar.gz -C /opt
   ```
   Pig is lost if the namenode container is recreated; repeat this step if so.

Run the script and publish the result to HDFS:
```bash
/opt/pig-0.17.0/bin/pig -x local /data/clean_grid_data.pig
hdfs dfs -put -f /tmp/grid_demand_clean/part-m-00000 /data/grid/clean/grid_demand_clean.csv
exit
```
The script removes the header row, records with missing `max_demand_met`, `energy_met` or `drawl_schedule`, and records with `max_demand_met <= 0`. Expected result: 69,012 → 62,972 rows.

## 6. Create the Hive Tables

```powershell
docker cp sql/create_tables.hql hive-server:/tmp/create_tables.hql
docker exec -it hive-server beeline -u jdbc:hive2://localhost:10000 -f /tmp/create_tables.hql
```
Check the row counts in `beeline`:
```sql
SELECT COUNT(*) FROM grid_demand_clean;   -- 62972
SELECT COUNT(*) FROM grid_demand;         -- 62972
```

## 7. Train the Models

```powershell
docker exec -it -u root -e HOME=/tmp spark-master spark-submit --master spark://spark-master:7077 --conf spark.jars.ivy=/tmp/.ivy2 /scripts/train_model.py
```
Predictions are written to `data\predictions\predictions.csv`.

## 8. Run the Dashboards

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run user_interface.py
streamlit run operator_interface.py --server.port 8502
```
The user dashboard opens at http://localhost:8501 and the operator console at http://localhost:8502.

## Daily Use

```powershell
docker compose up -d      # start
docker compose down       # stop (HDFS and Hive data are kept)
wsl --shutdown            # optional: free memory when finished
```
After a restart, re-run step 3 if the containers were recreated.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Virtualization support not detected" in Docker Desktop | Enable the Windows features in step 1 and run `wsl --update`. |
| `bitnami/spark` image not found | Use `bitnamilegacy/spark:3.5` in `docker-compose.yml`. |
| Ivy error: `basedir must be absolute` | Run `spark-submit` with `-u root`, `-e HOME=/tmp` and `--conf spark.jars.ivy=/tmp/.ivy2`. |
| `No module named numpy` or `pandas` | Repeat step 3. |
| Hadoop user/UGI error on `spark-submit` | Run as root (`-u root`). |
| Parquet output folder contains only `_SUCCESS` | Expected on this setup; the script writes CSV through pandas instead. |
| `ModuleNotFoundError: plotly` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| Cannot delete files created by containers | Remove them with `docker exec -u root`. |
| Docker engine stopped | Restart Docker Desktop, run `docker compose up -d`, then repeat step 3. |
| `wget` or `apt` fails inside the namenode container | The image is based on an end-of-life Debian release; download files on Windows and use the shared `data\` folder. |
