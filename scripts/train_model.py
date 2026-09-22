"""
Grid Load Forecasting — PySpark + Spark MLlib (ALL STATES)
Reads from Hive (grid_demand, partitioned by state), engineers time-series
features per state with window functions, encodes state as a categorical
feature, trains LinearRegression + RandomForestRegressor on the pooled
dataset, evaluates with RMSE/MAPE, and writes predictions for every state
to /data/predictions/predictions.csv.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ---------------------------------------------------------------------------
# 1. Spark session, connected to the Hive metastore
# ---------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("grid-load-forecast-all-states") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 2. Load from Hive (the partitioned Parquet table) — ALL STATES
# ---------------------------------------------------------------------------
df = spark.sql("SELECT * FROM grid_demand")

df = df.withColumn("demand_date", F.to_timestamp("demand_date")) \
       .withColumn("day_of_week", F.dayofweek("demand_date")) \
       .withColumn("month", F.month("demand_date")) \
       .withColumn("day_of_year", F.dayofyear("demand_date"))

# No state filter this time — every state stays in the dataset.

# ---------------------------------------------------------------------------
# 3. Feature engineering — rolling averages + lag features, PER STATE
# ---------------------------------------------------------------------------
# partitionBy("state") is essential here: it keeps each state's time series
# separate so lag/rolling values never leak across states.
w = Window.partitionBy("state").orderBy("demand_date")
w7 = w.rowsBetween(-6, 0)    # trailing 7-day window (inclusive of current day)
w30 = w.rowsBetween(-29, 0)  # trailing 30-day window

df = df \
    .withColumn("lag_1", F.lag("max_demand_met", 1).over(w)) \
    .withColumn("lag_7", F.lag("max_demand_met", 7).over(w)) \
    .withColumn("rolling_avg_7d", F.avg("max_demand_met").over(w7)) \
    .withColumn("rolling_avg_30d", F.avg("max_demand_met").over(w30)) \
    .withColumn("rolling_max_7d", F.max("max_demand_met").over(w7))

# ---------------------------------------------------------------------------
# 4. Encode state as a categorical feature
# ---------------------------------------------------------------------------
state_indexer = StringIndexer(inputCol="state", outputCol="state_index", handleInvalid="keep")
state_encoder = OneHotEncoder(inputCol="state_index", outputCol="state_ohe")

df = state_indexer.fit(df).transform(df)
df = state_encoder.fit(df).transform(df)

# ---------------------------------------------------------------------------
# 5. Drop rows with nulls in any feature column
# ---------------------------------------------------------------------------
feature_cols = [
    "day_of_week", "month", "day_of_year",
    "lag_1", "lag_7", "rolling_avg_7d", "rolling_avg_30d", "rolling_max_7d",
    "energy_met", "drawl_schedule", "state_ohe",
]
numeric_null_check_cols = [
    "day_of_week", "month", "day_of_year",
    "lag_1", "lag_7", "rolling_avg_7d", "rolling_avg_30d", "rolling_max_7d",
    "energy_met", "drawl_schedule",
]
df = df.na.drop(subset=numeric_null_check_cols + ["max_demand_met"])

# ---------------------------------------------------------------------------
# 6. Assemble feature vector
# ---------------------------------------------------------------------------
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
df_features = assembler.transform(df)

# ---------------------------------------------------------------------------
# 7. Time-based train/test split — PER STATE (not random, not global)
# ---------------------------------------------------------------------------
# Each state gets its own chronological 80/20 split, so a state with a
# shorter history isn't skewed by another state's date range.
split_window = Window.partitionBy("state").orderBy("demand_date")
df_features = df_features.withColumn("row_num", F.row_number().over(split_window))
df_features = df_features.withColumn("state_total", F.count("*").over(Window.partitionBy("state")))
df_features = df_features.withColumn("split_idx", (F.col("state_total") * 0.8).cast("int"))

train_df = df_features.filter(F.col("row_num") <= F.col("split_idx"))
test_df = df_features.filter(F.col("row_num") > F.col("split_idx"))

train_count = train_df.count()
test_count = test_df.count()
print(f"Train rows: {train_count}, Test rows: {test_count}")
print(f"States in dataset: {df.select('state').distinct().count()}")

# ---------------------------------------------------------------------------
# 8. Train models (pooled across all states, with state as a feature)
# ---------------------------------------------------------------------------
lr = LinearRegression(featuresCol="features", labelCol="max_demand_met")
lr_model = lr.fit(train_df)

rf = RandomForestRegressor(featuresCol="features", labelCol="max_demand_met", numTrees=50, maxDepth=8)
rf_model = rf.fit(train_df)

# ---------------------------------------------------------------------------
# 9. Evaluate: RMSE and MAPE for both models (overall, across all states)
# ---------------------------------------------------------------------------
def evaluate(model, name):
    preds = model.transform(test_df)
    rmse = RegressionEvaluator(labelCol="max_demand_met", predictionCol="prediction", metricName="rmse").evaluate(preds)
    mape_df = preds.withColumn(
        "abs_pct_error", F.abs((F.col("max_demand_met") - F.col("prediction")) / F.col("max_demand_met"))
    )
    mape = mape_df.agg(F.avg("abs_pct_error")).first()[0] * 100
    print(f"{name} — RMSE: {rmse:.2f} MW, MAPE: {mape:.2f}%")
    return preds, rmse, mape

lr_preds, lr_rmse, lr_mape = evaluate(lr_model, "Linear Regression")
rf_preds, rf_rmse, rf_mape = evaluate(rf_model, "Random Forest")

best_preds = rf_preds if rf_rmse < lr_rmse else lr_preds
best_name = "random_forest" if rf_rmse < lr_rmse else "linear_regression"
print(f"Best model: {best_name}")

# Per-state breakdown, useful for your results slide
print("\nPer-state MAPE (best model):")
per_state = best_preds.withColumn(
    "abs_pct_error", F.abs((F.col("max_demand_met") - F.col("prediction")) / F.col("max_demand_met")) * 100
).groupBy("state").agg(F.avg("abs_pct_error").alias("mape")).orderBy("mape")
per_state.show(50, truncate=False)

# ---------------------------------------------------------------------------
# 10. Collect predictions for ALL states and write to CSV
# ---------------------------------------------------------------------------
output = best_preds.select(
    F.col("demand_date").alias("timestamp"),
    F.col("state"),
    F.col("max_demand_met").alias("actual_load"),
    F.col("prediction").alias("predicted_load"),
)

output_count = output.count()
print(f"\nRows to write: {output_count}")
output.show(5, truncate=False)

# Dataset is still small enough (tens of thousands of rows across all
# states) to collect to the driver and write with pandas — sidesteps the
# distributed Parquet writer issues seen on this Docker/Windows setup.
pandas_df = output.toPandas()
print(f"Collected {len(pandas_df)} rows to driver")

import os
os.makedirs("/data/predictions", exist_ok=True)
pandas_df.to_csv("/data/predictions/predictions.csv", index=False)

print("Predictions written to /data/predictions/predictions.csv")
spark.stop()
