"""
Grid Load Forecasting - PySpark + Spark MLlib (ALL STATES)
Reads from Hive (grid_demand, partitioned by state), engineers per-state
time-series features with window functions, trains LinearRegression +
RandomForestRegressor, evaluates with RMSE/MAPE and writes predictions to
/data/predictions/predictions.csv for the Streamlit dashboards.

Essar steel (a single industrial consumer, data ends 2020-04-15) is excluded.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("grid-load-forecast-all-states") \
    .config("hive.metastore.uris", "thrift://hive-metastore:9083") \
    .enableHiveSupport() \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------- load
df = spark.sql("SELECT * FROM grid_demand")
df = df.withColumn("demand_date", F.to_timestamp("demand_date")) \
       .withColumn("day_of_week", F.dayofweek("demand_date")) \
       .withColumn("month", F.month("demand_date")) \
       .withColumn("day_of_year", F.dayofyear("demand_date"))

# Exclude Essar steel: not a state, ends 2020-04-15, ~19 test rows
df = df.filter(F.col("state") != "Essar steel")

# ---------------------------------------------------------------- window features (per state)
w = Window.partitionBy("state").orderBy("demand_date")
w7 = w.rowsBetween(-6, 0)
w30 = w.rowsBetween(-29, 0)

df = df \
    .withColumn("lag_1", F.lag("max_demand_met", 1).over(w)) \
    .withColumn("lag_7", F.lag("max_demand_met", 7).over(w)) \
    .withColumn("rolling_avg_7d", F.avg("max_demand_met").over(w7)) \
    .withColumn("rolling_avg_30d", F.avg("max_demand_met").over(w30)) \
    .withColumn("rolling_max_7d", F.max("max_demand_met").over(w7))

num_cols = [
    "day_of_week", "month", "day_of_year",
    "lag_1", "lag_7", "rolling_avg_7d", "rolling_avg_30d", "rolling_max_7d",
    "energy_met", "drawl_schedule",
]
df = df.na.drop(subset=num_cols + ["max_demand_met"])

# ---------------------------------------------------------------- state encoding
df = StringIndexer(inputCol="state", outputCol="state_idx", handleInvalid="keep").fit(df).transform(df)
df = OneHotEncoder(inputCols=["state_idx"], outputCols=["state_vec"]).fit(df).transform(df)

assembler = VectorAssembler(inputCols=num_cols + ["state_vec"],
                            outputCol="features", handleInvalid="skip")
df = assembler.transform(df)

# ---------------------------------------------------------------- chronological 80/20 split per state
df = df.withColumn("rn", F.row_number().over(Window.partitionBy("state").orderBy("demand_date"))) \
       .withColumn("n", F.count("*").over(Window.partitionBy("state")))
train_df = df.filter(F.col("rn") <= F.col("n") * 0.8)
test_df = df.filter(F.col("rn") > F.col("n") * 0.8)
print(f"Train rows: {train_df.count()}, Test rows: {test_df.count()}")

# ---------------------------------------------------------------- train
lr_model = LinearRegression(featuresCol="features", labelCol="max_demand_met").fit(train_df)
rf_model = RandomForestRegressor(featuresCol="features", labelCol="max_demand_met",
                                 numTrees=50, maxDepth=8).fit(train_df)

def with_ape(preds):
    return preds.withColumn(
        "abs_pct_error",
        F.abs((F.col("max_demand_met") - F.col("prediction")) / F.col("max_demand_met")))

def evaluate(model, name):
    preds = model.transform(test_df)
    rmse = RegressionEvaluator(labelCol="max_demand_met", predictionCol="prediction",
                               metricName="rmse").evaluate(preds)
    mape = with_ape(preds).agg(F.avg("abs_pct_error")).first()[0] * 100
    print(f"{name} - RMSE: {rmse:.2f} MW, MAPE: {mape:.2f}%")
    return preds, rmse, mape

lr_preds, lr_rmse, lr_mape = evaluate(lr_model, "Linear Regression")
rf_preds, rf_rmse, rf_mape = evaluate(rf_model, "Random Forest")

best_preds = rf_preds if rf_rmse < lr_rmse else lr_preds
print("Best model:", "random_forest" if rf_rmse < lr_rmse else "linear_regression")

# ---------------------------------------------------------------- per-state MAPE
print("\nPer-state MAPE (best model, excluding Essar steel):")
with_ape(best_preds).groupBy("state") \
    .agg(F.round(F.avg("abs_pct_error") * 100, 2).alias("mape_pct"),
         F.count("*").alias("test_rows")) \
    .orderBy(F.desc("mape_pct")) \
    .show(50, truncate=False)
print("Reference: previous run INCLUDING Essar steel had overall MAPE 6.26%")

# ---------------------------------------------------------------- output for dashboards
output = best_preds.select(
    F.col("demand_date").alias("timestamp"),
    F.col("state"),
    F.col("max_demand_met").alias("actual_load"),
    F.col("prediction").alias("predicted_load"),
).orderBy("state", "timestamp")

pandas_df = output.toPandas()
os.makedirs("/data/predictions", exist_ok=True)
pandas_df.to_csv("/data/predictions/predictions.csv", index=False)
print(f"Wrote {len(pandas_df)} rows to /data/predictions/predictions.csv")
spark.stop()
