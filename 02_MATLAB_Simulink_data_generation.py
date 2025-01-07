# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/ [TODO] and more information about this solution accelerator at https://www.databricks.com/solutions/accelerators/ [TODO].

# COMMAND ----------

# MAGIC %md
# MAGIC <div >
# MAGIC   <img src="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/solution-accelerator-logo.png?raw=true"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC
# MAGIC ### In this notebook you:
# MAGIC - Configure the environment
# MAGIC - Run Simulink model simulation on Databricks to generate electrical system fault data
# MAGIC - Output fault data to Delta Table

# COMMAND ----------

# MAGIC %md
# MAGIC # MATLAB Model Development
# MAGIC The workflow described in this document programmatically steps through fault sequences on each three-phase bus, two-phase bus and single-phase bus in the IEEE 123 Node Test Feeder. There are 70 three-phase buses, with 7 fault sequences possible per bus, giving a total of 490 faults for the three-phase buses. There are 3 two-phase buses, with 3 fault sequences possible per bus, giving a total of 9 faults for the two-phase buses. There are 55 single-phase buses, with 1 fault sequence possible per bus, giving a total of 55 faults for the single-phases buses. Overall, there are a total of 554 fault sequences in the system. The range in cell 4 provides several fault resistance values that is used in this example. Each simulation is ran for 55.5 seconds and then stops and starts a new one.
# MAGIC
# MAGIC <img src="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/IEEE123-Node-Test-Feeder.png?raw=true">
# MAGIC Magenta dot = substation, all other dots are faults and location

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cluster configuration
# MAGIC
# MAGIC When choosing cluster configuration, 3 sepearte tests were run, generating 10 simulations. 
# MAGIC Cluster Config as follows:
# MAGIC - Driver and worker type: r6id.xlarge, 32 GB memory, 4 cores, 4 worker nodes, Photon and autoscaling enabled (Delta Cache Memory Optimized)
# MAGIC - Driver and worker type: r5dn.2xlarge, 64 GB memory, 8 cores, 4 workers, Photon and autoscaling enbaled (Delta Cache Memory Optimized)
# MAGIC - Driver and worker type: r5dn.2xlarge, 128 GB memory, 16 cores, 4 workers, Photon and autoscaling enbaled (Delta Cache Memory Optimized)
# MAGIC
# MAGIC The first compute instance above runs for about 3.5 hours, the second instance above runs for ~1 hour and the third instance above runs for ~1 hour 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 1: Initiatlize catalog and schema

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.1: Define widgets

# COMMAND ----------

# DBTITLE 1,Define Widgets
dbutils.widgets.text("catalog", "main", "Catalog Name")
dbutils.widgets.text("schmea", "zach_jacobson", "Schema Name")

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.2: Create catalog and schema

# COMMAND ----------

# DBTITLE 1,Pull Widget Values
catalog_name = dbutils.widgets.get("catalog")
schema_name = dbutils.widgets.get("schema")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 2: Import special functions from wrapper

# COMMAND ----------

# DBTITLE 1,Import Libraries
from dbx.faults.wrapper import run_top_level_faults_output_names
from dbx.faults.wrapper import run_top_level_faults_output_schema
from dbx.faults.wrapper import run_top_level_faults_mapPartitions
from dbx.faults.wrapper import run_top_level_faults_applyInPandas
from dbx.faults.wrapper import run_top_level_faults_mapInPandas


# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 3: Generate inputs for Simulink model

# COMMAND ----------

# DBTITLE 1,Log Inputs for Simulink Model
# Resistance (Ron and Rg) of faults as random lograthmic like function of the resistance
# Simulation 1: 10 simulations on [TODO - incomplete comment]
from pyspark.sql import functions as f

N = 10

R = spark.range(N)

# Create logarithmic inputs for simulink model
log_inputs = (
    R.withColumn("Ron", f.pow(10.0, -1.0 * (N - R["id"].cast("double")) / N))
    .withColumn("Rg", f.pow(10.0, -2.0 * (N - R["id"].cast("double")) / N))
    .withColumn("StopTime", f.lit(1.0))
    .select("Ron", "Rg", "StopTime")
)

log_inputs.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 4: Create fault simulation dataset

# COMMAND ----------

# DBTITLE 1,Create Fault Simulation Dataset
# V1mag_13: Voltage magnitude 1 phase bus 13
# V1ph_13: Voltage phase angle 1 phase bus 13
import time

fault_simulation = (
    DF.groupBy("Ron")
    .applyInPandas(
        run_top_level_faults_applyInPandas, run_top_level_faults_output_schema
    )
    .cache()
)

display(fault_simulation)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 5: Write fault simulation output to Delta table

# COMMAND ----------

# DBTITLE 1,Write to Table
fault_simulation.write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.fault_simulation"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 11 steps per simulation are run on 1000 simulations which produces 11000 inputs (11*1000 = 11000 inputs) \
# MAGIC The output contains all fault types and location of fault type which will be located by the bus number. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC - Train a Convolutional Neural Network (CNN) model.

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2024]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
# MAGIC |PyTorch|PyTorch and Caffe2 |https://github.com/pytorch/pytorch/blob/main/LICENSE|https://github.com/pytorch/|