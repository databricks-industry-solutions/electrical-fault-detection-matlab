# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/ [TODO] and more information about this solution accelerator at https://www.databricks.com/solutions/accelerators/ [TODO].

# COMMAND ----------

# MAGIC %md
# MAGIC <div >
# MAGIC   <img src="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/solution-accelerator-logo.png?raw=true"; width="50%">
# MAGIC </div>
# MAGIC
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC Power outages, if not addressed promptly, can have adverse effects on consumers. A significant challenge facing utility companies today is transitioning from reactive to proactive detection of potential electrical faults. Moreover, once a prediction indicates a high likelihood of a fault, the next hurdle is accurately identifying the precise location within the electrical grid where the fault may occur. To effectively train models for this purpose, utility companies require a substantial amount of labeled data to develop fault detection algorithms. By leveraging MATLAB and Databricks, we utilize physical models to create powerful simulations that generate the necessary data for training machine learning models. This solution accelerator provides a head start to use simulated electrical fault data to train a deep learning model to predict electrical system faults. 
# MAGIC
# MAGIC **Authors**
# MAGIC - Zachary Jacbson
# MAGIC - Jenny Park
# MAGIC - Drew Triplett
# MAGIC - Karthiga Mahalingam

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2024]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
# MAGIC |PyTorch|PyTorch and Caffe2 |https://github.com/pytorch/pytorch/blob/main/LICENSE|https://github.com/pytorch/|