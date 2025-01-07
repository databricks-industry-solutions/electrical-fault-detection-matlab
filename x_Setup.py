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
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install lightning

# COMMAND ----------

# MAGIC %md
# MAGIC ##Helper Functions

# COMMAND ----------

@F.pandas_udf(StringType())
def map_codes(codes: pd.Series) -> pd.Series:
    """
    Map fault codes to their corresponding string values.

    Args:
    codes (pd.Series): A pandas Series containing fault codes.

    Returns:
    pd.Series: A pandas Series with fault codes mapped to their string values.
    """
    # Map the fault codes to their corresponding string values using the fault_types dictionary
    return codes.map(fault_types)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import re

# Define a UDF to extract all matching patterns from a given text
def extract_consecutive_letters(text, regex_pattern=r"(?<=_)[a-zA-Z]+"):
    """
    Extract all matching patterns from a given text based on the provided regex pattern.

    Args:
    text (str): The input text to search for patterns.
    regex_pattern (str): The regex pattern to use for matching. Defaults to r"(?<=_)[a-zA-Z]+".

    Returns:
    list: A list of all matching patterns found in the text.
    """
    # Match letters that are preceded by an underscore
    return re.findall(regex_pattern, text)

# Register the UDF with the appropriate return type
extract_udf = udf(extract_consecutive_letters, ArrayType(StringType()))

# COMMAND ----------

# Split out dataframe into simulation batches. 40 simulations, produces 550 records per match. Each simulation is placed at a 55.5 second threshold
def add_batch_id(pdf, batch_size=554):
    """
    Add a batch_id column to the dataframe to split it into simulation batches.

    Args:
    pdf (pd.DataFrame): The input pandas DataFrame.
    batch_size (int): The number of records per batch. Defaults to 554.

    Returns:
    pd.DataFrame: The DataFrame with an added batch_id column.
    """
    # Calculate number of rows in the dataframe
    num_rows = len(pdf)
    
    # Create the batch_id column by dividing the row index by batch_size and adding 1
    pdf["batch_id"] = (pd.Series(range(num_rows)) // batch_size) + 1
    
    return pdf

# COMMAND ----------

def arr_to_dataset(data_arr, label_vec):
    """
    Convert numpy arrays into a tensor dataset.

    Args:
    data_arr (np.ndarray): The input data array.
    label_vec (np.ndarray): The input label vector.

    Returns:
    torch.utils.data.TensorDataset: The resulting tensor dataset.
    """
    # Convert the data array into a tensor and transpose it to match the expected shape
    data_ten = torch.from_numpy(data_arr.transpose(0, 2, 1)).float()
    
    # Convert the label vector into a tensor by taking the argmax along axis 1
    label_ten = torch.from_numpy(label_vec.argmax(axis=1))
    
    # Create a TensorDataset from the data and label tensors
    dataset = torch.utils.data.TensorDataset(data_ten, label_ten)
    
    return dataset

# COMMAND ----------

def split_set(features, labels, p=0.7):
    """
    Split data and label arrays into train, validation, and test partitions.

    Args:
    features (np.ndarray): The input feature array.
    labels (np.ndarray): The input label array.
    p (float): The proportion of data to be used for training. Defaults to 0.7.

    Returns:
    tuple: A tuple containing the train, validation, and test datasets.
    """
    # Get the total number of samples
    n_total = np.shape(features)[0]
    
    # Calculate the number of training samples
    n_train = int(n_total * p)
    
    # Calculate the number of validation samples
    n_val = int(n_total * (0.7 + 0.15))
    
    # Generate a random permutation of indices
    idx = np.random.permutation(n_total)
    
    # Create masks for training, validation, and test sets
    train_mask = idx[:n_train]
    val_mask = idx[n_train:n_val]
    test_mask = idx[n_val:]
    
    # Create datasets using the masks
    trainset = arr_to_dataset(features[train_mask], labels[train_mask])
    valset = arr_to_dataset(features[val_mask], labels[val_mask])
    testset = arr_to_dataset(features[test_mask], labels[test_mask])

    return trainset, valset, testset

# COMMAND ----------

from mlflow.tracking import MlflowClient

def get_latest_model_uri(model_name):
    """
    Retrieve the URI of the latest version of a registered model.

    Args:
    model_name (str): The name of the registered model.

    Returns:
    str: The URI of the latest version of the model.
    """
    # Initialize an MLflow client
    client = MlflowClient()
    
    # Get the latest version of the model in the "None" stage
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
    
    # Return the source URI of the latest version
    return latest_version.source

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2024]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
# MAGIC |PyTorch|PyTorch and Caffe2 |https://github.com/pytorch/pytorch/blob/main/LICENSE|https://github.com/pytorch/|