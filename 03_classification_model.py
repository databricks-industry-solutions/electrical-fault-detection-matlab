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
# MAGIC - Load data and transform to prepare for training
# MAGIC - Train a Convolutional Neural Network (CNN) model
# MAGIC - Register trained model to MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Environment Set-up

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.1: Run setup notebook

# COMMAND ----------

# DBTITLE 1,Run Set-up Notebook
#Run Set-up Notebook
dbutils.notebook.run("x_Setup", 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.2: Define widgets

# COMMAND ----------

# DBTITLE 1,Create Widgets
dbutils.widgets.text("catalog", "main", "Catalog Name")
dbutils.widgets.text("schmea", "zach_jacobson", "Schema Name")

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.3: Create catalog and schema

# COMMAND ----------

# DBTITLE 1,Create Catalog & Schema
catalog_name = dbutils.widgets.get("catalog")
schema_name = dbutils.widgets.get("schema")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.4: Create mapping for fault output data

# COMMAND ----------

# DBTITLE 1,Create Mapping
# Orginal mapping from expore data
fault_types_orgi = {
    "0000": "No Fault",
    "1000": "Single Line to Ground A",
    "0100": "Single Line to Ground B",
    "0010": "Single Line to Ground C",
    "0011": "Line-to-Line BC",
    "0101": "Line-to-Line AC",
    "1001": "Line-to-Line AB",
    "1010": "Line-to-Line with Ground AB",
    "0101": "Line-to-Line with Ground AC",
    "0110": "Line-to-Line with Ground BC",
    "0111": "Three-Phase",
    "1111": "Three-Phase with Ground",
    "1011": "Line A Line B to Ground Fault",
}

# Map Fault codes to String Values. This will be our new target column
fault_types = {
    "z_z": "1",
    "a_c_g": "2",
    "c_g": "3",
    "a_g": "4",
    "a_b_c_g": "5",
    "b_g": "6",
    "a_b_g": "7",
    "b_c_g": "8",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Data Load and Transformation

# COMMAND ----------

# MAGIC %md
# MAGIC ####2.1: Define output fault class and location columns

# COMMAND ----------

# DBTITLE 1,Load Data
# Apply the UDF to add a new column with extracted sequences
df_fault_transformation = (
    spark.sql(f"select * from {catalog_name}.{schema_name}.fault_simulation")
    .withColumn(
        "fault_class",
        F.when(F.col("fault_class") == "unfaulted", "CB0_0_z_z").otherwise(
            F.col("fault_class")
        ),
    )
    .withColumn("Bus_loc_extr", F.split(F.col("fault_class"), "_"))
    .withColumn("Bus_loc", F.expr("try_element_at(Bus_loc_extr, 2)"))
    .withColumn(
        "extracted_letters", F.concat_ws("_", extract_udf(F.col("fault_class")))
    )
    .drop("Bus_loc_extr")
)

# Display the transformed DataFrame
df_fault_transformation.display()

# COMMAND ----------

# DBTITLE 1,Map Fields
# Map fault code and cast to an interger and add a temporary ID
inputs_pd = df_fault_transformation.withColumn(
    "fault_type_coded", map_codes(F.col("extracted_letters")).cast(IntegerType())
).withColumn("id_temp", F.monotonically_increasing_id())

# COMMAND ----------

# DBTITLE 1,Aggregate and Count Mappings
inputs_pd.groupBy("fault_type_coded").agg(F.count(F.col("fault_type_coded"))).display()

# COMMAND ----------

# Apply UDF to DataFrame, grouping by 'id' for demonstration purposes
# Since we are batching by fixed sizes rather than by ID, this example doesn't use the 'id' grouping directly for batch calculation.
df_with_batches = (
    inputs_pd.groupBy()
    .applyInPandas(
        add_batch_id,
        schema="V1mag_13 double,V1ph_13 double,V2mag_13 double, V2ph_13 double,V0mag_13 double,V0ph_13 double,I1mag_1 double,I1ph_1 double,I2mag_1 double,I2ph_1 double,I0mag_1 double,I0ph_1 double,fault_class string,Bus_loc string,extracted_letters string,fault_type_coded integer,id_temp long,batch_id integer",
    )
    .filter(F.col("batch_id") != "41")
)

# Show result
df_with_batches.sort("batch_id").display()

# COMMAND ----------

columns_to_round = [
    "V1mag_13",
    "V1ph_13",
    "V2mag_13",
    "V2ph_13",
    "V0mag_13",
    "V0ph_13",
    "I1mag_1",
    "I1ph_1",
    "I2mag_1",
    "I2ph_1",
    "I0mag_1",
    "I0ph_1",
]
df_with_batches_round = df_with_batches.select(
    "batch_id",
    "fault_type_coded",
    "extracted_letters",
    "Bus_loc",
    *[F.round(F.col(column), 3).alias(column) for column in columns_to_round]
)

# COMMAND ----------

### For Voltage Variables
inputs_pd_pandas = df_with_batches_round.toPandas()
plt.figure(figsize=(22, 5))
plt.plot(inputs_pd_pandas["V1mag_13"], "r", label="Va")
plt.plot(inputs_pd_pandas["V1ph_13"], "b", label="Vb")
plt.plot(inputs_pd_pandas["V1mag_13"], "g", label="Vc")

# Add legend to the plot
plt.legend()

# Set plot labels and title
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Voltage Over Time with Line Colors")

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ####2.2: Create probabilities for each fault class

# COMMAND ----------

# Here the class labels are pivoted to count occurences of each class per column. We will then create proabailities for each waveform/cycle (i.e. batch_id)
df_test = (
    df_with_batches_round.groupBy("batch_id")
    .pivot("fault_type_coded")
    .count()
    .fillna(0)
)

# Creating probabilities for each fault type class
df_target = (
    df_test.withColumn(
        "batch_class_total",
        F.col("1")
        + F.col("2")
        + F.col("3")
        + F.col("4")
        + F.col("5")
        + F.col("6")
        + F.col("7")
        + F.col("8"),
    )
    .withColumn(
        "fault_type",
        F.array(
            F.col("1") / F.col("batch_class_total"),
            F.col("2") / F.col("batch_class_total"),
            F.col("3") / F.col("batch_class_total"),
            F.col("4") / F.col("batch_class_total"),
            F.col("5") / F.col("batch_class_total"),
            F.col("6") / F.col("batch_class_total"),
            F.col("7") / F.col("batch_class_total"),
            F.col("8") / F.col("batch_class_total"),
        ),
    )
    .select("batch_id", "fault_type")
)

# COMMAND ----------

df_test.display()
df_target.display()

# COMMAND ----------

df = (
    df_with_batches.groupBy(F.col("batch_id"))
    .agg(
        F.collect_list(
            F.array(
                [
                    "V1mag_13",
                    "V1ph_13",
                    "V2mag_13",
                    "V2ph_13",
                    "V0mag_13",
                    "V0ph_13",
                    "I1mag_1",
                    "I1ph_1",
                    "I2mag_1",
                    "I2ph_1",
                    "I0mag_1",
                    "I0ph_1",
                ]
            )
        ).alias("features")
    )
    .join(df_target, ["batch_id"], "left")
)

df.display()

# COMMAND ----------

# Check to make sure each voltage array row is the same size. This is crucial for our input 3-D tensor to work later.
df.withColumn("feature_size", F.size("features")).select("feature_size").display()

# COMMAND ----------

df_locations = spark.sql(f"select * from {catalog_name}.{schema_name}.fault_locations").withColumn(
    "distance_sub_station",
    F.sqrt(F.pow(F.col("X_Coord"), 2) + F.pow(F.col("Y_Coord"), 2)),
)
df_locations.display()
inputs_pd.display()

# COMMAND ----------

# Shape I want to get into
x = df.select("features").rdd.map(lambda r: r[0]).collect()  # python list
features = np.array(x)

y = df.select("fault_type").rdd.map(lambda r: r[0]).collect()  # python list
labels = np.array(y)

# COMMAND ----------

# MAGIC %md
# MAGIC Definitions
# MAGIC DataLoader: Dataloaders are iterables over the dataset. So when you iterate over it, it will return B randomly from the dataset collected samples (including the data-sample and the target/label), where B is the batch-size 
# MAGIC
# MAGIC Pytorch_lighting.Trainer: 
# MAGIC * Automatically enabling/disabling grads,
# MAGIC * Running the training, validation and test dataloaders,
# MAGIC * Calling the Callbacks at the appropriate times,
# MAGIC * Putting batches and computations on the correct devices
# MAGIC
# MAGIC TorchDistributor: A class to support distributed training on PyTorch and PyTorch Lightning using PySpark.
# MAGIC
# MAGIC Lighting Required/Core Methods:https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Model Training

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.1: Set up deep learning environment

# COMMAND ----------

# DBTITLE 1,Set-up Deep Learning Environment
NUM_WORKERS = 2  # Number of worker nodes to use for distributed training

# NOTE: This assumes the driver node and worker nodes have the same instance type.
NUM_GPUS_PER_WORKER = (
    torch.cuda.device_count()
)  # Number of GPUs available per worker node
USE_GPU = NUM_GPUS_PER_WORKER > 0  # Boolean flag to indicate if GPUs are available

# Get the current Databricks username
username = spark.sql("SELECT current_user()").first()["current_user()"]

# Define the experiment path for MLflow tracking
experiment_path = f"/Users/{username}/pytorch-distributor"

# Retrieve the Databricks host URL for API calls
db_host = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .extraContext()
    .apply("api_url")
)

# Retrieve the Databricks API token for authentication
db_token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2: Define data module

# COMMAND ----------

# DBTITLE 1,Define Data Module
class FaultDataModule(pL.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling MNIST dataset.

    Args:
        dataset_object (tuple): A tuple containing train, validation, and test datasets.
        batch_size (int): Batch size for data loaders. Default is 554.
    """

    def __init__(self, dataset_object, batch_size: int = 554):
        super().__init__()
        self.batch_size = batch_size  # Set the batch size
        self.dataset_object = dataset_object  # Store the dataset object

    def setup(self, stage: str):
        """
        Setup datasets for different stages: train, validation, test, and predict.

        Args:
            stage (str): Stage of the setup ('fit', 'validate', 'test', 'predict').
        """
        mnist_full = self.dataset_object  # Full dataset (not used directly)
        self.mnist_test = self.dataset_object[2]  # Test dataset
        self.mnist_predict = self.dataset_object[2]  # Predict dataset (same as test)
        self.mnist_train, self.mnist_val = (
            self.dataset_object[0],
            self.dataset_object[1],
        )  # Train and validation datasets

    def train_dataloader(self):
        """
        DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        """
        DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        DataLoader for test data.

        Returns:
            DataLoader: DataLoader for test dataset.
        """
        return utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        """
        DataLoader for prediction data.

        Returns:
            DataLoader: DataLoader for prediction dataset.
        """
        return utils.data.DataLoader(self.mnist_predict, batch_size=self.batch_size)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.3: Define fault model training module

# COMMAND ----------

# DBTITLE 1,Fault Model Training Module
class Conv1DFaultModel(pL.LightningModule):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.scheduler = None
        # Define the 1D convolutional layer
        self.conv1d = nn.Conv1d(12, 32, 3, 1, padding=3 // 2)
        # Define the first fully connected layer
        self.dense1 = nn.Linear(277 * 32, 120)  # Hardcoded for now, can be improved
        # Define the second fully connected layer
        self.dense2 = nn.Linear(120, 60)
        # Define the third fully connected layer
        self.dense3 = nn.Linear(60, 8)  # Continue to add layers for better performance

    def forward(self, input_data):
        """
        Forward pass through the network.

        Args:
            input_data (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the network.
        """
        # Apply 1D convolution and ReLU activation
        output = F.relu(self.conv1d(input_data))
        # Apply max pooling
        output = F.max_pool1d(output, 2, 2)
        # Flatten the output for the fully connected layers
        shape1 = output.shape[1]  # Dimension 1 to flatten out for fully connected layer
        shape2 = output.shape[2]  # Dimension 2 to flatten out for fully connected layer
        output = output.view(-1, shape1 * shape2)
        # Pass through the first fully connected layer and apply ReLU activation
        output = F.relu(self.dense1(output))
        # Pass through the second fully connected layer and apply ReLU activation
        output = F.relu(self.dense2(output))
        # Pass through the third fully connected layer
        output = self.dense3(output)
        # Apply log softmax activation
        output = F.lsoftmax(output, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (tuple): Batch of data containing signals and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss value for the batch.
        """
        signals, labels = batch
        outputs = self(signals)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def val_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (tuple): Batch of data containing signals and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss value for the batch.
        """
        signals, labels = batch
        outputs = self(signals)
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Args:
            batch (tuple): Batch of data containing signals and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss value for the batch.
        """
        signals, labels = batch
        outputs = self(signals)
        loss = F.cross_entropy(outputs, labels)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            Optimizer: Configured optimizer.
        """
        optimizer = torch.optim.SGD(self.parameters(), 1e-4)
        return optimizer

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.4: Prepare labels

# COMMAND ----------

# DBTITLE 1,Prep Labels
torch.from_numpy(labels)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.5: Create training function

# COMMAND ----------

# DBTITLE 1,Create Training Function
def main_training_loop(features, labels, num_tasks, num_proc_per_task):

    """

    Main train and test loop

    """
    # add imports inside pl_train for pickling to work
    from torch import optim, nn, utils, Tensor
    from torchvision import datasets, transforms
    import pytorch_lightning as pl
    import mlflow
    import os

    ############################
    ##### Setting up MLflow ####
    # We need to do this so that different processes that will be able to find mlflow
    os.environ["DATABRICKS_HOST"] = db_host
    os.environ["DATABRICKS_TOKEN"] = db_token

    # NCCL P2P can cause issues with incorrect peer settings, so let's turn this off to scale for now
    os.environ["NCCL_P2P_DISABLE"] = "1"

    epochs = 5
    batch_size = 1
    dataset = split_set(features, labels)
    # init the autoencoder
    mlflow.pytorch.autolog()
    # Define the tensor specification
    input_tensor_spec = TensorSpec(
        type=np.dtype("float32"),
        shape=(40, 554, 12),  # Adjust to the correct data type your model expects
    )

    # Create an input schema using the tensor spec
    input_schema = Schema([input_tensor_spec])

    # Define the output schema if necessary; here we assume a basic float output
    # Modify according to your model's actual output
    output_schema = Schema([ColSpec("float", "fault")])

    # Create the model signature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    model = Conv1DFaultModel()
    datamodule = FaultDataModule(dataset, batch_size=1)

    # train the model
    if num_tasks == 1 and num_proc_per_task == 1:
        kwargs = {}
    else:
        kwargs = {"strategy": "ddp"}
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_proc_per_task,
        num_nodes=num_tasks,
        limit_train_batches=1000,
        max_epochs=epochs,
        **kwargs
    )
    if run_id is not None:
        mlflow.start_run(run_id=run_id)

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    return model, trainer.checkpoint_callback.best_model_path

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.6: Define model signature

# COMMAND ----------

# Define the tensor specification
input_tensor_spec = TensorSpec(
    np.dtype("float32"), (-1, 12, 554)
)  # Adjust to the correct data type your model expects

# Create an input schema using the tensor spec
input_schema = Schema([input_tensor_spec])

# Define the output schema if necessary; here we assume a basic float output
# Modify according to your model's actual output
output_tensor_spec = TensorSpec(np.dtype("float32"), (-1, 8))

output_schema = Schema([output_tensor_spec])

# Create the model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

print("Input schema for MLflow:")
print(signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.7: Track training experiment through MLflow

# COMMAND ----------

NUM_TASKS = 1
NUM_PROC_PER_TASK = 1
NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK


# Start an MLFlow run to automatically track our experiment
with mlflow.start_run() as run:
    run_id = mlflow.active_run().info.run_id
    # TorchDistributor allows us to easily distribute our taining to multiple nodes in a cluster
    (model, ckpt_path) = TorchDistributor(
        num_processes=NUM_PROC, local_mode=True, use_gpu=USE_GPU
    ).run(main_training_loop, features, labels, NUM_TASKS, NUM_PROC_PER_TASK)

    # Log the artifact to MLFlow with a signature that tells consumers what to pass in to the model and what to expect as output
    mlflow.pytorch.log_model(
        artifact_path="model",
        pytorch_model=model,
        signature=signature,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.8: Log trained model to MLflow

# COMMAND ----------

model_name = "matlab_fault_prediction"
latest_model_uri = get_latest_model_uri(model_name)

# COMMAND ----------

model_name = "matlab_fault_prediction"

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# Load the model
model_uri = latest_model_uri #"runs:/294298fd411c45cbbb87ea73e082410a/model"
model = mlflow.pyfunc.load_model(model_uri)
feature = features.astype(np.float32)
# Infer the model signature

# Register the model with the specified signature
mlflow.register_model(model_uri, f"{catalog_name}.{schema_name}.{model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps to try:
# MAGIC
# MAGIC Train a CNN Regression model to predict fault locations. You can use fault_locations data generated from MATLAB to train a regression model using the distance data frame. You can use this [article](https://bamblebam.medium.com/audio-classification-and-regression-using-pytorch-48db77b3a5ec) as reference.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## References
# MAGIC * https://github.com/tensorflow/ecosystem/blob/master/spark/spark-tensorflow-distributor/examples/simple/example.py
# MAGIC * https://github.com/nicknochnack/DeepAudioClassification/blob/main/AudioClassification.ipynb
# MAGIC * https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb
# MAGIC * https://www.kaggle.com/competitions/PLAsTiCC-2018
# MAGIC * https://github.com/databricks-industry-solutions/utilities-cv
# MAGIC * https://github.com/databricks-industry-solutions/utilities-cv/blob/main/03_MultiGPU_ModelTraining.py
# MAGIC * https://github.com/CSCfi/pytorch-ddp-examples/blob/master/mnist_lightning_ddp.py
# MAGIC * https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_CNN.py
# MAGIC * https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.torch.distributor.TorchDistributor.html
# MAGIC * https://www.youtube.com/watch?v=eQvI5eAL0nA
# MAGIC * https://github.com/mlflow/mlflow/blob/master/examples/pytorch/MNIST/mnist_autolog_example.py
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2024]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
# MAGIC |PyTorch|PyTorch and Caffe2 |https://github.com/pytorch/pytorch/blob/main/LICENSE|https://github.com/pytorch/|