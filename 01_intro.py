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
# MAGIC Power outages, if not addressed promptly, can have adverse effects on consumers. Utility companies are consistently striving to mitigate outages through preventive measures such as infrastructure maintenance, tree trimming, and other proactive strategies. However, when an outage does occur, it is crucial for these companies to efficiently and accurately identify its root cause to minimize the resulting impact. This solution accelerator provides a head start to use MATLAB to simulate electrical fault data and use it to train a deep learning model to predict faults on Databricks. 
# MAGIC
# MAGIC **Authors**
# MAGIC - Zachary Jacbson
# MAGIC - Jenny Park
# MAGIC - Drew Triplett
# MAGIC - Karthiga Mahalingam

# COMMAND ----------

# MAGIC %md
# MAGIC ## About This Series of Notebooks
# MAGIC - This series of notebooks is intended to help you use Databricks and MATLAB to classify faults in electrical data. 
# MAGIC - In support of this goal, we will:
# MAGIC - Setup MATLAB Runtime on Databricks cluster
# MAGIC - Generate fault simulation data using MATLAB and Simulink on Databricks clusters
# MAGIC - Write output data to Delta table in Unity Catalog
# MAGIC - Use the output to train a Convolutional Neural Networks (CNNs) to predict faults.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Power of Databricks and MATLAB
# MAGIC <img src ="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/MATLAB-databricks-interface.png?raw=true" width="1000">
# MAGIC
# MAGIC Reference: https://www.mathworks.com/solutions/partners/databricks.html <br>
# MAGIC Kit (JDK) 1.8 or later
# MAGIC
# MAGIC The MATLAB interface for Databricks® enables MATLAB® and Simulink® users to connect to data and compute capabilities in the cloud. Users can access and query big datasets remotely or deploy MATLAB code to run natively on a Databricks cluster.
# MAGIC
# MAGIC ## Features
# MAGIC
# MAGIC * Access data via JDBC
# MAGIC * Interactively access a Databricks cluster using DBConnect
# MAGIC * Access data via DBFS
# MAGIC * Control Databricks clusters and operation via the REST API
# MAGIC * **Deploy compiled MATLAB applications to a Databricks cluster**
# MAGIC * **Deploy compiled MATLAB functions as libraries on a Databricks cluster**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## About the data 
# MAGIC Using the simulation model developed in MATLAB, we generate a set of fault sequences to be used to train our deep learning model. Since this data gets written to a Delta table in Unity Catalog, we can interact with the data and share across Databricks workspaces with appropriate permissions. 
# MAGIC
# MAGIC Below are screenshots pertaining to the generated data. 
# MAGIC Here is what the schema of the MATLAB model output looks like: <br>
# MAGIC <img src ="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/MATLAB-generated-data-schema.png?raw=true" width="300">
# MAGIC
# MAGIC This shows the output target variable. The target variable “fault_class” is describing what type of fault occurs when the following features are measured. <br>
# MAGIC <img src ="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/MATLAB-generated-data-fault-class.png?raw=true" width="200">
# MAGIC
# MAGIC This screenshot shows data with the location (X_coord,Y_coord) where the fault occurred based from the substation (0,0) coordinate. You can use the distance formula to calculate the exact fault location in feet. <br>
# MAGIC <img src ="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/MATLAB-generated-data-fault-locations.png?raw=true" width="500">
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Step 1: Install MATLAB
# MAGIC
# MAGIC 1. **macOS**
# MAGIC     1. Install [Parallels](https://www.parallels.com/) software for Windows.
# MAGIC     2. From [MathWorks Downloads](https://www.mathworks.com/downloads/), select a MATLAB release for Windows and download the installer.
# MAGIC     3. After downloading follow the instructions [here](https://www.mathworks.com/matlabcentral/answers/1606600-matlab-using-parallels-on-m1-macbook-pro) to install and unzip MATLAB appropriately.
# MAGIC         1. Unzip the downloaded file, taking note of the temp folder the files are to be unzipped into (similar to "_temp_matlab_R2022a_win64")
# MAGIC         2. Go into <temp folder>\bin\win64 and double-click on MathWorksProductInstaller (Path might look similar to this: \Downloads\_temp_matlab_R2022a_win64\bin\win64\MathWorksProductInstaller)
# MAGIC     4. Follow the instructions from the installer to complete installation.
# MAGIC     5. [Start MATLAB](https://www.mathworks.com/help/matlab/matlab_env/start-matlab-on-windows-platforms.html).
# MAGIC 2. **Linux**
# MAGIC     1. If installing MATLAB on Mac or Windows machine:
# MAGIC         1. Launch EC2 instance
# MAGIC         2. Install ubuntu on the instance
# MAGIC         3. Configure port for open traffic
# MAGIC         4. Generate .pem file for ssh
# MAGIC         5. SSH into EC2 instance <br>
# MAGIC             `zach.jacobson@L44CFYWMW4 downloads % chmod 600 matlab_zj.pem` <br>
# MAGIC             `zach.jacobson@L44CFYWMW4 downloads % ssh -i matlab_zj.pem ubuntu@54.67.118.95` <br>
# MAGIC             For remote access using GUI Desktop, [follow these insructions](https://shrihariharidas73.medium.com/how-to-setup-gui-desktop-with-ubuntu-on-aws-ec2-ea713d836a58).
# MAGIC         6. Follow steps below in 2. to install MATLAB on the instance
# MAGIC     2. If installing MATLAB directly on Linux:
# MAGIC         1. From [MathWorks Downloads](https://www.mathworks.com/downloads/), select a MATLAB release for Windows and download the installer.
# MAGIC         2. Unzip the downloaded installer files and navigate to the unzipped folder. For example, use these commands to unzip the installer for release R2024b to a folder of the same name, and then navigate to the folder.
# MAGIC
# MAGIC             `unzip matlab_R2024b_Linux.zip -d ./matlab_R2024b_Linux` <br>
# MAGIC             `cd ./matlab_R2024b_Linux`
# MAGIC         3. In the installation folder, run the install script and follow the prompts to complete the installation.
# MAGIC
# MAGIC             `xhost +SI:localuser:root` <br>
# MAGIC             `sudo -H ./install` <br>
# MAGIC             `xhost -SI:localuser:root`
# MAGIC
# MAGIC             sudo is required only when you install products to a folder where you do not have write permissions, which might include the default installation folder. The xhost commands are required only when you install products as the root user with sudo. These commands temporarily give the root user access to the graphical display required to run the installer.
# MAGIC
# MAGIC             Default installation folder: /usr/local/MATLAB/R20XXy
# MAGIC         4. [Start MATLAB](https://www.mathworks.com/help/matlab/matlab_env/start-matlab-on-linux-platforms.html)
# MAGIC
# MAGIC 3. **Windows**
# MAGIC     1. From [MathWorks Downloads](https://www.mathworks.com/downloads/), select a MATLAB release for Windows and download the installer.
# MAGIC     2. Double-click the downloaded installer and follow the prompts to complete the installation. <br>
# MAGIC     Default installation folder: C:\Program Files\MATLAB\R20XXy
# MAGIC     3. [Start MATLAB](https://www.mathworks.com/help/matlab/matlab_env/start-matlab-on-windows-platforms.html).
# MAGIC
# MAGIC
# MAGIC     <img src="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/MATLAB-login-screen.png?raw=true" width="700">
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Download MATLAB Interface for Databricks
# MAGIC
# MAGIC 1. Go to [MathWorks Databricks partners page](https://www.mathworks.com/solutions/partners/databricks.html#) and scroll all the way to the bottom to click on "Download the MATLAB Interface for Databricks".
# MAGIC
# MAGIC 2. Extract the compressed zipped folder **“matlab-databricks-v4-0-7-build-...”** inside Program Files\\MATLAB. Once extracted you will see the **“matlab-databricks”** folder. Make sure the folders are in this folder and this hierarchy:  
# MAGIC <img src="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/MATLAB-databricks-download-screen.png?raw=true" width="700">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Install MATLAB Runtime on Databricks cluster
# MAGIC
# MAGIC 1. Launch MATLAB application from local Desktop application through Search bar and make sure to **run as an administrator** for overwrite of files and folders  
# MAGIC
# MAGIC 2. Go to the command line interface (CLI) in MATLAB and type “ver” to verify you have all the dependencies necessary:  
# MAGIC    <img src="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/MATLAB-ver-screenshot.png?raw=true" width="700">
# MAGIC
# MAGIC 3. Navigate to this path: C:\\Program Files\\MATLAB\\matlab-databricks\\Software\\MATLAB <br>
# MAGIC       `>> cd ‘C:\\Program Files\\MATLAB\\matlab-databricks\\Software\\MATLAB’` <br>
# MAGIC       Make sure the path you see in MATLAB at the top bar matches the path above.
# MAGIC 4. Run `>> install(download="14.3")`
# MAGIC 5. You will be prompted with several questions for configuring the cluster spin up. Enter authentication method, other methods can be configured manually following installation. 
# MAGIC        Choose one of: Chain, DotDatabricksConnect, PAT, Basic, OauthM2M or OauthU2M, \[PAT\]: <br>
# MAGIC        <img src="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/MATLAB-auth-screenshot.png?raw=true" width="700"> <br>
# MAGIC       You should see your cluster appear in Databricks.   
# MAGIC 6. Next you will receive this question: Enter the local path to the downloaded zip file for this package. 
# MAGIC       Point to the one on your local machine, for e.g., C:\\Users\\someuser\\Downloads\\matlab-databricks-v1.2.3\_Build\_A1234567.zip  
# MAGIC 7. A job will be created in Databricks automatically as shown below **(Make sure the job timeout is set to 30 minutes or greater to avoid timeout error)**  
# MAGIC       <img src="https://github.com/karthiga19/Fault-detection-matlab-databricks-images/blob/main/Databricks-job-screenshot.png?raw=true" width="700">
# MAGIC 8. Go to Workspace-\>Mathworks-\>4.0.7-\>runtime. Click on three dots and click “Copy URL/path" and go to the copied path
# MAGIC 9. You now have a Databricks cluster with MATLAB runtime installed. There are no additional steps required when using the cluster in future. It should have the necessary spark env variables and init script pointing to the shell file that was generated from the job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Connect to Databricks from MATLAB
# MAGIC
# MAGIC 1. Attach the below ENV variables and init script to the Databricks cluster
# MAGIC * Env Variables:  
# MAGIC   * LD\_LIBRARY\_PATH=/usr/local/MATLAB/MATLAB\_Runtime/R2024a/runtime/glnxa64:/usr/local/MATLAB/MATLAB\_Runtime/R2024a/bin/glnxa64:/usr/local/MATLAB/MATLAB\_Runtime/R2024a/sys/os/glnxa64:/usr/local/MATLAB/MATLAB\_Runtime/R2024a/extern/bin/glnxa64:/usr/local/MATLAB/MATLAB\_Runtime/R2024a/sys/opengl/lib/glnxa64 MW\_CONNECTOR\_CONNECTION\_PROFILES=noop PYSPARK\_PYTHON=/databricks/python3/bin/python3  
# MAGIC * INIT SCRIPT  
# MAGIC   * /Users/zach.jacobson@[databricks.com/MathWorks/4.0.7/runtime/runtime\_install\_r2024a.sh](http://databricks.com/MathWorks/4.0.7/runtime/runtime_install_r2024a.sh)   
# MAGIC
# MAGIC 2. Open Databricks notebook and move files from dbfs to volumes: <br> `dbutils.fs.cp("dbfs:/MathWorks", "/Volumes/main/zach\_jacobson/mathworks", recurse\=True)` <br>
# MAGIC This is because Databricks’ policy wipes the zip file in dbfs zip file. (This path can be found when you click on runtime\_install\_r2024a.sh)  
# MAGIC     Example: RUNTIME\_ZIP="/dbfs/MathWorks/runtime/MATLAB\_Runtime\_R2024a\_Update\_1\_glnxa64.zip"    
# MAGIC
# MAGIC 3. Build the wheel file. This creates a Python library, and then generates wrapper code that will make it easier to use the underlying compiled MATLAB functions in a Spark context. To build the wheel file in MATLAB, run the following command in CLI:
# MAGIC
# MAGIC     `>> PSB = build_python` <br>
# MAGIC     (Refer [here](https://github.com/mathworks-ref-arch/matlab-spark-api/blob/main/Documentation/PythonSparkBuilder.md](https://github.com/mathworks-ref-arch/matlab-spark-api/blob/main/Documentation/PythonSparkBuilder.md) for details.)   
# MAGIC     ***Make sure you have pip3 installed. Use this command in CLI:*** <br> `>> !sudo apt install python3-pip`
# MAGIC
# MAGIC     Notes:
# MAGIC     - Make sure matlab-databricks is not in MATLAB Runtime  
# MAGIC     - Run the .prj file to set up all the variables  
# MAGIC     - Run the .slx file and press Ctrl+D to refresh the model  
# MAGIC     - Run the build\_python\_function.m to build the wheel
# MAGIC
# MAGIC 4. Upload wheel file to databricks workspace: <br> `PSB.uploadWheelToDatabricks(“/dbfs/MathWorks”,”cluster_id”)` <br> Use volumes path above after uploaded to dbfs. Use same command: <br> `dbutils.fs.cp("dbfs:/MathWorks", "/Volumes/main/zach_jacobson/mathworks", recurse=True)`

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2024]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
# MAGIC |PyTorch|PyTorch and Caffe2 |https://github.com/pytorch/pytorch/blob/main/LICENSE|https://github.com/pytorch/|