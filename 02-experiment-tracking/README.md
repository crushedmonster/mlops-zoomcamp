# Module 2 Notes
By Wenna Loo

## Experiment Tracking
**Concepts:**
- ML experiment: the process of building an ML model
- Experiment run: each trial in an ML experiment
- Run artifact: any file associated with an ML run
- Experiment metadata: metadata tied to each experiment

**What's experiment tracking?**

Experiment tracking is the process of keeping track of all the **relevant information** from an **ML experiment**, which includes:
- Source code
- Environment
- Data
- Model
- Hyperparameters
- Metrics

**Why is experiment tracking so important?**

Experiment tracking helps with **Reproducibility, Organization and Optimization**.

## MLflow
Definition as per their [official documentation](https://mlflow.org/): "An open source platform for the machine learning lifecycle"

In practice, it's just a Python package that can be installed with pip, and it contains four main modules:
- Tracking
- Models
- Model Registry
- Projects *(Out of scope of this course)*

### Tracking experiments with MLflow
The MLflow tracking module allows you to organise your experiments into runs, and to keep track of:
- Parameters
- Metrics
- Metadata
- Artifacts
- Models

Along with this information, MLflow automatically logs extra information about the run:
- Source code
- Verson of the code (git commit)
- Start and end time
- Author

### Getting started with MLFlow
In the virtual conda environment created previously in [module 1](https://github.com/crushedmonster/mlops-zoomcamp/tree/main/01-intro), install the package with `conda`:

```sh
conda install -c conda-forge mlflow
```

To install the package with `pip`, simply run:

```sh
pip install mlflow
```

### MLflow UI
To run the MLflow UI locally we can run the following the command:

```sh
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
The backend storage is essential to access the features of MLflow. Here, we're telling mlflow that we want to store all the artifacts and metadata in SQLite, which is one of the alternatives for the backend store.

The UI should accessible via: http://localhost:5000/ .

### Running MLFlow in Jupyter Notebook
To import MLflow, simply just import the library:

```python
import mlflow
```

Next, we'll also need to set tracking URI. This is needed because we are running MLFlow with the sqlite backend.

```python
 mlflow.set_tracking_uri("sqlite:///mlflow.db")
 ```
 
 We'll also need to set the experiment. 
 
 ```python
 mlflow.set_experiment("my-branch-new-experiment")
 ```
 
MLFlow will check if this experiment exists, and if it doesn't exist it will create the experiment and then assign all the runs to that experiment.
If the experiment already exists, MLFlow will just append the runs to this existing experiment.

To start tracking our runs, we'll need to define the run with mlflow.start_run(). 

```python
# to track the run with mlflow
with mlflow.start_run():
    ...
 ```
 
 To log information about a run, we can set tags associated to it. One example of could be the name of the developer :
 
 ```python
 # set tag for mlflow
mlflow.set_tag("developer", "Wenna")
 ```
This will be useful when working in a big team or useful to find some runs from a specific person.
