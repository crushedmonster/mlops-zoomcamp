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

To log information about which datasets were used for the run, we can simply pass the data path of the dataset using log_params.

```python
# log information about the datasets used
mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")
```

In addition, we can also use log_params to log information about the hyperparameters used during model training.

```python
# log the hyperparameter alpha (for training a LASSO model)
alpha = 0.01
mlflow.log_param("alpha", alpha)
```

To keep track on the performance of the model, we can simply pass the metric of interest using log_metric. Eg. We are interested to monitor the RMSE score of the model:

```python
# calculate and log RMSE score
rmse = mean_squared_error(y_val, y_pred, squared=False)
mlflow.log_metric("rmse", rmse)
```

Finally, putting it altogether:

```python
with mlflow.start_run():
    # set tag for mlflow
    mlflow.set_tag("developer", "Wenna")
    
    # log information about the datasets used
    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")
    
    # log the hyperparameter alpha
    alpha = 0.01
    mlflow.log_param("alpha", alpha)
    lr = Lasso(alpha)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_val)
    # calculate and log RMSE score
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)
```

If we go to MLFlow UI, we should see that our new run has been recorded (click "refresh" if it did not show up).


### Hyperparameter Optimization Tuning
The following example shows how we can optimise the hyperparamters of XGBoost model using hyperopt and log all the information into MLFlow.

```python
import xgboost as xgb

# hyperopt is a library that uses some bayesian methods to find the best set of hyperparameters
# docs: http://hyperopt.github.io/hyperopt/
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# fmin: tries to minimise objective function
# tpe: algorithm that controls the logic to find the best set of hyperparameters
# hp: a library that contains a bunch of different methods to define the search space. eg. the ranges for each hyperparameter that we will use
# STATUS_OK: a signal that we will send at the end of each run to tell hyperopt that the objective function has run successfully
# Trials: will keep track of information from each run
from hyperopt.pyll import scope
# scope: used to define a range of type integer

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

# define an objective function
def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}

search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}

# minimize the objective over the space
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)
```
