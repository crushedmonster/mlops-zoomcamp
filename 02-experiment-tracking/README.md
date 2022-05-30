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
