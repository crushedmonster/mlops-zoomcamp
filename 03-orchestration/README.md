# Module 3 Notes
By Wenna Loo

## Negative Engineering and Workflow Orchestration
**What is workflow orchestration?**

It's a set of tools that schedule and monitor work that you want to accomplish. 

For example, if you have a machine learning pipeline that you want to run every week, you put it on a schedule and if that machine learning model fails or maybe the data coming in fails, then you want observability into the issues that caused to fail so that you can fix them.

Example of a Machine Learning Pipeline:
```
PostgresQL → Parquet 
              ↓ 
Rest API →  Pandas → Sklearn → mlflow
                        ↓
                      Flask 
```

In general, random points of failure can occur within the pipeline. The goal of the workflow orchestration is to both minimize the errors and to fail gracefully if it happens.
