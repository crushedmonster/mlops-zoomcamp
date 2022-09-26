# Module 3 Notes
By Wenna Loo

## Negative Engineering and Workflow Orchestration
**What is workflow orchestration?**

It's a set of tools that schedule and monitor work that you want to accomplish. 

For example, if you have a machine learning pipeline that you want to run every week, you put it on a schedule and if that machine learning model fails or maybe the data coming in fails, then you want observability into the issues that caused to fail so that you can fix them.

Example of a Machine Learning Pipeline:
```
PostgreSQL → Parquet 
              ↓ 
Rest API →  Pandas → Sklearn → mlflow
                        ↓
                      Flask 
```

In general, random points of failure can occur within the pipeline. The goal of the workflow orchestration is to both minimize the errors and to fail gracefully if it happens.

As we look at more interconnected pipelines, there are more places that things can fail because there's alot more activity going on. So, it is important to have some failure mechanism to deal with it, or if a step fails we can still run other steps. Workflow orchestration also dictates whether we want downstream tasks to run or not.

## Negative Engineering
**90% of engineering time is spent on:**
- Retries when APIs go down
- Malformed data
- Notifications
- Observability into Failure
- Conditional Failure Logic
- Timeouts
