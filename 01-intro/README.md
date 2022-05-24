# Module 1 Notes
By Wenna Loo

> Running on Windows 10 Home 64-bit operating system, locally.

## Environment Setup
### Installing Anaconda

Proceed to skip this step if anaconda has already been installed.
- https://www.anaconda.com/products/individual#Downloads
- Use 64-Bit Graphical Installer
- Add Anaconda to PATH environment variable

To check if anaconda is installed properly:
- Launch jupyter notebook using the command prompt *(Note: that anaconda has to be added to PATH environment variable)*
```sh
jupyter notebook
```
### Installing Docker
- Install the official Docker for Desktop from the website (https://docs.docker.com/docker-for-windows/install/)

### Create a virtual conda environment
Ensure Anaconda is installed. For this coursework, I'm using virtual conda environment to avoid any dependencies conflict on my local.

```sh
# Create a new Conda environment
conda create -n mlops-zoomcamp python=3.9.7
# Activate the new environment
conda activate mlops-zoomcamp
# Install any relevant packages required
conda install pandas
conda install jupyter
```

## Dataset
The New York taxi dataset for this module is downloaded from: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Download Green Taxi Trip Records for January and February 2021 
- Download For-Hire Vehicle Trip Records for January and February 2021 (This is used to complete the homework)

## What is MLOps?
MLOps is a set of best practices to put machine learning models into production.

## MLOps Maturity Model
- Level 0: No MLOps
  - All codes are mainly in jupyter notebook
  - No automation (eg. No proper pipelines/ experimental tracking)
  - Typically at a POC stage
- Level 1: Devops, No MLOps
  - Releases are automated
  - Unit and integration tests
  - CI/CD
  - Operational metrics
  - However, only best practices from software engineering are applied. There is still no experiment tracking and reproducibility.
  - Data Science team is separated from Devops Engineering team
  - Typically, this is a stage where you have a POC and are moving it to production
- Level 2: Automated training
  - Training pipeline
  - Experiment tracking
  - Model registry (to track currently deployed models)
  - Low friction deployment
  - At this point, the Data Science team works together with the Devops Engineering team (part of the same team)
  - Usually at this stage, your organisation will have multiple models (eg. three or more use cases) running in production so it makes sense to invest in infrastructure for automated training.
- Level 3: Automated model deployment
  - Easy to deploy model
  - Also, at this stage the ML platform which hosts your model has capabilities of running A/B tests.
  - Models can be monitored
- Level 4: Full MLOps
  - Full automation with no human involved
  - Models are easily monitored

Reference: [Machine Learning operations maturity model - Microsoft Docs](https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)

## Homework
My solution can be found here: [homework.ipynb](https://github.com/crushedmonster/mlops-zoomcamp/blob/main/1-intro/homework.ipynb)
