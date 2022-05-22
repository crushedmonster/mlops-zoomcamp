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
Ensure Anaconda that is installed. For this coursework, I'll be making use of virtual conda environment to avoid any dependencies conflicts on my local.

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
