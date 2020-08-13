# Udacity Machine Learning Engineer Nanodegree Capstone Project 

## The Starbuck's Challenge 

The Capstone Project has the objective of genuively applying all the knowledge learned on the Nanodegree. In this case, what we would be mostly applying is the High Level Interface to deploy an estimator and building a custom Scikit-Learn model. 

For a description of the Starbuck's Challenge and the data provided you can consult `notebooks/Starbucks_Capstone_noteboo`

## Contents 

We provide a brief description of these repository files. 

* `proposal.pdf`. File that describes the project one decides to work on with a machine learning solution. 

* `project.pdf`. Summary of major insights gained in the Exploratory Data Analysis, Confirmatory Data Analysis, Feature Engineer, Model Building and Deployment. This could be taken as a reference guide, if you want to further dig on the code you can consult the respective `notebooks` folder. 

* `helpers.py`. Under the directory `helpers` it includes additional functions to process data. 

* `train.py`. Under `source_sklearn` directory contains the script for training a Scikit-Learn model through a SageMaker Estimator. 

* `Data_Exploration_and_Confirmatory_Data_Analysis.ipynb`. EDA and CDA of data 

* `Feature_Engineer.ipynb`. Feature Engineer of data, for a description see the Section Engineering Features on `project.pdf`. 

* `Model_Building`. Model Selection and Evaluation, for a description see the section `Why a Bayesian approach` and `Bootstraping data` on `project.pdf`. 

* `Model_Deploying`. Deploying the custom scikit-learn model through a SageMaker Endpoint. 

## Requirements

To smoothly run the code first install the requirements file. Most of the notebooks could be run outside a SageMaker Instance, however, for `Model_Deploying` it would be required one to run the notebook.   

`pip install requirements.txt`

## Code Guidelines

To format the python scripts, [black](https://github.com/psf/black) was used and then check for an issue with [pylint](https://pypi.org/project/pylint/). An example is provided

```
black helpers.py 
pylint helpers.py
```

Most of the code is commented. For example, if instantiating an estimator, one writes

```
# Initialize Bayesian Model
        clf = BayesianRidge(
                compute_score=True, 
                normalize=True
              )
``` 

        