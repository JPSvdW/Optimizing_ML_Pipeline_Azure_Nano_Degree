# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains information on potential target customers collected from market research done for a bank.  The predicted result wil be done on the deposit column, whether a customer would make a deposit or not.

The best performing model was found to be the AutoML model with a the best performing pipeline being a VotingEnsemble and an accuracy of 91.77%, the accuracy of the logistic regression model implemented using Hyperdrive was slightly lower with an accuracy of 90.89%.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

I accessed the data from the following URL: https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv using the TabularDatasetFactory which creates a tabular dataset.

The data imported is then cleaned using a clean_data function.

The cleaned data is then split into a training and test set with a 80:20 split ratio.

A Scikit-learn logistic regression model is used.  A parameter sampler is set up.  Two hyperparameters are tuned (C and max_iter).  C is called a inverse regularisation parameter and max_iter is the number of iterations.

A SKLearn estimator and parameter sampler is created to configure the HyperDrive.

The experiments are submitted and the accuracy is created for all of the runs and the best model is saved.

**What are the benefits of the parameter sampler you chose?**
I made use of the RandomParameterSampling, this reduces the computation time needed and gives a reasonable result.  If GridParameterSampling where to be used, a lot more computational time would have been needed.  This sampler also supports early stopping.

**What are the benefits of the early stopping policy you chose?**
I made use of a BanditPolicy.  A bandit policy is based on a slack factor or slack amount and evaluation interval.  Runs are ended when the primary metric is outside of the specified slack facto or slack amount of the best run.  This is an non-conservative policy which saves on computational time.

## AutoML
AutoML allowed me to run multiple experiments and chose the best model.

A total of 30 models were executed.  The best model happened to be the VotingEnsemble and had a slightly better accuracy of 91.77%.  This is a majority voting ensemble which combines the predictions from multiple models.  This wil give a better performing model than using a single model that was used in the ensemble.

## Pipeline comparison
It is clear that AutoML delivered a model with better accuracy using a Votin Ensemble.  The Voting Ensemble delivered an accuracy 0f 91.77% compared to using logistic regression and HyperDrive which delivered an accuracy of 90.89%.

The bigest difference is that AutoML automates the execution of multiple models.  If I want to do the same using HyperDrive, I would need to manually set up a pipeline for each model.
## Future work
Future work can include making sure that the data is balanced to ensure that predictions are now skewed.  Further investigation can be made on the data to ensure that there are no bias towards certain groups of people.

Looking at other metrics other than accuracy could provide better insights or predictions.  For example Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, etc.

