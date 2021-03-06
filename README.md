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

**An overview of parameters used in the AutoML configuration.**
experiment_tmeout_minutes=30 - This is the maximum amount in minutes that the AutoML has to complete iterations before the experiment is ended.  This was set to 30 minutes due to resource constraints.

task='classification' - This defines the type of model you would like to run.  In this case it will run a classification task.  Regression or forecasting can also be used.

primary_metric='accuracy' - This is the metric that AutoML will optomize and will lead to the selection of the best model.

training_data=final_training_data - This is where the training dataset is fed to AutoML.

label_column_name='y' - This specifies the column name that the prediction must be done on.

n_cross_validations=6 - This sets the number of cross validations that AutoML must perform when no validation dataset is provided.

compute_target=cpu_cluster - This defines which compute target AutoML should use to run the experiment on.

enable_early_stopping=True - This defines if AutoML will enforce an early stopping policy.

enable_onnx_compatible_models=True - This allows for Open Neural Network Exchange compatibility.  This compatibility is useful for the inferencing stage of a model.  This allows inferencing accros multiple frameworks without optomizing hardware.

## Pipeline comparison
It is clear that AutoML delivered a model with better accuracy using a Votin Ensemble.  The Voting Ensemble delivered an accuracy 0f 91.77% compared to using logistic regression and HyperDrive which delivered an accuracy of 90.89%.

The bigest difference is that AutoML automates the execution of multiple models.  If I want to do the same using HyperDrive, I would need to manually set up a pipeline for each model.
## Future work
Future work can include making sure that the data is balanced to ensure that predictions are not skewed.  Further investigation can be made on the data to ensure that there are no bias towards certain groups of people.

Looking at other metrics other than accuracy could provide better insights or predictions.  For example Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, etc.

**Why balancing of data would improve the model?**
When data sets are balanced, acuracy of models increase.  Unbalanced classes mean that the number of observations per class is not equally distributed.  This causes one class to overpower the other classes and the prediction will be biased towards the class with more observations.  Unbalanced datasets can be balanced by resampling, either oversampling or undersampling.  Undersampling is the process where samples are randomly removed from the class that has too many observations.  Overersampling is the process where data is randomly generated to add observations to the class that has fewer observations.

**Why removing bias from data would improve the model?**
Bias in datasets causes certain elements in datasets to have a higher weighting than other elements.  Bias causes skewed predictions and lower accuracy of models.  Removing any bias from the data will ensure a model representative of the real world and higher accuracy.  Bias can be removed by using diverse data sets and making bias testing part of the model development.

**Why looking at other metrics would improve the model?**
Accuracy is traditianally used to measure classification models and metrics like Root Mean Squared Error is used to measure regression models.  By using other metrics like the Root Mean Squared Error for this model could provide a better comparison to the regression model used with Hyperdrive. 

## Proof of Cluster clean up
![image](https://user-images.githubusercontent.com/77330289/143196429-e0646b16-f054-4780-a1ec-5e38fd87fa60.png)
![image](https://user-images.githubusercontent.com/77330289/143196443-bdf664a7-4ca3-4549-a161-af9507370dd2.png)




