# Detecting credit card fraud via machine learning

My capstone project for Udacity's Machine Learning Engineer Nanodegree focuses on detecting instances of credit card fraud. The objective of this project falls into the category of anomaly detection, where classification is attempted on a dataset with a highly imbalanced distribution of classes. In the case of the dataset used here instances of fraud account for only 0.172% of all instances. Anomaly detection is a fascinating area of machine learning that often requires special techniques and care in interpretation compared to other machine learning use cases. 

## Dataset
### Overview
The dataset used in this project is the openly available Kaggle credit card fraud dataset (https://www.kaggle.com/mlg-ulb/creditcardfraud). This dataset consists of the results of a PCA transformation on the original credit card transaction data. PCA was carried out to protect sensitive information present in the original data. In addition to the principle coordinates the data contains the time since the first transaction that was logged and the amount (in $) of the transaction. It is a binary classification problem where the positive label (1) corresponds to an instance of fraud and the negative label (0) is a normal transaction. The dataset is highly imbalanced with very few instances of fraud present.

### Task
The objective of this project is to classify samples as either 'normal' or 'fraudulent' transactions. In order to do this I will use all available features present in the dataset. 

### Access
In order to access the Kaggle dataset, I downloaded the compressed dataset, extracted it, and registered it as a dataset in my Azure workspace. From this point I programmatically interacted with it through the SDK.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
