# Detecting credit card fraud via machine learning

My capstone project for Udacity's Machine Learning Engineer Nanodegree focuses on detecting instances of credit card fraud. The objective of this project falls into the category of anomaly detection, where classification is attempted on a dataset with a highly imbalanced distribution of classes. In the case of the dataset used here instances of fraud account for only 0.172% of all instances. Anomaly detection is a fascinating area of machine learning that often requires special techniques and care in interpretation compared to other machine learning use cases. 

*See a video outlining model training, deployment, and interaction here.*

## Dataset
### Overview
The dataset used in this project is the openly available Kaggle credit card fraud dataset (https://www.kaggle.com/mlg-ulb/creditcardfraud). This dataset consists of the results of a PCA transformation on the original credit card transaction data. PCA was carried out to protect sensitive information present in the original data. In addition to the principle coordinates the data contains the time since the first transaction that was logged and the amount (in $) of the transaction. It is a binary classification problem where the positive label (1) corresponds to an instance of fraud and the negative label (0) is a normal transaction. The dataset is highly imbalanced with very few instances of fraud present.

### Task
The objective of this project is to classify samples as either 'normal' or 'fraudulent' transactions. In order to do this I will use all available features present in the dataset. 

### Access
In order to access the Kaggle dataset, I downloaded the compressed dataset, extracted it, and registered it as a dataset in my Azure workspace. From this point I programmatically interacted with it through the SDK.

## Automated ML
For the automated ML run, I chose to limit the experiment to a total duration of 1 hr to reduce the potential for session timeout. The primary metric that I used as an objective for the autoML run was weighted AUC (area under the curve), which is the suggested metric for anomaly detection in the Azure documentation. AUC accounts for precision and recall of both classes and allows accurate characterisation of the performance of a model on an imbalanced dataset.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
I chose Random Forests as the model to tune with Hyperdrive. Random Forests are versatile and perform well across a wide variety of machine learning problems. I am familiar with Random Forests from previous work and have a good understanding of the hyperparameters that impact classification strongly. For the Hyperdrive run I selected the number of estimators (trees in the forest) and the maximum depth of each tree as the hyperparameters to tune. Together these two hyperparameters can drastically improve the performance of a Random Forest model. I chose random parameter sampling over intervals that I have seen good performance for in the past (10-1000 for number of estimators and 1-100 for maximum depth). For an early termination policy I chose the Bandit Policy with a slack factor of 0.2 and a delay interval of 5. 


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
