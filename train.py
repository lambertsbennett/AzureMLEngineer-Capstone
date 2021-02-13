from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset

ws = Workspace.from_config()
dataset = Dataset.get_by_name(ws, name='cc-fraud')

run = Run.get_context()

def process_data(data):
    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("Class")
    return x_df, y_df

# Drop NAs and encode data.
x, y = process_data(dataset)

#Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33)
    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_estimators', type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=None, help="Maximum depth of tree.")

    args = parser.parse_args()

    run.log("Number of Estimators:", np.int(args.num_estimators))
    run.log("Max iterations:", np.int(args.max_depth))

    model = RandomForestClassifier(num_estimators=args.num_estimators, max_depth=args.max_depth).fit(x_train, y_train)

    roc_auc = roc_auc_score(model.predict(x_test), y_test)
    run.log("ROC_AUC", np.float(roc_auc))

if __name__ == '__main__':
    main()

