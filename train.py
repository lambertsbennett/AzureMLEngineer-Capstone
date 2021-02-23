from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core import Dataset
from azureml.core.run import Run


run = Run.get_context()
ws = run.experiment.workspace

def process_data(data):
    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("Class")
    return x_df, y_df
    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=None, help="Maximum depth of tree.")
    parser.add_argument('--input_data', type=str)

    args = parser.parse_args()

    dataset = Dataset.get_by_id(ws, id=args.input_data)

    # Drop NAs and encode data.
    x, y = process_data(dataset)

    #Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33)

    run.log("Number of Estimators:", np.int(args.n_estimators))
    run.log("Max iterations:", np.int(args.max_depth))

    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth).fit(x_train, y_train)

    roc_auc = roc_auc_score(model.predict(x_test), y_test)
    run.log("auc", np.float(roc_auc))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()

