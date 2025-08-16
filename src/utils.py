import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def load_csv(path, sample_frac=1.0, random_state=42):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    return df

def feature_engineer_df(df):
    df = df.copy()
    if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
        df["orig_balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    else:
        df["orig_balance_diff"] = 0.0
    if "oldbalanceDest" in df.columns and "newbalanceDest" in df.columns:
        df["dest_balance_diff"] = df["oldbalanceDest"] - df["newbalanceDest"]
    else:
        df["dest_balance_diff"] = 0.0
    if "amount" in df.columns:
        df["abs_amount"] = df["amount"].abs()
    else:
        df["abs_amount"] = 0.0
    if {"nameOrig", "nameDest"}.issubset(df.columns):
        df["orig_dest_same"] = (df["nameOrig"] == df["nameDest"]).astype(int)
    return df

def get_numeric_categorical(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for drop in ["nameOrig", "nameDest"]:
        if drop in cat_cols:
            cat_cols.remove(drop)
    return numeric_cols, cat_cols

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
