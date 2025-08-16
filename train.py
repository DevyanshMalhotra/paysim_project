import os
import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
from src.utils import load_csv, feature_engineer_df, get_numeric_categorical, ensure_dir

DATA_PATH = "data/paysim.csv"
ARTIFACTS_DIR = "artifacts"
N_TRIALS = 24
TEST_SIZE = 0.2
RANDOM_STATE = 42

ensure_dir(ARTIFACTS_DIR)

df_raw = load_csv(DATA_PATH)
df_raw = df_raw.dropna(subset=["isFraud"])
y = df_raw["isFraud"].astype(int)
X_raw = df_raw.drop(columns=["isFraud"])
X_fe = feature_engineer_df(X_raw)
numeric_cols, cat_cols = get_numeric_categorical(X_fe)
neg = int((y == 0).sum())
pos = int((y == 1).sum())
scale_pos_weight = neg / max(1, pos)

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def train_xgb_on_transformed(X_train_raw, X_val_raw, y_train, y_val, params):
    fe = FunctionTransformer(lambda df: feature_engineer_df(df), validate=False)
    ohe = make_ohe()
    pre = ColumnTransformer(
        [("num", StandardScaler(), numeric_cols),
         ("cat", ohe, cat_cols)],
        remainder="drop"
    )
    X_train_t = pre.fit_transform(fe.transform(X_train_raw))
    X_val_t = pre.transform(fe.transform(X_val_raw))
    clf = XGBClassifier(**params, eval_metric="logloss")
    clf.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
    return clf, pre, fe

def objective(trial):
    X_tr, X_val, y_tr, y_val = train_test_split(X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "n_estimators": 500,
        "use_label_encoder": False,
        "objective": "binary:logistic",
        "random_state": RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "predictor": "auto"
    }
    clf, pre, fe = train_xgb_on_transformed(X_tr, X_val, y_tr, y_val, params)
    X_val_t = pre.transform(fe.transform(X_val))
    preds = clf.predict(X_val_t)
    return f1_score(y_val, preds)

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    best_params.update({
        "n_estimators": 1000,
        "use_label_encoder": False,
        "objective": "binary:logistic",
        "random_state": RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "predictor": "auto"
    })
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    clf, pre, fe = train_xgb_on_transformed(X_train_full, X_test, y_train_full, y_test, best_params)
    final_preprocessor = Pipeline([("fe", fe), ("pre", pre)])
    pipeline_full = Pipeline([("preprocessor", final_preprocessor), ("model", clf)])
    pipeline_path = os.path.join(ARTIFACTS_DIR, "pipeline.joblib")
    joblib.dump(pipeline_full, pipeline_path)
    expected_raw_cols = X_raw.columns.tolist()
    joblib.dump(expected_raw_cols, os.path.join(ARTIFACTS_DIR, "feature_columns.joblib"))
    metadata = {
        "best_params": best_params,
        "train_pos": int(y_train_full.sum()),
        "train_neg": int((y_train_full == 0).sum()),
        "test_pos": int(y_test.sum()),
        "test_neg": int((y_test == 0).sum())
    }
    joblib.dump(metadata, os.path.join(ARTIFACTS_DIR, "metadata.joblib"))
    X_test_t = pipeline_full.named_steps["preprocessor"].transform(X_test)
    preds = pipeline_full.named_steps["model"].predict(X_test_t)
    probs = pipeline_full.named_steps["model"].predict_proba(X_test_t)[:, 1]
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(ARTIFACTS_DIR, "classification_report.csv"))
    pd.DataFrame(confusion_matrix(y_test, preds), index=["neg","pos"], columns=["neg","pos"]).to_csv(os.path.join(ARTIFACTS_DIR, "confusion_matrix.csv"))
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    pd.DataFrame({"precision": pr_precision, "recall": pr_recall}).to_csv(os.path.join(ARTIFACTS_DIR, "precision_recall_curve.csv"), index=False)
    fi = pipeline_full.named_steps["model"].feature_importances_
    pre = pipeline_full.named_steps["preprocessor"].named_steps["pre"]
    ohe_names = []
    if "cat" in pre.named_transformers_:
        ohe = pre.named_transformers_["cat"]
        if hasattr(ohe, "get_feature_names_out"):
            try:
                ohe_names = list(ohe.get_feature_names_out(input_features=cat_cols))
            except Exception:
                ohe_names = list(ohe.get_feature_names_out())
    feature_names = numeric_cols + ohe_names
    if len(feature_names) != len(fi):
        feature_names = [f"f{i}" for i in range(len(fi))]
    fi_series = pd.Series(fi, index=feature_names).sort_values(ascending=False)
    fi_series.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"))
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        fi_series.head(30).plot(kind="bar")
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance.png"))
        plt.close()
    except Exception:
        pass
    try:
        import shap
        explainer = shap.TreeExplainer(pipeline_full.named_steps["model"])
        shap_values = explainer.shap_values(X_test_t)
        try:
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, features=X_test_t, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(ARTIFACTS_DIR, "shap_summary.png"))
            plt.close()
        except Exception:
            pass
    except Exception:
        pass
    summary_lines = [
        f"F1:{f1:.6f}",
        f"ROC_AUC:{auc:.6f}",
        f"AP:{ap:.6f}",
        f"BEST_PARAMS:{best_params}",
        f"TRAIN_POS:{metadata['train_pos']}",
        f"TRAIN_NEG:{metadata['train_neg']}"
    ]
    with open(os.path.join(ARTIFACTS_DIR, "summary.md"), "w") as fh:
        fh.write("\n".join(summary_lines))
    try:
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
        nb = new_notebook()
        nb.cells = []
        nb.cells.append(new_markdown_cell("# PaySim Fraud Detection Notebook"))
        nb.cells.append(new_code_cell("import pandas as pd\nfrom src.utils import feature_engineer_df\nprint('load dataset')"))
        nb.cells.append(new_code_cell("df = pd.read_csv('data/paysim.csv')\ndf.head()"))
        nb.cells.append(new_code_cell("df.info()"))
        nb.cells.append(new_code_cell("import matplotlib.pyplot as plt\nimport seaborn as sns\nsns.set()\nplt.figure()\nplt.hist(df['amount'].dropna(), bins=100)\nplt.yscale('log')\nplt.title('amount distribution (log y)')\nplt.savefig('artifacts/amount_dist.png')\nplt.close()"))
        nb.cells.append(new_code_cell("sns.countplot(data=df, x='type')\nimport matplotlib.pyplot as plt\nplt.xticks(rotation=45)\nplt.tight_layout()\nplt.savefig('artifacts/type_count.png')\nplt.close()"))
        nb.cells.append(new_code_cell("fraud_by_type = df.groupby('type')['isFraud'].mean().sort_values(ascending=False)\nimport matplotlib.pyplot as plt\nfraud_by_type.plot(kind='bar')\nplt.tight_layout()\nplt.savefig('artifacts/fraud_rate_by_type.png')\nplt.close()"))
        nb.cells.append(new_code_cell("from sklearn.metrics import classification_report\nimport joblib\npipe = joblib.load('artifacts/pipeline.joblib')\nX = df.drop(columns=['isFraud'])\nprobs = pipe.predict_proba(X)[:,1]\nprint('sample probs', probs[:5])"))
        nb.cells.append(new_code_cell("print('notebook generated')"))
        nb_path = os.path.join("notebooks", "paysim_fraud_detection.ipynb")
        os.makedirs("notebooks", exist_ok=True)
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    except Exception:
        pass
    print("PIPELINE_SAVED", pipeline_path)
    print("F1", f1)
    print("AUC", auc)

if __name__ == "__main__":
    main()
