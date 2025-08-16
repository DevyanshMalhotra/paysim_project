import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
ARTIFACTS_DIR = "artifacts"
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "pipeline.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")
st.set_page_config(page_title="PaySim Fraud Detector", layout="wide")
if not os.path.exists(PIPELINE_PATH):
    st.error("Pipeline not found. Run training first.")
    st.stop()
pipeline = joblib.load(PIPELINE_PATH)
expected_raw_cols = joblib.load(FEATURES_PATH)
st.title("PaySim Fraud Detector")
uploaded = st.file_uploader("Upload CSV file with raw PaySim columns", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Unable to read file: {e}")
        st.stop()
    missing = [c for c in expected_raw_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required raw columns: {missing}")
    else:
        probs = pipeline.predict_proba(df)[:, 1]
        out = df.copy()
        out["fraud_prob"] = probs
        st.dataframe(out)
        if st.checkbox("Show SHAP explanation (first row)"):
            try:
                explainer = shap.TreeExplainer(pipeline.named_steps["model"])
                X_t = pipeline.named_steps["preprocessor"].transform(df)
                shap_values = explainer.shap_values(X_t)
                feature_names = None
                pre = pipeline.named_steps["preprocessor"].named_steps["pre"]
                ohe_names = []
                if "cat" in pre.named_transformers_:
                    ohe = pre.named_transformers_["cat"]
                    if hasattr(ohe, "get_feature_names_out"):
                        ohe_names = list(ohe.get_feature_names_out())
                numeric_cols = pipeline.named_steps["preprocessor"].named_steps["pre"].transformers_[0][2]
                feature_names = list(numeric_cols) + ohe_names
                shap.force_plot(explainer.expected_value, shap_values[0], features=X_t[0], feature_names=feature_names, matplotlib=True, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(ARTIFACTS_DIR, "shap_row0.png"))
                plt.close()
                st.image(os.path.join(ARTIFACTS_DIR, "shap_row0.png"))
            except Exception:
                st.write("SHAP explanation unavailable")
else:
    st.write("Run sample prediction from local data")
    if st.button("Load sample data/paysim.csv (10 rows)"):
        try:
            sample = pd.read_csv("data/paysim.csv").head(10)
            missing = [c for c in expected_raw_cols if c not in sample.columns]
            if missing:
                st.error(f"Local sample missing: {missing}")
            else:
                probs = pipeline.predict_proba(sample)[:, 1]
                sample["fraud_prob"] = probs
                st.dataframe(sample)
        except Exception as e:
            st.error(f"Error loading sample: {e}")
