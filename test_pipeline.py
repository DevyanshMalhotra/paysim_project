import joblib
from src.utils import load_csv
pipeline = joblib.load("artifacts/pipeline.joblib")
df = load_csv("data/paysim.csv", sample_frac=0.005)
df = df.head(20)
probs = pipeline.predict_proba(df)[:, 1]
print("predicted_probs", probs)
