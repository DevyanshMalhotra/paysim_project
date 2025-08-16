from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
PIPELINE_PATH = os.path.join("artifacts", "pipeline.joblib")
pipeline = joblib.load(PIPELINE_PATH)
app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        df = pd.DataFrame([payload])
    probs = pipeline.predict_proba(df)[:, 1]
    preds = pipeline.predict(df)
    out = df.copy()
    out["fraud_prob"] = probs
    out["isFraud_pred"] = preds
    return jsonify(out.to_dict(orient="records"))
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
