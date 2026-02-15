"""
Flask app that loads the trained Engie model and scaler from /model,
and exposes a simple page, a health check, and a prediction endpoint.
"""
import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")

app = Flask(__name__)
model = None
scaler = None
feature_columns = None


def load_artifacts():
    """Load the saved model, scaler and feature list from MODEL_DIR if they exist."""
    global model, scaler, feature_columns
    model_path = os.path.join(MODEL_DIR, "dnn_model.keras")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_columns.json")
    if not os.path.isfile(model_path) or not os.path.isfile(scaler_path) or not os.path.isfile(features_path):
        return False
    import tensorflow as tf
    import joblib
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path) as f:
        feature_columns = json.load(f)
    return True


@app.route("/health")
def health():
    """Check that the app is up and the model is loaded."""
    if model is None and not load_artifacts():
        return jsonify({"status": "error", "message": "Model or scaler not found"}), 503
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/", methods=["GET"])
def index():
    """Home page with a short explanation and a box to try a prediction."""
    if feature_columns is None:
        load_artifacts()
    n = len(feature_columns) if feature_columns else 79
    return render_template("index.html", feature_count=n)


@app.route("/predict", methods=["POST"])
def predict():
    """Expects JSON with key 'features' (list of 79 numbers). Returns predicted power in kW."""
    if model is None and not load_artifacts():
        return jsonify({"error": "Model not loaded"}), 503
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Send a JSON body"}), 400
    features = data.get("features")
    if features is None:
        return jsonify({"error": "Missing 'features' (list of %d numbers)" % len(feature_columns)}), 400
    try:
        X = np.array(features, dtype=np.float64).reshape(1, -1)
    except (TypeError, ValueError):
        return jsonify({"error": "features must be a list of numbers"}), 400
    if X.shape[1] != len(feature_columns):
        return jsonify({
            "error": "Expected %d features, got %d" % (len(feature_columns), X.shape[1])
        }), 400
    X_scaled = scaler.transform(X)
    pred = float(model.predict(X_scaled, verbose=0).flatten()[0])
    return jsonify({"prediction": pred, "unit": "kW"})


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
