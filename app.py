"""
Breast Cancer Prediction Web App
================================
Production-ready Flask application that serves a trained
Artificial Neural Network (ANN) for breast cancer classification.

Author: Akinmusire Oluwankorinola
Model: Feed-Forward Neural Network (Keras)
Deployment Target: Render
"""

# ---------------------------------------------------------
# Imports
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Initialize Flask app
app = Flask(__name__)

# Load trained model using file-relative path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_cancer_predictor.h5"

# Ensure model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Load breast cancer feature names from sklearn
data = load_breast_cancer(as_frame=True)
features = list(data.frame.columns[:-1])  # All columns except target

# Fit scaler using training data (using full dataset for reference scaling)
scaler = StandardScaler()
scaler.fit(data.frame[features])

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prediction_class = None
    prediction_prob = None

    if request.method == "POST":
        try:
            # Get form values
            form_data = {}
            for feature in features:
                form_data[feature] = float(request.form.get(feature, 0))

            # Convert to DataFrame and scale
            input_df = pd.DataFrame([form_data])
            input_scaled = scaler.transform(input_df)

            # Make prediction
            prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
            prediction_class = "malignant" if prediction_prob < 0.5 else "benign"
            prediction = f"🔬 Prediction: {prediction_class.upper()} ({prediction_prob*100:.2f}%)"
            prediction_class = "benign" if prediction_prob >= 0.5 else "malignant"

        except Exception as e:
            prediction = f"Error: {str(e)}"
            prediction_class = "error"

    return render_template("index.html", prediction=prediction, prediction_class=prediction_class, features=features)


if __name__ == "__main__":
    print("Starting Flask server...")
    print("Open this link in your browser: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
