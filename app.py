from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os


# Load the exported scaler and scaled model
scaler_filename = 'scaler.joblib'
model_filename = 'logistic_regression_model_scaled.joblib'

scaler = joblib.load(scaler_filename)
model = joblib.load(model_filename)

# Initialize Flask app
app = Flask(__name__)

# Define a simple root route
@app.route('/')
def home():
    return 'API is running!'

# Define the /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received prediction request.")
        data = request.get_json(force=True)

        if data is None:
            return jsonify({"error": "No input data provided."}), 400

        # Feature names in correct order
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        # Convert input to DataFrame
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format. Expected a JSON object or list of objects."}), 400

        # Ensure all required features exist
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            return jsonify({"error": f"Missing features: {list(missing_cols)}"}), 400

        # Reorder columns to match training data
        input_df = input_df[feature_names]

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make predictions
        predictions = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)

        # Prepare response
        results = []
        for i in range(len(predictions)):
            result = {
                "prediction": int(predictions[i]),
                "probability_class_0": float(probabilities[i][0]),
                "probability_class_1": float(probabilities[i][1])
            }
            results.append(result)

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
