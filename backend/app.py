from flask import Flask, request, jsonify
import joblib
import pandas as pd
import io

# Initialize Flask app
superkart_api = Flask(__name__)

# Load the pre-trained champion pipeline (preprocessor + model)
model_path = 'superkart_model.joblib'
try:
    model = joblib.load(model_path)
except Exception as e:
    model = joblib.load('backend/superkart_model.joblib')

@superkart_api.route("/")
def home():
    return jsonify({"message": "SuperKart Flask API is running!"})

@superkart_api.route('/v1/predict', methods=['POST'])
def predict():
    try:
        # Parse request payload
        data = request.get_json()

        # Convert payload to DataFrame matching expected preprocessor fields
        df = pd.DataFrame([data])

        # Perform inference
        prediction = model.predict(df)[0]

        return jsonify({'Product_Store_Sales_Total_Predicted': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@superkart_api.route('/v1/predictbatch', methods=['POST'])
def predict_batch():
    try:
        # Ensure file was sent
        if 'file' not in request.files:
            return jsonify({'error': 'No file key found in request'}), 400

        file = request.files['file']

        # Read CSV into a DataFrame
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))

        # Perform batch inference
        predictions = model.predict(df)

        # Map row indices to predicted outputs
        results = {str(i): float(pred) for i, pred in enumerate(predictions)}
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    superkart_api.run(host='0.0.0.0', port=7860)
