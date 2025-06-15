from flask import Flask, request, jsonify
from model import OnnxModel, ImagePreprocessor
import os
import json
import numpy as np
app = Flask(__name__)

# Define the path to your ONNX model
ONNX_MODEL_FILE = "mtailor_model.onnx"

# Initialize model and preprocessor globally for efficiency
# This will load the model once when the Flask app starts
model_instance = None
preprocessor_instance = None

try:
    if os.path.exists(ONNX_MODEL_FILE):
        model_instance = OnnxModel(ONNX_MODEL_FILE)
        preprocessor_instance = ImagePreprocessor()
        print("Application initialized successfully with ONNX model and preprocessor.")
    else:
        print(f"Error: ONNX model '{ONNX_MODEL_FILE}' not found. Prediction will fail.")
except Exception as e:
    print(f"Failed to initialize model or preprocessor: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for image classification.
    Expects a multipart/form-data request with an image file.
    """
    if model_instance is None or preprocessor_instance is None:
        return jsonify({"error": "Server not initialized. Model or preprocessor not loaded."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No image file provided. Please upload an image with key 'file'."}), 400

    image_file = request.files['file']
    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Read image data
        image_bytes = image_file.read()

        # Preprocess the image
        preprocessed_input = preprocessor_instance.preprocess(image_bytes)

        # Get prediction from ONNX model
        raw_predictions = model_instance.predict(preprocessed_input)

        # Convert raw predictions (logits) to probabilities if necessary
        # For ImageNet, often the model output is already logits, so apply softmax
        # If your model directly outputs probabilities, skip softmax.
        probabilities = np.exp(raw_predictions) / np.sum(np.exp(raw_predictions), axis=1, keepdims=True)
        probabilities = probabilities.flatten().tolist() # Convert to a flat list

        # Get the predicted class ID (index with highest probability)
        predicted_class_id = int(np.argmax(raw_predictions))

        response = {
            "predicted_class_id": predicted_class_id,
            "probabilities": probabilities,
            "message": "Prediction successful"
        }
        return jsonify(response), 200

    except RuntimeError as re:
        print(f"Prediction runtime error: {re}")
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

# Inside app.py, add this new route:
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the ImageNet ONNX Model API. Use /predict for inference and /health for status."}), 200

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify server and model status.
    """
    status = "healthy" if model_instance is not None and model_instance.session is not None else "unhealthy"
    message = "Model and server are running." if status == "healthy" else "Model or server not initialized properly."
    return jsonify({"status": status, "message": message}), 200 if status == "healthy" else 503

if __name__ == '__main__':
    # When deploying with Gunicorn (as per Dockerfile), this block won't run.
    # It's for local development/testing directly.
    # debug=True should NOT be used in production.
    app.run(host='0.0.0.0', port=5000, debug=True)