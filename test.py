import numpy as np
import os
import onnxruntime as ort
from PIL import Image
import io
from model import ImagePreprocessor, OnnxModel
#from convert_to_onx import ONNX_MODEL_PATH # Import path for local testing
from export_onnx import ONNX_MODEL_PATH
# Ensure data directory and images exist for testing
SAMPLE_IMAGE_TENCH_PATH = "/app//data/n01440764_tench.jpeg"
SAMPLE_IMAGE_TURTLE_PATH = "/app//data/n01667114_mud_turtle.jpeg"

# Pre-defined expected class IDs (from ImageNet)
EXPECTED_TENCH_ID = 0
EXPECTED_TURTLE_ID = 35

def test_image_preprocessor():
    """Tests the ImagePreprocessor class."""
    print("\n--- Testing ImagePreprocessor ---")
    preprocessor = ImagePreprocessor()

    if not os.path.exists(SAMPLE_IMAGE_TENCH_PATH):
        print(f"Skipping ImagePreprocessor test: Sample image not found at {SAMPLE_IMAGE_TENCH_PATH}")
        return False

    try:
        with open(SAMPLE_IMAGE_TENCH_PATH, 'rb') as f:
            image_bytes = f.read()

        preprocessed_image = preprocessor.preprocess(image_bytes)

        # Check shape: (batch, channels, height, width)
        expected_shape = (1, 3, 224, 224)
        if preprocessed_image.shape == expected_shape:
            print(f"ImagePreprocessor shape test PASSED: {preprocessed_image.shape}")
        else:
            print(f"ImagePreprocessor shape test FAILED: Expected {expected_shape}, got {preprocessed_image.shape}")
            return False

        # Basic check for normalization values (should be roughly between -2.5 and 2.5 after normalization)
        if np.min(preprocessed_image) >= -5.0 and np.max(preprocessed_image) <= 5.0: # Wide range for safety
            print("ImagePreprocessor normalization range test PASSED.")
        else:
            print("ImagePreprocessor normalization range test FAILED: Values outside expected range.")
            return False

        print("ImagePreprocessor test PASSED.")
        return True
    except Exception as e:
        print(f"ImagePreprocessor test FAILED: {e}")
        return False

def test_onnx_model_loading():
    """Tests loading of the ONNX model."""
    print("\n--- Testing ONNX Model Loading ---")
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Skipping ONNX Model Loading test: ONNX model not found at {ONNX_MODEL_PATH}")
        print("Please run convert_to_onnx.py first.")
        return False

    try:
        model = OnnxModel(ONNX_MODEL_PATH)
        if model.session is not None:
            print("ONNX Model Loading test PASSED: Model loaded successfully.")
            return True
        else:
            print("ONNX Model Loading test FAILED: Model session is None.")
            return False
    except Exception as e:
        print(f"ONNX Model Loading test FAILED: {e}")
        return False

def test_full_inference_pipeline():
    """
    Tests the full inference pipeline from image file to prediction,
    using locally generated ONNX model and preprocessor.
    """
    print("\n--- Testing Full Inference Pipeline ---")
    print(f"Current working directory in test_full_inference_pipeline: {os.getcwd()}")

    # --- ADD THESE DEBUG LINES ---
    abs_tench_path = os.path.abspath(SAMPLE_IMAGE_TENCH_PATH)
    abs_turtle_path = os.path.abspath(SAMPLE_IMAGE_TURTLE_PATH)
    print(f"DEBUG: Absolute path for tench: {abs_tench_path}")
    print(f"DEBUG: Does {abs_tench_path} exist? {os.path.exists(abs_tench_path)}")
    if os.path.exists(abs_tench_path):
        try:
            tench_stat = os.stat(abs_tench_path)
            print(f"DEBUG: Tench file permissions (octal): {oct(tench_stat.st_mode)}")
            print(f"DEBUG: Tench file size: {tench_stat.st_size} bytes")
        except OSError as e:
            print(f"DEBUG: Could not get stats for tench: {e}")
    else:
        # If it doesn't exist, let's try to list the parent directory from within python
        parent_dir = os.path.dirname(abs_tench_path)
        print(f"DEBUG: Listing contents of {parent_dir}:")
        try:
            for item in os.listdir(parent_dir):
                print(f"  - {item}")
        except OSError as e:
            print(f"DEBUG: Could not list directory {parent_dir}: {e}")

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Skipping Full Inference Pipeline test: ONNX model not found at {ONNX_MODEL_PATH}")
        print("Please run convert_to_onnx.py first.")
        return False
    if not os.path.exists(SAMPLE_IMAGE_TENCH_PATH) or not os.path.exists(SAMPLE_IMAGE_TURTLE_PATH):
        print(f"Skipping Full Inference Pipeline test: Sample images not found in {SAMPLE_IMAGE_TENCH_PATH} directory.")
        return False

    preprocessor = ImagePreprocessor()
    model = OnnxModel(ONNX_MODEL_PATH)

    if model.session is None:
        print("Skipping Full Inference Pipeline test: ONNX model failed to load.")
        return False

    all_passed = True

    # Test with tench image
    try:
        print(f"Testing with {SAMPLE_IMAGE_TENCH_PATH}...")
        with open(SAMPLE_IMAGE_TENCH_PATH, 'rb') as f:
            image_bytes = f.read()
        preprocessed_input = preprocessor.preprocess(image_bytes)
        raw_predictions = model.predict(preprocessed_input)
        predicted_class_id = np.argmax(raw_predictions).item()

        print(f"  Predicted class ID for tench: {predicted_class_id}, Expected: {EXPECTED_TENCH_ID}")
        if predicted_class_id == EXPECTED_TENCH_ID:
            print("  Tench prediction PASSED.")
        else:
            print("  Tench prediction FAILED (might be due to random weights if conversion failed).")
            # Note: This test relies on loaded weights from convert_to_onnx.py.
            # If the weights download/load failed there, this will be inaccurate.
            all_passed = False
    except Exception as e:
        print(f"  Tench prediction FAILED: {e}")
        all_passed = False

    # Test with mud turtle image
    try:
        print(f"Testing with {SAMPLE_IMAGE_TURTLE_PATH}...")
        with open(SAMPLE_IMAGE_TURTLE_PATH, 'rb') as f:
            image_bytes = f.read()
        preprocessed_input = preprocessor.preprocess(image_bytes)
        raw_predictions = model.predict(preprocessed_input)
        predicted_class_id = np.argmax(raw_predictions).item()

        print(f"  Predicted class ID for mud turtle: {predicted_class_id}, Expected: {EXPECTED_TURTLE_ID}")
        if predicted_class_id == EXPECTED_TURTLE_ID:
            print("  Mud turtle prediction PASSED.")
        else:
            print("  Mud turtle prediction FAILED (might be due to random weights if conversion failed).")
            all_passed = False
    except Exception as e:
        print(f"  Mud turtle prediction FAILED: {e}")
        all_passed = False

    if all_passed:
        print("Full Inference Pipeline test PASSED.")
    else:
        print("Full Inference Pipeline test FAILED (at least one sub-test failed).")
    return all_passed

if __name__ == "__main__":
    print("--- Starting Local Model Tests ---")
    overall_status = True

    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please place sample images inside it.")

    # You might want to run convert_to_onnx.py first if it hasn't been run
    if not os.path.exists(ONNX_MODEL_PATH):
        print("\nWARNING: model.onnx not found. Attempting to run convert_to_onnx.py...")
        try:
            import subprocess
            subprocess.run(["python", "convert_to_onnx.py"], check=True)
            print("convert_to_onnx.py executed successfully.")
        except Exception as e:
            print(f"Failed to run convert_to_onnx.py: {e}")
            print("Please ensure you have downloaded pytorch_model_weights.pth and requirements are met.")
            overall_status = False # Cannot proceed meaningfully without ONNX model

    if overall_status:
        overall_status &= test_image_preprocessor()
        overall_status &= test_onnx_model_loading()
        overall_status &= test_full_inference_pipeline()

    print("\n--- Local Model Tests Complete ---")
    if overall_status:
        print("ALL LOCAL TESTS PASSED!")
    else:
        print("SOME LOCAL TESTS FAILED. Please check the logs above.")