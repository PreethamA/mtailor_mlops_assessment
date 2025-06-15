# convert_to_onnx.py
import torch
import torch.nn as nn
import onnx
import numpy as np
import onnxruntime as ort
import os
import requests
from pytorch_model_definitions import MyImageNetModel # Import your PyTorch model definition

# Define constants sddsdasd
MODEL_WEIGHTS_URL = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0"
MODEL_WEIGHTS_PATH = "pytorch_model_weights.pth"
ONNX_MODEL_PATH = "mtailor_model.onnx"
INPUT_IMAGE_SIZE = (224, 224)
# MEAN and STD are not directly used in this script's conversion part,
# but are important for `model.py`'s preprocessor.
# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]

def download_file(url, dest_path):
    """Downloads a file from a URL to a destination path."""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}. Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        raise # Re-raise to stop execution if download fails

def convert_pytorch_to_onnx(model: nn.Module, dummy_input: torch.Tensor, onnx_path: str):
    """
    Converts a PyTorch model to ONNX format.
    """
    print(f"Converting PyTorch model to ONNX at: {onnx_path}")
    try:
        torch.onnx.export(
            model,                   # PyTorch Model
            dummy_input,             # Dummy input for tracing
            onnx_path,               # Output ONNX file name
            export_params=True,      # Export model parameters
            opset_version=11,        # ONNX opset version
            do_constant_folding=True,# Fold constants for optimization
            input_names=['input'],   # Input name for the ONNX graph
            output_names=['output'], # Output name for the ONNX graph
            dynamic_axes={'input': {0: 'batch_size'}} # Allow variable batch size
        )
        print("ONNX model conversion successful!")

        # Verify the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful!")

    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        raise

def test_onnx_model(onnx_path: str, dummy_input_numpy: np.ndarray):
    """
    Tests the converted ONNX model using ONNX Runtime.
    """
    print("\nTesting ONNX model with ONNX Runtime...")
    try:
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        ort_inputs = {input_name: dummy_input_numpy}
        ort_outputs = ort_session.run([output_name], ort_inputs)

        print("ONNX Runtime test successful!")
        print("ONNX Runtime Output shape:", ort_outputs[0].shape)
        return ort_outputs[0]
    except Exception as e:
        print(f"Error testing ONNX model with ONNX Runtime: {e}")
        raise

if __name__ == "__main__":
    print("--- Starting ONNX Conversion Process ---")
    # 1. Download model weights
    try:
        download_file(MODEL_WEIGHTS_URL, MODEL_WEIGHTS_PATH)
    except Exception as e:
        print(f"Failed to download model weights. Please check the URL or your network connection. Error: {e}")
        exit(1) # Exit if weights can't be downloaded

    # 2. Instantiate PyTorch model and load weights
    model = MyImageNetModel(num_classes=1000)
    try:
        # Load weights, ensuring to map to CPU if training was on GPU
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
        model.eval() # Set to evaluation mode
        print("PyTorch model loaded successfully and set to eval mode.")
    except Exception as e:
        print(f"Failed to load PyTorch model weights from {MODEL_WEIGHTS_PATH}: {e}")
        print("Proceeding with randomly initialized model. ONNX inference will NOT be accurate.")
        # If loading fails, ensure model is still in eval mode for export
        model.eval()

    # 3. Create a dummy input (batch_size, channels, height, width)
    # The image is 224x224, 3 channels (RGB)
    dummy_input = torch.randn(1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], requires_grad=True)
    dummy_input_numpy = dummy_input.detach().numpy()

    # 4. Convert and verify
    convert_pytorch_to_onnx(model, dummy_input, ONNX_MODEL_PATH)

    # 5. Test the ONNX model (optional, for local verification)
    # The dummy input used here is random, so output values won't be meaningful
    # but shapes and execution path will be verified.
    onnx_output = test_onnx_model(ONNX_MODEL_PATH, dummy_input_numpy)

    print(f"\nConversion and initial ONNX Runtime test complete. ONNX model saved to {ONNX_MODEL_PATH}")