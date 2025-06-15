import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from torchvision import transforms

class ImagePreprocessor:
    """
    Handles image preprocessing steps required by the model.
    """
    def __init__(self):
        # ImageNet mean and std for normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = (224, 224)

        # Define the torchvision transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), # Converts to [0,1] and rearranges dims HWC -> CHW
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def preprocess(self, image_data: bytes) -> np.ndarray:
        """
        Loads an image from bytes, converts to RGB, resizes, normalizes,
        and adds a batch dimension.

        Args:
            image_data (bytes): Raw image data (e.g., from request.files['file'].read()).

        Returns:
            np.ndarray: Preprocessed image as a numpy array, ready for model input.
                        Shape: (1, C, H, W) where C=3, H=224, W=224.
        """
        # Load image using PIL
        img = Image.open(io.BytesIO(image_data)).convert('RGB') # Ensure RGB

        # Apply transformations
        input_tensor = self.transform(img)

        # Add batch dimension and convert to numpy
        input_batch = input_tensor.unsqueeze(0).numpy() # (1, C, H, W)

        return input_batch

class OnnxModel:
    """
    Loads an ONNX model and provides an inference method.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model()

    def _load_model(self):
        """Loads the ONNX model using ONNX Runtime."""
        try:
            # Prefer CUDAExecutionProvider if GPU is available, fallback to CPU
            self.session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"ONNX model '{self.model_path}' loaded successfully.")
            print(f"Input name: {self.input_name}, Output name: {self.output_name}")
        except Exception as e:
            print(f"Error loading ONNX model from {self.model_path}: {e}")
            self.session = None # Indicate that the model failed to load

    def predict(self, preprocessed_image_input: np.ndarray) -> np.ndarray:
        """
        Performs inference on the preprocessed image input.

        Args:
            preprocessed_image_input (np.ndarray): The preprocessed image data
                                                    ready for the ONNX model.
                                                    Shape: (1, 3, 224, 224).

        Returns:
            np.ndarray: The model's raw output (e.g., logits or probabilities).
                        Shape: (1, num_classes).
        """
        if self.session is None:
            raise RuntimeError("ONNX model is not loaded. Cannot perform prediction.")

        try:
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed_image_input})
            return outputs[0] # Returns the numpy array of predictions
        except Exception as e:
            raise RuntimeError(f"Error during ONNX model prediction: {e}")