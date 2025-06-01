import base64
import io
from PIL import Image
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("mtailor_model.onnx")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize to [0,1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Change data layout from HWC to CHW
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array.astype(np.float32)

def predict(base64_image: str) -> str:
    try:
        image_bytes = base64.b64decode(base64_image)
        input_tensor = preprocess_image(image_bytes)
        output = session.run(None, {"input": input_tensor})
        prediction = np.argmax(output[0], axis=1)[0]
        return str(prediction)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    with open("test.b64") as f:
        base64_img = f.read()
    print("Predicted class ID:", predict(base64_img))

