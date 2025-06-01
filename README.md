
# Mtailor ML Ops Assessment

This project demonstrates deploying an image classification model using ONNX runtime in a Python environment. The model is containerized with Docker and deployed on the Cerebrium platform.

---

## Project Overview

- Model: Image classification ONNX model (`mtailor_model.onnx`)
- Inference script: `predictor.py` — preprocesses images, runs ONNX inference, and returns predicted class
- Containerized using a lightweight Python 3.10 Docker image with necessary dependencies
- Supports image input as base64 encoded strings
- Deployable via Cerebrium for easy cloud hosting and API access

---

## Setup & Usage

### Requirements

- Docker
- Python 3.10+ (for local testing)
- Cerebrium CLI (`pip install cerebrium --upgrade`) — for deployment

### Build Docker Image

```bash
docker build -t mtailor-model:slim .
````

### Run Docker Container Locally

```bash
docker run --rm mtailor-model:slim
```

*Make sure `test.b64` (a base64 encoded test image) is in the container context.*

### Predicting Locally

The container runs `predictor.py` which:

* Reads the base64 image from `test.b64`
* Preprocesses the image (resize, crop, normalize)
* Runs ONNX inference
* Prints the predicted class ID

---

## Deployment on Cerebrium

1. Install Cerebrium CLI:

```bash
pip install cerebrium --upgrade
```

2. Initialize Cerebrium project:

```bash
cerebrium init mtailor-mlops
cd mtailor-mlops
```

3. Copy the following files into the Cerebrium project folder:

* `predictor.py`
* `mtailor_model.onnx`
* `requirements.txt`
* Test image file (optional)

4. Deploy the app:

```bash
cerebrium deploy
```

5. Use the provided API endpoint to send base64-encoded images for inference.

---

## Code Structure

* `predictor.py`: Main inference script using ONNX runtime and PIL for image preprocessing
* `requirements.txt`: Python dependencies (`onnxruntime`, `Pillow`, `numpy`)
* `Dockerfile`: Containerizes the app with minimal dependencies and setup
* `test.b64`: Sample base64 encoded test image for local testing

---

## Notes

* The `predictor.py` script replaces torchvision transforms with PIL and numpy to minimize dependencies.
* Input image preprocessing matches model expectations (224x224 size, normalization).
* The ONNX runtime expects input tensors with `float32` data type.

---

## Contact

For any questions, please reach out to (saikrupaelate@gmail.com) or visit the GitHub repository.

```
