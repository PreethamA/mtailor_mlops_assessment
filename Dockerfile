# Use a slim Python base image
FROM python:3.9-slim-buster

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
# This includes app.py, model.py, pytorch_model_definitions.py, etc.
COPY . .


# Convert the PyTorch model to ONNX during build time
# This means the .onnx file is part of the final image, not generated at runtime
# convert_to_onnx.py now correctly imports MyImageNetModel from pytorch_model_definitions.py
RUN python export_onnx.py

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application using Gunicorn
# Gunicorn is a production-ready WSGI server
# -w: number of worker processes (usually 2 * CPU_CORES + 1)
# -b: bind to all interfaces on port 5000
# app:app refers to the 'app' Flask instance within the 'app.py' module
CMD ["gunicorn", "--timeout", "120", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]