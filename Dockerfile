FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Flower and other dependencies
RUN pip install --no-cache-dir \
    flwr \
    flwr-datasets[vision] \
    flwr[simulation] \
    numpy \
    matplotlib \
    jupyter \
    jupyterlab \
    ipywidgets

# Copy project files
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Expose Jupyter port
EXPOSE 8888

# Default command to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
