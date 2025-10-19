# Frugal AI - Federated Learning Tutorial

This repository contains a comprehensive step-by-step tutorial on Federated Learning using the Flower framework and PyTorch. It serves as an educational resource for the Frugal AI class at MIAI Grenoble. The tutorial is built upon the Flower Examples.

## 📚 What is Federated Learning?

Federated Learning (FL) is a machine learning technique that trains models across multiple decentralized devices or servers holding local data samples, without exchanging the raw data itself. This approach ensures:

- **Privacy**: Data never leaves the local device
- **Decentralization**: Training happens on multiple clients
- **Efficiency**: Reduced data transfer and storage costs
- **Compliance**: Helps meet GDPR and data protection requirements

## 🎯 Tutorial Overview

This tutorial consists of 5 interactive Jupyter notebooks that guide you through:

1. **Introduction to Federated Learning** - Core concepts and motivation
2. **Understanding the Dataset and Model** - FashionMNIST dataset and CNN architecture
3. **Building a Flower Client** - Implementing local training logic
4. **Building a Flower Server** - Implementing aggregation and coordination
5. **Running the Full Experiment** - Complete federated learning pipeline

## 🚀 Quick Start

### Option 1: Using Docker (Recommended for Students)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd frugal-ai-example
   ```

2. **Build and start the Docker container**:
   ```bash
   docker-compose up
   ```

3. **Open Jupyter Lab in your browser**:
   ```
   http://localhost:8888
   ```

4. **Navigate to the `notebooks/` folder and start with notebook 01**

### Option 2: Without Docker

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -e .
pip install jupyter jupyterlab

# 3. Start Jupyter
jupyter lab

# 4. Open notebooks/01_introduction_to_federated_learning.ipynb
```

## 📖 Tutorial Structure

### Notebooks

- [`01_introduction_to_federated_learning.ipynb`](notebooks/01_introduction_to_federated_learning.ipynb) - Learn the fundamentals of FL
- [`02_dataset_and_model.ipynb`](notebooks/02_dataset_and_model.ipynb) - Explore FashionMNIST and build a CNN
- [`03_flower_client.ipynb`](notebooks/03_flower_client.ipynb) - Implement the client-side logic
- [`04_flower_server.ipynb`](notebooks/04_flower_server.ipynb) - Implement the server-side logic
- [`05_running_federated_learning.ipynb`](notebooks/05_running_federated_learning.ipynb) - Run and analyze experiments

### Source Code

The `src/fltutorial/` directory contains the production-ready implementation:

- [`task.py`](src/fltutorial/task.py) - Dataset loading, model definition, training/testing functions
- [`client.py`](src/fltutorial/client.py) - Flower client implementation
- [`server.py`](src/fltutorial/server.py) - Flower server implementation

## 🔬 Running Federated Learning Experiments

After going through the notebooks, you can run a full federated learning experiment:

### Using Flower CLI (Simulation Mode)

```bash
# Run federated learning with 5 clients for 10 rounds
flwr run
```

The configuration is defined in [`pyproject.toml`](pyproject.toml):
- 5 simulated clients (supernodes)
- 10 federated rounds
- Learning rate: 0.01
- 3 local epochs per round
- Batch size: 32

### Customizing the Experiment

Edit the `[tool.flwr.app.config]` section in `pyproject.toml` to modify:
- Number of rounds
- Learning rate
- Local epochs
- Batch size
- Evaluation strategy

## 📊 Dataset

We use **FashionMNIST** for this tutorial:
- 60,000 training images + 10,000 test images
- 28×28 grayscale images
- 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

The dataset is automatically partitioned across clients using IID (Independent and Identically Distributed) partitioning.

## 🏗️ Architecture

### Model: Simple CNN
```
Input (28×28×1)
    ↓
Conv2D (6 filters) → ReLU → MaxPool
    ↓
Conv2D (16 filters) → ReLU → MaxPool
    ↓
Fully Connected (120) → ReLU
    ↓
Fully Connected (84) → ReLU
    ↓
Output (10 classes)
```

### Federated Learning Setup
- **Server**: Coordinates training, aggregates model updates (FedAvg)
- **Clients**: Train on local data, send model updates to server
- **Rounds**: Multiple iterations of training and aggregation
- **Communication**: Only model weights are exchanged (not data)

## 🛠️ Requirements

- Python 3.10+
- PyTorch
- Flower (flwr)
- Jupyter Lab
- NumPy, Matplotlib
- Docker (optional, but recommended)


## 🔗 Additional Resources

### Documentation
- [Flower Documentation](https://flower.ai/docs/)
- [Flower Examples](https://github.com/adap/flower/tree/main/examples)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)


## 🤝 Contributing

This is an educational project for the Frugal AI class. Students are encouraged to:
- Report issues or suggest improvements
- Share interesting experiments
- Extend the tutorial with advanced topics

## 📧 Contact

For questions about this tutorial, please contact the course instructors/

## 📄 License

This project is for educational purposes as part of the MIAI Grenoble Frugal AI course.

---


