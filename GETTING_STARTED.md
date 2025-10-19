# Getting Started with the Federated Learning Tutorial

Welcome! This guide will help you get up and running with the tutorial quickly.

## 🎯 What You'll Build

By the end of this tutorial, you will:
- Understand federated learning fundamentals
- Implement a complete FL system using Flower and PyTorch
- Train a CNN on FashionMNIST across multiple simulated clients
- Analyze the privacy-utility trade-offs

## 📦 Prerequisites

- Basic Python knowledge
- Understanding of neural networks (helpful but not required)
- Docker installed (recommended) OR Python 3.10+

## 🚀 Quick Start (5 minutes)

### Using Docker (Easiest)

```bash
# 1. Start the container
docker-compose up

# 2. Open your browser
# Navigate to: http://localhost:8888

# 3. Open notebooks/01_introduction_to_federated_learning.ipynb
# Start learning!
```

### Without Docker

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

## 📚 Tutorial Path

Follow the notebooks in order:

### 1️⃣ Introduction (15 minutes)
**File:** `01_introduction_to_federated_learning.ipynb`

Learn:
- What is federated learning?
- Why is it important?
- How does FedAvg work?

### 2️⃣ Dataset & Model (30 minutes)
**File:** `02_dataset_and_model.ipynb`

Build:
- Load and explore FashionMNIST
- Partition data across clients
- Create a CNN model
- Train and test functions

### 3️⃣ Flower Client (25 minutes)
**File:** `03_flower_client.ipynb`

Implement:
- Client-side training logic
- Message handling
- Weight updates
- Local evaluation

### 4️⃣ Flower Server (25 minutes)
**File:** `04_flower_server.ipynb`

Implement:
- Server coordination logic
- FedAvg aggregation
- Global evaluation
- Model saving

### 5️⃣ Full Experiment (30 minutes)
**File:** `05_running_federated_learning.ipynb`

Execute:
- Complete FL training
- Result analysis
- Comparison with centralized learning
- Communication cost analysis

**Total Time:** ~2 hours

## 🔬 Running the Experiment

After completing the notebooks, run a real FL experiment:

```bash
# Run federated learning with 5 clients for 10 rounds
flwr run

# The model will be saved as final_model.pt
```

### Configuration

Edit `pyproject.toml` to customize:

```toml
[tool.flwr.app.config]
num-server-rounds = 10      # Number of FL rounds
learning-rate = 0.01        # Learning rate
local-epochs = 3            # Epochs per round per client
batch-size = 32             # Training batch size
fraction-evaluate = 1.0     # Fraction of clients for evaluation

[tool.flwr.federations.local-simulation]
options.num-supernodes = 5  # Number of simulated clients
```

## 📊 Expected Results

After 10 rounds with 5 clients:
- **Test Accuracy:** ~80-83%
- **Training Time:** ~10-15 minutes (CPU)
- **Model Size:** ~0.2 MB
- **Communication:** ~2 MB total

## 🛠️ Troubleshooting

### Docker Issues

**Problem:** Port 8888 already in use
```bash
# Solution: Use a different port
# Edit docker-compose.yml, change "8888:8888" to "8889:8888"
# Then access: http://localhost:8889
```

**Problem:** Container won't start
```bash
# Solution: Rebuild the image
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Python Issues

**Problem:** Module not found
```bash
# Solution: Reinstall in editable mode
pip install -e .
```

**Problem:** CUDA out of memory
```bash
# Solution: The code automatically uses CPU if no GPU
# Or reduce batch size in pyproject.toml
```

### Dataset Issues

**Problem:** Dataset download fails
```bash
# Solution: The dataset will auto-download on first run
# If it fails, check your internet connection
# Or manually download from: https://huggingface.co/datasets/zalando-datasets/fashion_mnist
```

## 💡 Tips for Students

1. **Run cells in order** - Don't skip ahead in notebooks
2. **Read the comments** - Code comments explain what's happening
3. **Do the exercises** - They reinforce learning
4. **Experiment** - Try changing parameters and see what happens
5. **Ask questions** - Use the course forum or office hours

## 🎓 Learning Objectives Checklist

After completing the tutorial, you should be able to:

- [ ] Explain how federated learning works
- [ ] Describe the FedAvg algorithm
- [ ] Implement a Flower client
- [ ] Implement a Flower server
- [ ] Run federated learning experiments
- [ ] Analyze FL results and trade-offs
- [ ] Compare FL with centralized learning
- [ ] Understand communication costs in FL

## 📖 Additional Exercises

Want to learn more? Try these extensions:

1. **Non-IID Data**: Modify notebook 2 to use non-IID partitioning
2. **More Clients**: Change num-supernodes to 10 or 20
3. **Different Models**: Replace the CNN with a larger network
4. **Different Dataset**: Use CIFAR-10 instead of FashionMNIST
5. **Privacy Analysis**: Research differential privacy in FL

## 🔗 Next Steps

After mastering the basics:
- Explore [Flower examples](https://github.com/adap/flower/tree/main/examples)
- Read research papers on FL
- Implement advanced strategies (FedProx, FedAdam)
- Deploy FL on real distributed systems

## 📞 Need Help?

- Check the [Flower documentation](https://flower.ai/docs/)
- Visit the course discussion forum
- Attend office hours
- Open an issue in this repository

## 🎉 Have Fun!

Federated learning is an exciting field with real-world impact. Enjoy the learning journey!

---

**Ready to start? Open notebook 01 and let's go! 🚀**
