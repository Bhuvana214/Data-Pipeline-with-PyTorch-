# PyTorch Tutorials Collection

This repository contains two Jupyter notebooks that introduce core PyTorch concepts and practical code scenarios for beginners and intermediate users. The notebooks comprehensively cover data pipelines, gradient computation, model design, loss functions, training loops, and evaluation metrics in PyTorch.

---

## Notebook 1: Data Pipelines with PyTorch

**Filename:** `Data Pipelines with PyTorch.ipynb`

This notebook provides a hands-on introduction to fundamental PyTorch practices, focusing on:

- **Tensor Operations and Gradients:**  
  - Demonstrates how to create tensors with `requires_grad=True`.
  - Calculates gradients using the `.backward()` method.
  - Explains how gradients flow through computation graphs and their importance in neural network training.

- **Custom Datasets and Data Pipelines:**  
  - Builds custom datasets using the PyTorch `Dataset` class.
  - Shows how to manage sample features and labels.
  - Splits datasets into training and testing sets with `random_split`.

**Learning Objectives:**
- Understand PyTorch’s gradient tracking and automatic differentiation.
- Create custom datasets for more flexible data loading.
- Prepare datasets for model training and evaluation.

---

## Notebook 2: Neural Network with PyTorch

**Filename:** `Neural Network with PyTorch.ipynb`

This notebook moves from data handling to model creation and evaluation, including:

- **Custom Dataset Implementation:**  
  - Defines and validates a dataset class for features and labels.

- **Neural Network Design:**  
  - Implements a simple feed-forward neural network using `nn.Module`.
  - Discusses device selection (CPU/GPU) and non-linear activation functions (e.g., `Tanh`).
  - Shows how non-linearities allow neural networks to learn complex patterns.

- **Loss Functions and Outliers:**  
  - Illustrates how to use L1 (Mean Absolute Error) and MSE (Mean Squared Error) loss.
  - Experimentally compares their sensitivity to outliers.

- **Training Loops:**  
  - Demonstrates basic model training:
    - Forward and backward passes
    - Loss calculation
    - Parameter updates using stochastic gradient descent
    - Monitoring training progress via loss per epoch

- **Evaluation Metrics:**  
  - Calculates and explains key regression metrics such as MAE and R².
  - Explores classification metrics: precision, recall, accuracy, etc.
  - Shows evaluation best practices using `torch.no_grad()` to disable gradient calculation during inference.

**Learning Objectives:**
- Build and train a simple neural network with PyTorch.
- Select and interpret loss functions appropriate for different scenarios.
- Evaluate model performance with standard metrics.
- Understand correct usage for model evaluation in PyTorch.

---

## Getting Started

**Requirements:**
- Python 3.7+
- PyTorch (Tested on 1.13 or higher)
- `matplotlib` and `scikit-learn` for metrics evaluation and plotting (second notebook)
- Jupyter Notebook or JupyterLab

**How to Run:**

```bash
# (Optional) Create and activate a virtual environment
conda create -n pytorch-tutorials python=3.10 -y
conda activate pytorch-tutorials

# Install PyTorch (choose CUDA version as appropriate)
pip install torch torchvision

# Install other dependencies
pip install matplotlib scikit-learn jupyter

# Start Jupyter
jupyter notebook
```

Open the `.ipynb` files in Jupyter Notebook or JupyterLab, and execute the cells step by step to interactively follow the examples.

---

## File Summary

| File                              | Topic                                         |
|----------------------------------- |-----------------------------------------------|
| Data Pipelines with PyTorch.ipynb  | Tensors, gradients, custom datasets, splitting|
| Neural Network with PyTorch.ipynb   | Custom datasets, model architecture, loss, training, evaluation metrics |

---

## License

This repository is for educational purposes.

---
