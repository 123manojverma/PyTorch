# 🔥 PyTorch Learning Notes

A comprehensive collection of everything learned while studying PyTorch — from tensors and autograd to full neural network training pipelines.

---

## 📚 Table of Contents

1. [Tensors](#1-tensors)
2. [Autograd](#2-autograd)
3. [PyTorch Pipeline (Manual)](#3-pytorch-pipeline-manual)
4. [nn.Module — Building Models](#4-nnmodule--building-models)
5. [Dataset & DataLoader](#5-dataset--dataloader)
6. [Full Training Pipeline with Dataset & DataLoader](#6-full-training-pipeline-with-dataset--dataloader)
7. [ANN on Fashion MNIST](#7-ann-on-fashion-mnist)
8. [ANN on Fashion MNIST (GPU)](#8-ann-on-fashion-mnist-gpu)
9. [Files in this Repository](#9-files-in-this-repository)

---

## 1. Tensors

Tensors are the core data structure in PyTorch — similar to NumPy arrays but with GPU support and automatic differentiation.

### Tensor Creation

```python
torch.empty(2, 3)              # Uninitialized tensor
torch.zeros(2, 3)              # All zeros
torch.ones(2, 3)               # All ones
torch.rand(2, 3)               # Uniform random [0, 1)
torch.manual_seed(100)         # Set seed for reproducibility
torch.tensor([[1, 2], [3, 4]]) # From Python data
torch.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
torch.linspace(0, 10, 10)      # 10 evenly spaced values
torch.eye(5)                   # 5x5 identity matrix
torch.full((3, 3), 5)          # All 5s
```

### Shape Utilities

```python
x.shape                        # Tensor dimensions
torch.empty_like(x)            # Same shape, uninitialized
torch.zeros_like(x)            # Same shape, all zeros
torch.ones_like(x)             # Same shape, all ones
torch.rand_like(x, dtype=torch.float64)  # Same shape, random
```

### Data Types

```python
x.dtype                                  # Check dtype
torch.tensor([1.0], dtype=torch.int32)   # Specify dtype
torch.tensor([1],   dtype=torch.float64) # float64
```

### Mathematical Operations

**Scalar operations:**
```python
x + 2  |  x - 2  |  x * 3  |  x / 3
(x * 100) // 3    # Integer division
(x * 100) % 3     # Modulo
x ** 2            # Power
```

**Element-wise between two tensors:**
```python
a + b  |  a - b  |  a * b  |  a / b  |  a % b  |  a ** b
```

**Extra math functions:**
```python
torch.abs(tensor)
torch.neg(tensor)
torch.round(tensor)
torch.ceil(tensor)
torch.floor(tensor)
torch.clamp(tensor, min=..., max=...)
```

### Reduction Operations

```python
torch.sum(tensor)           # Total sum
torch.sum(tensor, dim=0)    # Sum along rows
torch.mean(tensor)          # Mean
torch.mean(tensor, dim=0)   # Mean along rows
torch.median(tensor)
torch.min(tensor) / torch.max(tensor)
torch.prod(tensor)          # Product of all elements
torch.std(tensor)
torch.var(tensor)
torch.argmax(tensor)        # Index of max value
torch.argmin(tensor)        # Index of min value
```

### Matrix Operations

```python
torch.matmul(a, b)          # Matrix multiplication
torch.dot(v1, v2)           # Dot product
torch.transpose(t, 0, 1)    # Transpose
torch.det(m)                # Determinant
torch.inverse(m)            # Inverse
```

### Comparison Operations

```python
tensor > other
tensor < other
tensor == other
tensor != other
```

### Activation & Special Functions

```python
torch.log(tensor)
torch.exp(tensor)
torch.sqrt(tensor)
torch.sigmoid(tensor)
torch.softmax(tensor, dim=0)
torch.relu(tensor)
```

### In-place Operations

```python
tensor.add_(other)    # In-place add (modifies tensor directly)
tensor.relu_()        # In-place ReLU
tensor.clone()        # Deep copy (use before modifying)
```

> **Note:** In-place ops end with `_`. Assigning `a = b` creates a reference, not a copy.

### GPU Support

```python
torch.cuda.is_available()              # Check GPU
device = torch.device('cuda')
tensor = torch.rand((2, 3), device=device)  # Create on GPU
tensor = tensor.to(device)             # Move to GPU
torch.cuda.synchronize()               # Wait for GPU to finish
```

### Tensor Reshaping

```python
tensor.reshape(...)
tensor.flatten()
tensor.permute(...)
tensor.unsqueeze(dim)   # Add a dimension
tensor.squeeze(dim)     # Remove a dimension
```

### NumPy Interoperability

```python
tensor.numpy()              # Tensor → NumPy (CPU only)
torch.from_numpy(ndarray)   # NumPy → Tensor
```

---

## 2. Autograd

Autograd is PyTorch's automatic differentiation engine. It tracks operations on tensors and computes gradients automatically.

### Key Concepts

| Concept | Description |
|---|---|
| `requires_grad=True` | Enable gradient tracking on a tensor |
| `tensor.grad` | Stores gradient after backward pass |
| `y.backward()` | Compute gradients via backpropagation |
| `x.grad.zero_()` | Clear gradients before next iteration |
| `torch.no_grad()` | Disable gradient tracking (inference) |
| `tensor.detach()` | Detach from computation graph |
| `tensor.requires_grad_(False)` | Turn off gradient tracking |

### Basic Example — Scalar

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2           # y = x²
y.backward()         # dy/dx = 2x
print(x.grad)        # → tensor(6.)
```

### Chain Rule Example — Composed Functions

```python
x = torch.tensor(4.0, requires_grad=True)
y = x ** 2
z = torch.sin(y)     # z = sin(x²)
z.backward()         # dz/dx = 2x·cos(x²)
print(x.grad)        # → tensor(-7.6613)
```

### Manual vs Autograd Comparison (Binary Cross-Entropy)

**Manual gradient computation:**
```python
dloss_dy_pred = (y_pred - y) / (y_pred * (1 - y_pred))
dy_pred_dz    = y_pred * (1 - y_pred)   # sigmoid derivative
dz_dw         = x
dz_db         = 1

dL_dw = dloss_dy_pred * dy_pred_dz * dz_dw
dL_db = dloss_dy_pred * dy_pred_dz * dz_db
```

**Autograd version:**
```python
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
z = w * x + b
y_pred = torch.sigmoid(z)
loss = binary_cross_entropy_loss(y_pred, y)
loss.backward()
print(w.grad, b.grad)  # Same result as manual computation
```

### Gradient Accumulation & Clearing

Gradients **accumulate** by default. Always clear them before the next iteration:

```python
x.grad.zero_()    # Clear gradients
```

### Disabling Gradient Tracking

```python
# Method 1: Context manager (preferred for inference)
with torch.no_grad():
    output = model(input)

# Method 2: Detach
x_no_grad = x.detach()

# Method 3: Turn off in-place
x.requires_grad_(False)
```

> **Key insight:** Only **leaf tensors** (those created by the user) accumulate `.grad`. Intermediate tensors do not unless `.retain_grad()` is called.

---

## 3. PyTorch Pipeline (Manual)

A complete supervised learning loop built from scratch — without `nn.Module` or optimizers.

### Pipeline Steps

1. Load and clean data with `pandas`
2. Split with `train_test_split`
3. Scale features with `StandardScaler`
4. Encode labels with `LabelEncoder`
5. Convert to `torch.Tensor`
6. Define model with a `forward()` method and manual parameters
7. Compute loss (e.g., binary cross-entropy)
8. Call `loss.backward()` to get gradients
9. Update weights manually with gradient descent
10. Evaluate with `torch.no_grad()`

### Example (Breast Cancer Dataset)

```python
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv(url)
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
```

---

## 4. `nn.Module` — Building Models

PyTorch's `nn.Module` is the standard base class for all neural network models.

### Defining a Model

```python
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.network(features)
```

### Using the Model

```python
features = torch.rand(10, 5)
model = Model(features.shape[1])

# Forward pass (preferred over model.forward())
output = model(features)

# Access weights
model.network[0].weight  # First Linear layer weights
model.network[0].bias    # First Linear layer bias
```

### Model Summary with `torchinfo`

```python
from torchinfo import summary
summary(model, input_size=(10, 5))
```

Output:
```
Layer (type)     Output Shape   Param #
─────────────────────────────────────
Linear           [10, 3]        18
ReLU             [10, 3]        --
Linear           [10, 1]        4
Sigmoid          [10, 1]        --
─────────────────────────────────────
Total params: 22
```

### Common `nn` Layers

| Layer | Purpose |
|---|---|
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.ReLU()` | ReLU activation |
| `nn.Sigmoid()` | Sigmoid activation |
| `nn.Sequential(...)` | Chain layers in order |

### Loss Functions

| Loss | Use Case |
|---|---|
| `nn.BCELoss()` | Binary classification |
| `nn.CrossEntropyLoss()` | Multi-class classification |
| `nn.MSELoss()` | Regression |

---

## 5. Dataset & DataLoader

PyTorch's `Dataset` and `DataLoader` provide a clean, memory-efficient way to feed data into models in mini-batches.

### Creating a Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

### Creating a DataLoader

```python
dataset    = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_features, batch_labels in dataloader:
    print(batch_features)
    print(batch_labels)
```

### Key Parameters

| Parameter | Description |
|---|---|
| `batch_size` | Number of samples per batch |
| `shuffle=True` | Shuffle data each epoch (use for training) |
| `shuffle=False` | Keep order (use for testing/evaluation) |

---

## 6. Full Training Pipeline with Dataset & DataLoader

This combines `nn.Module`, `Dataset`, `DataLoader`, a loss function, and an optimizer into a complete training loop.

### Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

### Training Loop

```python
model     = MyModel(num_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(num_epochs):
    total_loss = 0

    for batch_features, batch_labels in train_loader:
        # Forward pass
        outputs = model(batch_features)

        # Compute loss
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()   # Clear old gradients
        loss.backward()         # Compute gradients

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

### Evaluation Loop

```python
model.eval()   # Switch to eval mode (disables dropout, batchnorm, etc.)

total   = 0
correct = 0

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)
        total   += batch_labels.shape[0]
        correct += (predicted == batch_labels).sum().item()

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")
```

### Common Optimizers

| Optimizer | Notes |
|---|---|
| `optim.SGD(params, lr)` | Stochastic Gradient Descent |
| `optim.Adam(params, lr)` | Adaptive, works well in practice |
| `optim.RMSprop(params, lr)` | Good for RNNs |

---

## 7. ANN on Fashion MNIST

A full Artificial Neural Network trained on the Fashion MNIST dataset using the complete PyTorch pipeline.

### Dataset

- **Source:** `fmnist_small.csv` (28×28 grayscale images flattened to 784 pixels)
- **Labels:** 10 clothing categories (0–9)
- **Preprocessing:** Pixel values normalized to [0, 1] by dividing by 255

### Model Architecture

```
Input (784)  →  Linear(784, 128)  →  ReLU
             →  Linear(128, 64)   →  ReLU
             →  Linear(64, 10)    →  (logits)
```

```python
class MyNN(nn.Module):
    def __init__(self, num_features):
        super(MyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 100 |
| Learning Rate | 0.1 |
| Optimizer | SGD |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |

### Results

- **Final Training Loss:** ~0.001 (after 100 epochs)
- **Test Accuracy:** **83.5%**

### Visualization

Images were visualized before training using `matplotlib`:

```python
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    img = df.iloc[i, 1:].values.reshape(28, 28)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Label: {df.iloc[i, 0]}")
plt.show()
```

### Fashion MNIST Label Map

| Label | Category |
|---|---|
| 0 | T-shirt/Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## 8. ANN on Fashion MNIST (GPU)

The same ANN architecture as Section 7, but retrained with GPU acceleration enabled for faster training.

### Key Differences from CPU Version

| Aspect | CPU Version | GPU Version |
|---|---|---|
| Device | `cpu` | `cuda` |
| Tensor placement | Default | `.to(device)` on model & data |
| Training speed | Baseline | Significantly faster on large datasets |

### GPU Setup

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyNN(num_features).to(device)

# Move batch tensors to GPU inside the training loop
for batch_features, batch_labels in train_loader:
    batch_features = batch_features.to(device)
    batch_labels   = batch_labels.to(device)
    ...
```

---

## 9. Files in this Repository

| File | Description |
|---|---|
| [`Tensors.ipynb`](./Tensors.ipynb) | Tensor creation, dtypes, operations, GPU, reshaping, NumPy interop |
| [`pytorch_autograd.ipynb`](./pytorch_autograd.ipynb) | Autograd, backward passes, manual vs autograd gradients, no_grad |
| [`Pipeline.ipynb`](./Pipeline.ipynb) | Manual pipeline for breast cancer classification |
| [`Pipeline1.ipynb`](./Pipeline1.ipynb) | Variation/extension of the manual pipeline |
| [`pytorch_nn_module.ipynb`](./pytorch_nn_module.ipynb) | Building models with `nn.Module` and `nn.Sequential` |
| [`dataset_and_dataloader_demo.ipynb`](./dataset_and_dataloader_demo.ipynb) | Custom `Dataset` class and `DataLoader` usage |
| [`pytorch_training_pipeline_using_dataset_and_dataloader.ipynb`](./pytorch_training_pipeline_using_dataset_and_dataloader.ipynb) | Full pipeline with Dataset, DataLoader, optimizer, training & eval loops |
| [`ann_fashion_mnist_python.ipynb`](./ann_fashion_mnist_python.ipynb) | ANN trained on Fashion MNIST (83.5% accuracy) |
| [`ann_fashion_mnist_python_gpu.ipynb`](./ann_fashion_mnist_python_gpu.ipynb) | ANN on Fashion MNIST with GPU acceleration (`cuda`) |
| [`fmnist_small.csv`](./fmnist_small.csv) | Fashion MNIST dataset (CSV format) |

---

## Quick Reference Cheatsheet

```python
# Tensor basics
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Forward
y = x ** 2
z = y.mean()

# Backward
z.backward()
print(x.grad)        # Gradients

# Model building
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Training step
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = criterion(model(X), y)
loss.backward()
optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test)
```
