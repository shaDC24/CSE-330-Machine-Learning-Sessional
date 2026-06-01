# 🔥 PyTorch Quick Reference Sheet

## 📦 Essential Imports
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
```

---

## 🎯 Tensor Basics

### Creating Tensors
```python
torch.tensor([1, 2, 3])           # From list
torch.randn(3, 4)                 # Random 3×4 tensor
torch.zeros(2, 3)                 # All zeros
torch.ones(2, 3)                  # All ones
torch.arange(0, 10, 2)            # [0, 2, 4, 6, 8]
```

### Tensor Operations
```python
x.shape                           # Get shape
x.view(a, b)                      # Reshape to (a, b)
x.view(-1, n)                     # Auto-calculate first dim
x.reshape(a, b)                   # Alternative to view
x.size(0)                         # Get dimension 0 size
x.numel()                         # Total number of elements
```

### GPU Operations
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)                  # Move tensor to GPU
model.to(device)                  # Move model to GPU
```

---

## 🏗️ Building Models

### Method 1: Sequential (সহজ)
```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)
```

### Method 2: Custom Class (Flexible)
```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MyModel()
```

---

## 🧱 Common Layers

### Fully Connected
```python
nn.Linear(in_features, out_features)
# Example: nn.Linear(784, 128)
```

### Convolutional
```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
# Example: nn.Conv2d(1, 16, 3, padding=1)
```

### Pooling
```python
nn.MaxPool2d(kernel_size, stride=None)
# Example: nn.MaxPool2d(2)  # 2×2 pooling
```

### Normalization
```python
nn.BatchNorm1d(num_features)  # For FC layers
nn.BatchNorm2d(num_features)  # For Conv layers
```

### Regularization
```python
nn.Dropout(p=0.5)            # Randomly drop 50% neurons
```

### Utilities
```python
nn.Flatten()                 # Multi-dim → 1D
```

---

## ⚡ Activation Functions

| Function | Range         | Code ----------------------           | Use Case                    |
|----------|-------        |----------------------------           |----------                   |
| ReLU     | [0, ∞)        | `F.relu(x)` or `nn.ReLU()`            | Hidden layers (most common) |
| Sigmoid  | [0, 1]        | `torch.sigmoid(x)` or `nn.Sigmoid()`   | Binary output, probabilities |
| Tanh     | [-1, 1]       | `torch.tanh(x)` or `nn.Tanh()`         | Hidden layers, RNN |
| Softmax | [0, 1] (sum=1) | `F.softmax(x, dim=1)`                  | Multi-class output |

---

## 📉 Loss Functions

### Classification
```python
nn.CrossEntropyLoss()        # Multi-class (includes softmax)
nn.BCELoss()                 # Binary classification (need sigmoid first)
nn.BCEWithLogitsLoss()       # Binary (sigmoid included)
```

### Regression
```python
nn.MSELoss()                 # Mean Squared Error (L2)
nn.L1Loss()                  # Mean Absolute Error
```

---

## 🔧 Optimizers

```python
# SGD - Basic, stable
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam - Most popular, adaptive ⭐
optim.Adam(model.parameters(), lr=0.001)

# AdamW - Adam with weight decay
optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop - Good for RNNs
optim.RMSprop(model.parameters(), lr=0.01)
```

**সাধারণত Adam ব্যবহার করো!**

---

## 📊 Data Loading

### Simple Way: TensorDataset
```python
X = torch.randn(100, 5)
y = torch.randint(0, 3, (100,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Custom Dataset
```python
class MyDataset(Dataset):
    def __init__(self):
        self.data = ...
        self.labels = ...
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 🎓 Training Loop (এটা মুখস্থ করো!)

```python
# Setup
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Training mode
    
    for inputs, labels in train_loader:
        # Move to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # ⭐ 5 GOLDEN STEPS ⭐
        optimizer.zero_grad()           # 1. Clear gradients
        outputs = model(inputs)         # 2. Forward pass
        loss = criterion(outputs, labels)  # 3. Calculate loss
        loss.backward()                 # 4. Backward pass
        optimizer.step()                # 5. Update weights
    
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

---

## 🧪 Evaluation Loop

```python
model.eval()  # Evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No gradient calculation
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
```

---

## 💾 Save & Load Models

### Save (Recommended Method)
```python
torch.save(model.state_dict(), 'model.pth')
```

### Load
```python
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Load on Different Device
```python
# Train on GPU, load on CPU
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# Train on CPU, load on GPU
model.load_state_dict(torch.load('model.pth', map_location='cuda'))
```

---

## 🐛 Common Mistakes to Avoid

### ❌ Forgetting to zero gradients
```python


### ❌ Wrong loss function
```python
# Multi-class classification (0-9)
# CORRECT ✓: nn.CrossEntropyLoss()

# Binary classification (0 or 1)
# CORRECT ✓: nn.BCELoss() or nn.BCEWithLogitsLoss()
```

### ❌ Shape mismatch in CNN → FC
```python

# CORRECT ✓
x = conv_layers(x)
x = x.view(x.size(0), -1)  # Flatten
x = fc_layer(x)
```

### ❌ Not moving data to GPU
```python


# CORRECT ✓
model = model.to(device)
for inputs, labels in loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
```

### ❌ Using model.train() during evaluation
```python

# CORRECT ✓
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
```

---

## 🎯 CNN Architecture Example

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 28×28 → 28×28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 14×14 → 14×14
        self.pool = nn.MaxPool2d(2, 2)                # Halves size
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Other layers
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))  # 28×28
        x = self.pool(x)            # 14×14
        
        # Conv block 2
        x = F.relu(self.conv2(x))  # 14×14
        x = self.pool(x)            # 7×7
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 32*7*7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

---

## 📝 পরীক্ষার জন্য Must-Remember

### 1. Training Loop এর 5 Steps (এটা 100% আসবে!)
```
1. optimizer.zero_grad()
2. outputs = model(inputs)
3. loss = criterion(outputs, labels)
4. loss.backward()
5. optimizer.step()
```

### 2. Common Patterns
- **Classification:** CrossEntropyLoss + Adam
- **Regression:** MSELoss + Adam
- **CNN:** Conv2d → ReLU → MaxPool → Flatten → Linear
- **FNN:** Linear → ReLU → Dropout → Linear

### 3. Shape Calculations
- Conv2d: `out_size = (in_size + 2*padding - kernel_size) / stride + 1`
- MaxPool2d: `out_size = in_size / kernel_size`
- Linear: `(batch_size, in_features) → (batch_size, out_features)`

### 4. Debugging Tips
```python
print(x.shape)        # Check tensor shape
print(model)          # See model architecture
print(loss.item())    # Get loss value as number
```

---

## Adagrad implementation

def adagrad_update(params, grads, states, lr=0.01, eps=1e-8):
    for p, g in zip(params, grads):
        # state['r'] 
        if p not in states:
            states[p] = torch.zeros_like(p)
            
        # 1. r = r + g^2
        states[p] += g**2
        
        # 2. theta = theta - (lr / sqrt(r + eps)) * g
        p.data -= (lr / (torch.sqrt(states[p]) + eps)) * g

## RMSprop implementation

def rmsprop_update(params, grads, states, lr=0.01, rho=0.9, eps=1e-8):
    for p, g in zip(params, grads):
        if p not in states:
            states[p] = torch.zeros_like(p)
            
        # 1. r = rho * r + (1 - rho) * g^2
        states[p] = rho * states[p] + (1 - rho) * (g**2)
        
        # 2. theta = theta - (lr / sqrt(r + eps)) * g
        p.data -= (lr / (torch.sqrt(states[p]) + eps)) * g


## ADAM implementation

def adam_update(params, grads, states, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    for p, g in zip(params, grads):
        if p not in states:
            states[p] = {'s': torch.zeros_like(p), 'r': torch.zeros_like(p)}
        
        state = states[p]
        
        # 1. First moment (Momentum): s = beta1 * s + (1 - beta1) * g
        state['s'] = beta1 * state['s'] + (1 - beta1) * g
        
        # 2. Second moment (RMSProp style): r = beta2 * r + (1 - beta2) * g^2
        state['r'] = beta2 * state['r'] + (1 - beta2) * (g**2)
        
        # 3. Bias correction
        s_hat = state['s'] / (1 - beta1**t)
        r_hat = state['r'] / (1 - beta2**t)
        
        # 4. Update: theta = theta - lr * (s_hat / (sqrt(r_hat) + eps))
        p.data -= lr * (s_hat / (torch.sqrt(r_hat) + eps))
## How to implement them in code

# ট্রেনিং লুপের ভেতরে
t = 0 # Adam এর জন্য timestep
states = {}

for epoch in range(num_epochs):
    for X, y in train_loader:
        t += 1
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        
        with torch.no_grad():
            # তোমার পছন্দমতো যেকোনো একটি ফাংশন কল করো
            adam_update(model.parameters(), [p.grad for p in model.parameters()], states, t)
            
            # গ্রাডিয়েন্ট ক্লিয়ার করা
            model.zero_grad()