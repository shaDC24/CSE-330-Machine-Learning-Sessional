import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.optim.optimizer import Optimizer


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# -----------------------------
# Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU()

   
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.gap(x)        
        x = x.view(x.size(0), -1)  

        x = self.fc(x)
        return x

# -----------------------------
# Data
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
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
# -----------------------------
# Training Setup
# -----------------------------
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training Loop
# -----------------------------
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f}, "
          f"Accuracy: {epoch_acc:.2f}%")
    