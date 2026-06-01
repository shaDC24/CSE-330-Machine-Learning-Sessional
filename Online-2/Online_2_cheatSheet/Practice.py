import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import datasets, transforms
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split


'''
#FNN with binary classification on Iris dataset
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

num_epochs = 25
# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Binary classification: Iris-V

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.long)

dataset=TensorDataset(X_train_tensor,y_train_tensor)
dataloader=DataLoader(dataset,batch_size=16,shuffle=True)


class BinaryFNN(nn.Module):
    def __init__(self, num_features):
        super(BinaryFNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
    
model=BinaryFNN(num_features=X_train.shape[1]).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (100.0 * correct) / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f}, "
          f"Accuracy: {epoch_acc:.2f}%")
    
'''


'''
#FNN with multiple hidden layers
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

num_epochs = 25
# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target  # Multi-class classification: Iris-V

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.long)

dataset=TensorDataset(X_train_tensor,y_train_tensor)
dataloader=DataLoader(dataset,batch_size=16,shuffle=True)



class MultiClassFNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MultiClassFNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)
        # self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out= self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out= self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out= self.dropout(out)
        out = self.fc4(out)
        return out
    
model=MultiClassFNN(num_features=X_train.shape[1],num_classes=3).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

num_epochs=50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (100.0 * correct) / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f}, "
          f"Accuracy: {epoch_acc:.2f}%")

model.eval()
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    acc = 100.0 * (predicted == y_test).sum().item() / y_test.size(0)

print(f"Test Accuracy: {acc:.2f}%")

y_true = y_test.cpu().numpy()
y_pred = predicted.cpu().numpy()

cm = confusion_matrix(y_true, y_pred)
print(cm)




class_names = ['Setosa', 'Versicolor', 'Virginica']

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add numbers inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()
'''


'''

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
print("Using device:", device)

# -----------------------------
# Model (More Complex CNN)
# -----------------------------
class ComplexCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ComplexCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # Input: 1x32x32, Output: 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # Output: 32x16x16
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # Output: 64x8x8
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))     # Output: 128x1x1
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# -----------------------------
# Data (Any Image Dataset)
# -----------------------------
# For MNIST
transform = transforms.Compose([
    transforms.Resize((32, 32)),   # Resize images to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean & Std of MNIST
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
total_size = len(dataset)
test_size = int(0.2 * total_size)   # 20% for test
train_size = total_size - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# Training Setup
# -----------------------------
model = ComplexCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
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
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"Test Accuracy: {test_acc:.2f}%")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
print(cm)

# Plot
class_names = [str(i) for i in range(10)]  # MNIST digits
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("MNIST Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=8)

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# -----------------------------
# 1. Setup & Data (CIFAR-10)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# এই আর্কিটেকচারগুলো ২২৪x২২৪ ইমেজে ভালো কাজ করে, তাই আমরা রিসাইজ করছি
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# -----------------------------
# 2. AlexNet Implementation
# -----------------------------
# স্লাইড অনুযায়ী: ৫টি Conv লেয়ার এবং ৩টি FC লেয়ার
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -----------------------------
# 3. VGG-16 Implementation
# -----------------------------
# স্লাইড অনুযায়ী: ছোট ৩x৩ ফিল্টার ব্যবহার করে গভীর নেটওয়ার্ক
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        self.classifier = nn.Linear(512 * 7 * 7, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -----------------------------
# 4. ResNet (Residual Block)
# -----------------------------
# স্লাইড অনুযায়ী: Skip connection বা Identity Mapping
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Skip Connection
        return torch.relu(out)

# -----------------------------
# 5. Training Loop (Example with AlexNet)
# -----------------------------
model = AlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting Training...")
for epoch in range(1): # টেস্টের জন্য ১ এপোক
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}], Step [{i+1}], Loss: {loss.item():.4f}')



import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Conv Layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Activation
        self.act = nn.Tanh()

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer
        # 64x64 -> conv -> 64x64 -> pool -> 32x32
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Training Setup
model = SimpleCNN()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.LeakyReLU(0.01)
        self.pool = nn.MaxPool2d(2, 2)

        # 64x64 -> pool -> 32x32
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Training Setup
model = SimpleCNN()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(32 * 32 * 32, 10)

        # He Initialization
        nn.init.kaiming_normal_(
            self.conv1.weight,
            mode='fan_out',
            nonlinearity='relu'
        )

        # Xavier for FC
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

# Training Setup
model = SimpleCNN()
criterion = nn.NLLLoss()



import torch
import torch.nn as nn

class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.std=(p/(1-p))**0.5

    def forward(self, x):
        if self.training:
            # TODO
            noise = torch.randn_like(x) * self.std + 1
            x = x * noise 
        return x
import torch
import torch.nn as nn

class AlphaDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.act = nn.SELU()
        self.dropout = nn.AlphaDropout(p=0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # TODO
        x=self.fc1(x)
        x=self.act(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class DropConnectLinear(nn.Linear):
    def __init__(self, in_features, out_features, p=0.5):
        super().__init__(in_features, out_features)
        self.p = p

    def forward(self, x):
        # TODO
        if self.training:
            mask = (torch.rand_like(self.weight) > self.p).float()
            weight = self.weight * mask / (1 - self.p)
        else:
            weight = self.weight
        return F.linear(x, weight, self.bias)


