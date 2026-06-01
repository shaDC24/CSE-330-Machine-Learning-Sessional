"""
ML Online 2 - Practice Problems with Solutions
Topics: Optimization, FNN, CNN

Problem নম্বর change করে run করতে হবে (CURRENT_PROBLEM variable)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# কোন problem run করতে চাও সেটা এখানে set করো (1-5)
CURRENT_PROBLEM = 1

# ==============================================================================
# PROBLEM 1: Simple FNN for Binary Classification
# ==============================================================================
# Task: 2 features থেকে binary (0/1) classification করতে হবে
# Complete the missing parts marked with # YOUR CODE HERE

def problem_1():
    print("=" * 60)
    print("PROBLEM 1: Simple FNN for Binary Classification")
    print("=" * 60)
    
    # Dummy dataset
    X = torch.randn(100, 2)  # 100 samples, 2 features
    y = (X[:, 0] + X[:, 1] > 0).long()  # Binary labels
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Model Definition
    class BinaryFNN(nn.Module):
        def __init__(self):
            super(BinaryFNN, self).__init__()
            # YOUR CODE HERE: Define 3 linear layers
            # Input: 2, Hidden: 8, Output: 2 (for binary classification)
            self.fc1 = nn.Linear(2, 8)
            self.fc2 = nn.Linear(8, 8)
            self.fc3 = nn.Linear(8, 2)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # YOUR CODE HERE: Complete forward pass
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = BinaryFNN()
    criterion = nn.CrossEntropyLoss()
    # YOUR CODE HERE: Define Adam optimizer with lr=0.01
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    for epoch in range(5):
        for inputs, labels in loader:
            # YOUR CODE HERE: Complete training loop
            # 1. Zero gradients
            # 2. Forward pass
            # 3. Calculate loss
            # 4. Backward pass
            # 5. Update weights
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
    
    print("\n✓ Problem 1 Solution Complete!\n")


# ==============================================================================
# PROBLEM 2: Compare SGD vs Adam Optimizer
# ==============================================================================
# Task: Same model, different optimizers - observe the difference

def problem_2():
    print("=" * 60)
    print("PROBLEM 2: SGD vs Adam Comparison")
    print("=" * 60)
    
    # Dataset
    X = torch.randn(200, 4)
    y = (X.sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(4, 16)
            self.fc2 = nn.Linear(16, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Train with SGD
    print("\nTraining with SGD:")
    model_sgd = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    # YOUR CODE HERE: Create SGD optimizer with lr=0.01
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
    
    for epoch in range(5):
        total_loss = 0
        for inputs, labels in loader:
            optimizer_sgd.zero_grad()
            outputs = model_sgd(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_sgd.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
    
    # Train with Adam
    print("\nTraining with Adam:")
    model_adam = SimpleModel()
    # YOUR CODE HERE: Create Adam optimizer with lr=0.01
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.01)
    
    for epoch in range(5):
        total_loss = 0
        for inputs, labels in loader:
            optimizer_adam.zero_grad()
            outputs = model_adam(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_adam.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
    
    print("\n✓ Problem 2 Solution Complete!")
    print("Notice: Adam usually converges faster!\n")


# ==============================================================================
# PROBLEM 3: Build a CNN from Scratch
# ==============================================================================
# Task: Complete CNN architecture for image classification

def problem_3():
    print("=" * 60)
    print("PROBLEM 3: CNN Architecture")
    print("=" * 60)
    
    # Fake image data: 50 images, 1 channel, 28x28
    X = torch.randn(50, 1, 28, 28)
    y = torch.randint(0, 10, (50,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # YOUR CODE HERE: Complete CNN layers
            # Conv1: 1 input channel -> 16 output, kernel_size=3, padding=1
            # Conv2: 16 -> 32, kernel_size=3, padding=1
            # MaxPool: kernel_size=2
            # FC: Calculate input size (32 * 7 * 7 after 2 pooling) -> 10 classes
            
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(32 * 7 * 7, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # YOUR CODE HERE: Complete forward pass
            # conv1 -> relu -> pool -> conv2 -> relu -> pool -> flatten -> fc
            
            x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
            x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc(x)
            return x
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\nTraining CNN...")
    for epoch in range(3):
        total_loss = 0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/3, Loss: {total_loss/len(loader):.4f}")
    
    print("\n✓ Problem 3 Solution Complete!\n")


# ==============================================================================
# PROBLEM 4: FNN with Multiple Hidden Layers
# ==============================================================================
# Task: Build deeper FNN and use different activation functions

def problem_4():
    print("=" * 60)
    print("PROBLEM 4: Deep FNN with Dropout")
    print("=" * 60)
    
    # Regression task dataset
    X = torch.randn(100, 5)
    y = (X.sum(dim=1, keepdim=True) * 2 + torch.randn(100, 1) * 0.1)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    
    class DeepFNN(nn.Module):
        def __init__(self):
            super(DeepFNN, self).__init__()
            # YOUR CODE HERE: Build network
            # Input: 5 -> Hidden1: 32 -> Hidden2: 16 -> Output: 1
            # Use Dropout(0.2) between hidden layers
            
            self.fc1 = nn.Linear(5, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.dropout = nn.Dropout(0.2)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # YOUR CODE HERE: fc1 -> relu -> dropout -> fc2 -> relu -> fc3
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = DeepFNN()
    # YOUR CODE HERE: Use MSELoss for regression
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    model.train()  # Important for dropout
    for epoch in range(5):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5, MSE Loss: {total_loss/len(loader):.4f}")
    
    print("\n✓ Problem 4 Solution Complete!\n")


# ==============================================================================
# PROBLEM 5: Complete Training Loop (Most Important!)
# ==============================================================================
# Task: Fill in the blanks in a complete training setup

def problem_5():
    print("=" * 60)
    print("PROBLEM 5: Complete Training Loop")
    print("=" * 60)
    print("This is the MOST COMMON exam pattern!\n")
    
    # Multi-class classification
    X = torch.randn(150, 10)
    y = torch.randint(0, 3, (150,))  # 3 classes
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=30, shuffle=True)
    
    # Model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    )
    
    # YOUR CODE HERE: Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Complete training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            # YOUR CODE HERE: Complete these 5 steps
            # Step 1: Zero the gradients
            optimizer.zero_grad()
            
            # Step 2: Forward pass
            outputs = model(inputs)
            
            # Step 3: Calculate loss
            loss = criterion(outputs, labels)
            
            # Step 4: Backward pass
            loss.backward()
            
            # Step 5: Update weights
            optimizer.step()
            
            # Calculate accuracy
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    print("\n✓ Problem 5 Solution Complete!")
    print("\n" + "="*60)
    print("KEY POINTS TO REMEMBER:")
    print("="*60)
    print("1. Training Loop এর 5 steps মুখস্থ করো:")
    print("   - optimizer.zero_grad()")
    print("   - outputs = model(inputs)")
    print("   - loss = criterion(outputs, labels)")
    print("   - loss.backward()")
    print("   - optimizer.step()")
    print("\n2. Classification: CrossEntropyLoss")
    print("   Regression: MSELoss or L1Loss")
    print("\n3. Optimizer: Adam (fast) vs SGD (stable)")
    print("="*60 + "\n")


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    problems = {
        1: problem_1,
        2: problem_2,
        3: problem_3,
        4: problem_4,
        5: problem_5
    }
    
    print("\n" + "="*60)
    print("ML ONLINE 2 - PRACTICE PROBLEMS")
    print("="*60)
    print("Available Problems:")
    print("  1. Simple FNN for Binary Classification")
    print("  2. SGD vs Adam Comparison")
    print("  3. CNN Architecture from Scratch")
    print("  4. Deep FNN with Dropout")
    print("  5. Complete Training Loop (MOST IMPORTANT!)")
    print("="*60)
    print(f"\nRunning Problem {CURRENT_PROBLEM}...\n")
    
    if CURRENT_PROBLEM in problems:
        problems[CURRENT_PROBLEM]()
    else:
        print("Invalid problem number! Set CURRENT_PROBLEM to 1-5")
    
    print("\n💡 TIP: Change CURRENT_PROBLEM variable (line 12) to practice different problems!")
    print("📚 Study all 5 problems before your exam!\n")