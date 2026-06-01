"""
PyTorch Complete Guide - PDF এর সব কিছু Example সহ
প্রতিটা section আলাদা আলাদা run করতে পারবে

Section change করতে CURRENT_SECTION variable পরিবর্তন করো
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

# কোন section দেখতে চাও? (1-10)
CURRENT_SECTION = 7

print("="*70)
print("PyTorch Complete Tutorial - বাংলায়")
print("="*70)

# =============================================================================
# SECTION 1: BASIC IMPORTS এবং PACKAGES
# =============================================================================
def section_1_imports():
    """
    PyTorch এর মূল packages এবং তাদের কাজ
    """
    print("\n" + "="*70)
    print("SECTION 1: Basic Imports এবং তাদের ব্যবহার")
    print("="*70)
    
    print("\n1️⃣ import torch")
    print("   → PyTorch এর root package, সব basic functions এখানে")
    print("   Example:")
    x = torch.tensor([1, 2, 3])
    print(f"   torch.tensor([1,2,3]) = {x}")
    
    print("\n2️⃣ import torch.nn as nn")
    print("   → Neural Network layers, models এর জন্য")
    print("   Example:")
    layer = nn.Linear(5, 3)  # 5 input → 3 output
    print(f"   nn.Linear(5, 3) creates a layer with shape: {layer.weight.shape}")
    
    print("\n3️⃣ import torch.nn.functional as F")
    print("   → Activation functions (ReLU, Sigmoid), Loss functions")
    print("   Example:")
    x = torch.tensor([-1.0, 0.0, 1.0])
    print(f"   Input: {x}")
    print(f"   F.relu(x) = {F.relu(x)}  (negative values → 0)")
    
    print("\n4️⃣ from torchvision import datasets, models, transforms")
    print("   → Image datasets (MNIST, CIFAR), pre-trained models, transforms")
    print("   Example:")
    print("   datasets.MNIST() → MNIST handwritten digits dataset")
    print("   transforms.ToTensor() → Image কে tensor এ convert করে")
    
    print("\n✓ Section 1 Complete!\n")


# =============================================================================
# SECTION 2: TENSOR OPERATIONS
# =============================================================================
def section_2_tensors():
    """
    Tensor তৈরি করা এবং manipulation
    """
    print("\n" + "="*70)
    print("SECTION 2: Tensor Operations")
    print("="*70)
    
    print("\n1️⃣ torch.Tensor(L) - List থেকে tensor তৈরি")
    my_list = [1, 2, 3, 4]
    tensor_from_list = torch.Tensor(my_list)
    print(f"   List: {my_list}")
    print(f"   Tensor: {tensor_from_list}")
    print(f"   Shape: {tensor_from_list.shape}")
    
    print("\n2️⃣ torch.randn(*size) - Random tensor তৈরি")
    random_tensor = torch.randn(3, 4)  # 3 rows, 4 columns
    print(f"   torch.randn(3, 4):")
    print(f"{random_tensor}")
    print(f"   Shape: {random_tensor.shape}")
    
    print("\n3️⃣ tensor.view(a, b, ...) - Shape পরিবর্তন করা")
    original = torch.randn(2, 6)  # 2×6 tensor
    reshaped = original.view(3, 4)  # 3×4 তে convert
    print(f"   Original shape: {original.shape}")
    print(f"   After .view(3, 4): {reshaped.shape}")
    print(f"   Total elements same: {original.numel()} = {reshaped.numel()}")
    
    # view(-1, n) এর মানে
    batch_view = torch.randn(2, 3, 4).view(-1, 4)
    print(f"\n   torch.randn(2,3,4).view(-1, 4):")
    print(f"   Original: (2,3,4), After: {batch_view.shape}")
    print(f"   -1 মানে automatically calculate করো (এখানে 6)")
    
    print("\n4️⃣ requires_grad=True - Gradient tracking চালু করা")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2  # y = x²
    z = y.sum()  # z = sum of all y
    z.backward()  # Calculate gradients
    print(f"   x = {x.data}")
    print(f"   y = x² = {y.data}")
    print(f"   z = sum(y) = {z.item()}")
    print(f"   dz/dx = {x.grad}  (gradient of z w.r.t x)")
    print(f"   (Manual check: dz/dx = 2x = {2*x.data})")
    
    print("\n✓ Section 2 Complete!\n")


# =============================================================================
# SECTION 3: DEFINING MODELS - দুইভাবে
# =============================================================================
def section_3_model_definition():
    """
    Neural Network define করার দুইটা method
    """
    print("\n" + "="*70)
    print("SECTION 3: Model Definition (2 Ways)")
    print("="*70)
    
    print("\n📌 Method A: nn.Sequential (সহজ, sequential layers এর জন্য)")
    print("-" * 70)
    
    model_a = nn.Sequential(
        nn.Conv2d(1, 16, 3),      # Input: 1 channel, Output: 16, Kernel: 3×3
        nn.ReLU(),                # Activation
        nn.MaxPool2d(2),          # Pooling with 2×2 kernel
        nn.Flatten(),             # 2D → 1D করা
        nn.Linear(16 * 13 * 13, 10)  # Fully connected layer
    )
    
    print("model = nn.Sequential(")
    print("    nn.Conv2d(1, 16, 3),")
    print("    nn.ReLU(),")
    print("    nn.MaxPool2d(2),")
    print("    nn.Flatten(),")
    print("    nn.Linear(16*13*13, 10)")
    print(")")
    
    # Test
    dummy_input = torch.randn(1, 1, 28, 28)  # 1 image, 1 channel, 28×28
    output = model_a(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Sequential model সহজ কিন্তু flexibility কম")
    
    print("\n" + "-"*70)
    print("📌 Method B: Custom Class (flexible, custom logic এর জন্য)")
    print("-" * 70)
    
    class CustomNet(nn.Module):
        def __init__(self):
            super(CustomNet, self).__init__()
            # Layers define করো __init__ এ
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 14 * 14, 64)
            self.fc2 = nn.Linear(64, 10)
        
        def forward(self, x):
            # Forward pass logic এখানে
            x = self.conv1(x)           # Conv layer
            x = F.relu(x)               # Activation
            x = self.pool(x)            # Pooling
            x = x.view(-1, 16*14*14)    # Flatten
            x = F.relu(self.fc1(x))     # FC layer 1
            x = self.fc2(x)             # FC layer 2
            return x
    
    model_b = CustomNet()
    output = model_b(dummy_input)
    
    print("class CustomNet(nn.Module):")
    print("    def __init__(self):")
    print("        super().__init__()")
    print("        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)")
    print("        self.pool = nn.MaxPool2d(2, 2)")
    print("        self.fc1 = nn.Linear(16*14*14, 64)")
    print("        self.fc2 = nn.Linear(64, 10)")
    print("")
    print("    def forward(self, x):")
    print("        x = F.relu(self.conv1(x))")
    print("        x = self.pool(x)")
    print("        x = x.view(-1, 16*14*14)")
    print("        x = F.relu(self.fc1(x))")
    print("        return self.fc2(x)")
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Class method flexible, custom operations সহজ")
    
    print("\n💡 কখন কোনটা ব্যবহার করবে?")
    print("   Sequential → সহজ models (layers একের পর এক)")
    print("   Class → Complex models (skip connections, multiple paths)")
    
    print("\n✓ Section 3 Complete!\n")


# =============================================================================
# SECTION 4: LAYERS - সব ধরনের layers
# =============================================================================
def section_4_layers():
    """
    PyTorch এর সব common layers এবং তাদের ব্যবহার
    """
    print("\n" + "="*70)
    print("SECTION 4: Common Layers")
    print("="*70)
    
    print("\n1️⃣ nn.Linear(m, n) - Fully Connected Layer")
    print("   m input neurons → n output neurons")
    fc_layer = nn.Linear(5, 3)
    input_data = torch.randn(2, 5)  # 2 samples, 5 features each
    output = fc_layer(input_data)
    print(f"   Input: {input_data.shape} → Output: {output.shape}")
    print(f"   Formula: output = input @ weight.T + bias")
    
    print("\n2️⃣ nn.Conv2d(in_channels, out_channels, kernel_size)")
    print("   Image processing এর জন্য convolution layer")
    conv = nn.Conv2d(1, 16, 3, padding=1)
    img = torch.randn(1, 1, 28, 28)  # 1 image, 1 channel, 28×28
    conv_output = conv(img)
    print(f"   Input: {img.shape} → Output: {conv_output.shape}")
    print(f"   16টা filters, প্রতিটা 3×3 size")
    
    print("\n3️⃣ nn.MaxPool2d(kernel_size) - Downsampling")
    print("   Image size কমায়, important features রাখে")
    pool = nn.MaxPool2d(2)  # 2×2 pooling
    pooled = pool(conv_output)
    print(f"   Input: {conv_output.shape} → Output: {pooled.shape}")
    print(f"   28×28 → 14×14 (size অর্ধেক হয়ে গেছে)")
    
    print("\n4️⃣ nn.ReLU() - Activation Function")
    print("   Negative values → 0, Positive → থাকে same")
    relu = nn.ReLU()
    test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    activated = relu(test_input)
    print(f"   Input:  {test_input}")
    print(f"   Output: {activated}")
    
    print("\n5️⃣ nn.Dropout(p=0.5) - Overfitting রোধ করে")
    print("   Training এ randomly p% neurons বন্ধ করে দেয়")
    dropout = nn.Dropout(0.5)
    x = torch.ones(1, 10)
    dropout.train()  # Training mode
    dropped = dropout(x)
    print(f"   Original: {x}")
    print(f"   After Dropout: {dropped}")
    print(f"   (কিছু values 0 হয়ে গেছে randomly)")
    
    print("\n6️⃣ nn.Flatten() - Multi-dimensional → 1D")
    print("   CNN থেকে FC layer এ যাওয়ার আগে লাগে")
    flatten = nn.Flatten()
    x = torch.randn(2, 3, 4, 4)  # 2 samples, 3 channels, 4×4 images
    flattened = flatten(x)
    print(f"   Input: {x.shape} → Output: {flattened.shape}")
    print(f"   (2, 3, 4, 4) → (2, 48)  [3×4×4 = 48]")
    
    print("\n7️⃣ nn.BatchNorm2d(n) - Normalization")
    print("   Training stable করে, faster convergence")
    bn = nn.BatchNorm2d(16)  # 16 channels
    x = torch.randn(4, 16, 28, 28)  # 4 images, 16 channels
    normalized = bn(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Normalized shape: {normalized.shape}")
    print(f"   Mean ≈ 0, Std ≈ 1 হয়ে যায় প্রতিটা channel এ")
    
    print("\n✓ Section 4 Complete!\n")


# =============================================================================
# SECTION 5: ACTIVATION FUNCTIONS
# =============================================================================
def section_5_activations():
    """
    Different activation functions এবং কখন ব্যবহার করবে
    """
    print("\n" + "="*70)
    print("SECTION 5: Activation Functions")
    print("="*70)
    
    test_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"\nTest Input: {test_values}")
    print("-" * 70)
    
    print("\n1️⃣ ReLU (Rectified Linear Unit)")
    print("   Formula: f(x) = max(0, x)")
    print("   Range: 0 to ∞")
    print("   ব্যবহার: Most common, সব hidden layers এ")
    relu_out = F.relu(test_values)
    print(f"   Output: {relu_out}")
    print(f"   ✓ Negative → 0, Positive → same")
    
    print("\n2️⃣ Sigmoid")
    print("   Formula: f(x) = 1 / (1 + e^(-x))")
    print("   Range: 0 to 1")
    print("   ব্যবহার: Binary classification output, probability")
    sigmoid_out = torch.sigmoid(test_values)
    print(f"   Output: {sigmoid_out}")
    print(f"   ✓ সব values 0 এবং 1 এর মধ্যে")
    
    print("\n3️⃣ Tanh (Hyperbolic Tangent)")
    print("   Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))")
    print("   Range: -1 to 1")
    print("   ব্যবহার: Hidden layers, RNN")
    tanh_out = torch.tanh(test_values)
    print(f"   Output: {tanh_out}")
    print(f"   ✓ -1 থেকে +1 এর মধ্যে")
    
    print("\n4️⃣ Softmax (Multi-class classification)")
    print("   সব outputs এর sum = 1 (probability distribution)")
    logits = torch.tensor([2.0, 1.0, 0.1])
    softmax_out = F.softmax(logits, dim=0)
    print(f"   Input (logits): {logits}")
    print(f"   Output (probabilities): {softmax_out}")
    print(f"   Sum: {softmax_out.sum()}")
    
    print("\n💡 কোনটা কখন ব্যবহার করবে?")
    print("   • Hidden Layers → ReLU (fast, efficient)")
    print("   • Binary Output → Sigmoid (0 to 1)")
    print("   • Multi-class Output → Softmax (probabilities)")
    print("   • RNN/LSTM → Tanh (-1 to 1)")
    
    print("\n✓ Section 5 Complete!\n")


# =============================================================================
# SECTION 6: LOSS FUNCTIONS
# =============================================================================
def section_6_loss_functions():
    """
    Different loss functions এবং কখন ব্যবহার করবে
    """
    print("\n" + "="*70)
    print("SECTION 6: Loss Functions")
    print("="*70)
    
    print("\n1️⃣ nn.CrossEntropyLoss - Multi-class Classification")
    print("   ব্যবহার: MNIST (0-9), ImageNet (1000 classes)")
    criterion = nn.CrossEntropyLoss()
    # Predictions: 3 samples, 4 classes
    predictions = torch.randn(3, 4)
    targets = torch.tensor([0, 2, 1])  # Correct classes
    loss = criterion(predictions, targets)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Targets: {targets}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✓ Lower is better!")
    
    print("\n2️⃣ nn.MSELoss - Regression (Mean Squared Error)")
    print("   ব্যবহার: House price prediction, temperature forecasting")
    mse_criterion = nn.MSELoss()
    predictions = torch.tensor([2.5, 0.0, 2.1])
    targets = torch.tensor([3.0, -0.5, 2.0])
    mse_loss = mse_criterion(predictions, targets)
    print(f"   Predictions: {predictions}")
    print(f"   Targets: {targets}")
    print(f"   MSE Loss: {mse_loss.item():.4f}")
    print(f"   Formula: mean((pred - target)²)")
    
    print("\n3️⃣ nn.L1Loss - Mean Absolute Error")
    print("   ব্যবহার: Robust regression (outliers এর বিরুদ্ধে)")
    l1_criterion = nn.L1Loss()
    l1_loss = l1_criterion(predictions, targets)
    print(f"   Predictions: {predictions}")
    print(f"   Targets: {targets}")
    print(f"   L1 Loss: {l1_loss.item():.4f}")
    print(f"   Formula: mean(|pred - target|)")
    
    print("\n4️⃣ nn.BCELoss - Binary Cross Entropy")
    print("   ব্যবহার: Multi-label classification, Autoencoders")
    bce_criterion = nn.BCELoss()
    # Predictions must be between 0-1 (use sigmoid first!)
    predictions = torch.sigmoid(torch.randn(3, 4))
    targets = torch.randint(0, 2, (3, 4)).float()
    bce_loss = bce_criterion(predictions, targets)
    print(f"   Predictions (after sigmoid): {predictions[0]}")
    print(f"   Targets: {targets[0]}")
    print(f"   BCE Loss: {bce_loss.item():.4f}")
    
    print("\n💡 কোন Loss Function কখন?")
    print("   • Multi-class Classification → CrossEntropyLoss")
    print("   • Regression (continuous values) → MSELoss বা L1Loss")
    print("   • Binary Classification → BCELoss")
    print("   • Multi-label Classification → BCELoss")
    
    print("\n✓ Section 6 Complete!\n")


# =============================================================================
# SECTION 7: OPTIMIZERS
# =============================================================================
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

def section_7_optimizers():
    """
    Different optimizers এবং তাদের differences
    """
    print("\n" + "="*70)
    print("SECTION 7: Optimizers")
    print("="*70)
    
    # Simple model for demonstration
    model = nn.Linear(5, 1)
    
    print("\n1️⃣ optim.SGD - Stochastic Gradient Descent")
    print("   সবচেয়ে basic optimizer")
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print(f"   Learning rate: 0.01")
    print(f"   Momentum: 0.9 (previous gradients এর effect রাখে)")
    print(f"   ✓ Simple, stable, কিন্তু slow convergence")
    
    print("\n2️⃣ optim.Adam - Adaptive Moment Estimation")
    print("   সবচেয়ে popular optimizer")
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    print(f"   Learning rate: 0.001")
    print(f"   ✓ Adaptive learning rate প্রতিটা parameter এর জন্য")
    print(f"   ✓ Fast convergence, most tasks এর জন্য best")
    
    print("\n3️⃣ optim.Adagrad - Adaptive Gradient")
    print("   Sparse data এর জন্য ভালো")
    optimizer_adagrad = optim.Adagrad(model.parameters(), lr=0.01)
    print(f"   ✓ Frequently updated params → smaller learning rate")
    print(f"   ✓ Rare params → larger learning rate")
    
    print("\n4️⃣ optim.RMSprop - Root Mean Square Propagation")
    print("   RNN এর জন্য ভালো")
    optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01)
    print(f"   ✓ Adagrad এর improved version")
    print(f"   ✓ Divides learning rate by exponential average of gradients")
    
    print("\n📊 Comparison:")
    print("-" * 70)
    print("Optimizer  | Speed    | Memory | Use Case")
    print("-" * 70)
    print("SGD        | Slow     | Low    | When you have lots of time")
    print("Adam       | Fast     | High   | ⭐ Default choice, most tasks")
    print("Adagrad    | Medium   | Medium | Sparse features, NLP")
    print("RMSprop    | Medium   | Medium | RNNs, non-stationary problems")
    print("-" * 70)
    
    print("\n💡 কোনটা ব্যবহার করবে?")
    print("   99% cases এ Adam ব্যবহার করো!")
    print("   Research paper reproduce করলে তাদের optimizer ব্যবহার করো")
    
    print("\n✓ Section 7 Complete!\n")


# =============================================================================
# SECTION 8: DATA LOADING
# =============================================================================
def section_8_data_loading():
    """
    Dataset এবং DataLoader ব্যবহার করা
    """
    print("\n" + "="*70)
    print("SECTION 8: Data Loading")
    print("="*70)
    
    print("\n1️⃣ Custom Dataset তৈরি করা")
    print("-" * 70)
    
    class MyDataset(Dataset):
        def __init__(self, num_samples=100):
            # Data তৈরি করো
            self.data = torch.randn(num_samples, 5)
            self.labels = torch.randint(0, 2, (num_samples,))
        
        def __len__(self):
            # Dataset এর size return করো
            return len(self.data)
        
        def __getitem__(self, idx):
            # একটা sample return করো
            return self.data[idx], self.labels[idx]
    
    dataset = MyDataset(100)
    print(f"   Dataset size: {len(dataset)}")
    sample_data, sample_label = dataset[0]
    print(f"   Sample data shape: {sample_data.shape}")
    print(f"   Sample label: {sample_label.item()}")
    
    print("\n2️⃣ DataLoader - Batching এবং Shuffling")
    print("-" * 70)
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    print(f"   Batch size: 16")
    print(f"   Shuffle: True")
    print(f"   Total batches: {len(dataloader)}")
    
    # একটা batch দেখো
    for batch_data, batch_labels in dataloader:
        print(f"\n   First batch:")
        print(f"   Data shape: {batch_data.shape}")
        print(f"   Labels shape: {batch_labels.shape}")
        break
    
    print("\n3️⃣ Train/Test Split")
    print("-" * 70)
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% training
    test_size = total_size - train_size  # 20% testing
    
    from torch.utils.data import random_split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"   Total samples: {total_size}")
    print(f"   Training samples: {len(train_dataset)} (80%)")
    print(f"   Testing samples: {len(test_dataset)} (20%)")
    
    print("\n4️⃣ TensorDataset - সহজ alternative")
    print("-" * 70)
    
    X = torch.randn(50, 3)
    y = torch.randint(0, 2, (50,))
    
    tensor_dataset = TensorDataset(X, y)
    tensor_loader = DataLoader(tensor_dataset, batch_size=10)
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   TensorDataset created with {len(tensor_dataset)} samples")
    
    print("\n✓ Section 8 Complete!\n")


# =============================================================================
# SECTION 9: TRAINING এবং EVALUATION
# =============================================================================
def section_9_training_evaluation():
    """
    Complete training loop এবং evaluation
    """
    print("\n" + "="*70)
    print("SECTION 9: Training & Evaluation")
    print("="*70)
    
    # Dummy data
    X_train = torch.randn(80, 5)
    y_train = torch.randint(0, 3, (80,))
    X_test = torch.randn(20, 5)
    y_test = torch.randint(0, 3, (20,))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)
    
    # Model
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 3)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("\n1️⃣ Training Loop")
    print("-" * 70)
    
    print("\n# Training loop structure:")
    print("for epoch in range(num_epochs):")
    print("    model.train()  # Training mode")
    print("    for inputs, labels in train_loader:")
    print("        optimizer.zero_grad()    # Step 1: Clear gradients")
    print("        outputs = model(inputs)  # Step 2: Forward pass")
    print("        loss = criterion(outputs, labels)  # Step 3: Calculate loss")
    print("        loss.backward()          # Step 4: Backward pass")
    print("        optimizer.step()         # Step 5: Update weights")
    
    # Actual training
    print("\nActual Training:")
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\n2️⃣ Evaluation Loop")
    print("-" * 70)
    
    print("\n# Evaluation structure:")
    print("model.eval()  # Evaluation mode")
    print("with torch.no_grad():  # No gradient calculation")
    print("    for inputs, labels in test_loader:")
    print("        outputs = model(inputs)")
    print("        # Calculate accuracy, loss, etc.")
    
    # Actual evaluation
    print("\nActual Evaluation:")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"   Test Accuracy: {accuracy:.2f}%")
    
    print("\n💡 Key Differences:")
    print("   Training:")
    print("   • model.train() → Dropout, BatchNorm active")
    print("   • Gradients calculated")
    print("   • Weights updated")
    print("")
    print("   Evaluation:")
    print("   • model.eval() → Dropout off, BatchNorm frozen")
    print("   • torch.no_grad() → Save memory")
    print("   • Weights NOT updated")
    
    print("\n✓ Section 9 Complete!\n")


# =============================================================================
# SECTION 10: GPU TRAINING এবং SAVE/LOAD
# =============================================================================
def section_10_gpu_and_saving():
    """
    GPU ব্যবহার এবং model save/load করা
    """
    print("\n" + "="*70)
    print("SECTION 10: GPU Training & Save/Load Models")
    print("="*70)
    
    print("\n1️⃣ GPU Setup")
    print("-" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    print(f"   Current device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ GPU available!")
    else:
        print(f"   ✓ GPU not available, using CPU")
    
    print("\n2️⃣ Moving Model and Data to GPU")
    print("-" * 70)
    
    model = nn.Sequential(nn.Linear(5, 3))
    model = model.to(device)  # Model কে GPU তে পাঠাও
    print(f"   model = model.to(device)")
    print(f"   ✓ Model is now on {device}")
    
    # Data also needs to be on GPU
    print("\n   # Training loop এ:")
    print("   for inputs, labels in dataloader:")
    print("       inputs = inputs.to(device)")
    print("       labels = labels.to(device)")
    print("       # ... rest of training ...")
    
    print("\n3️⃣ Save Model")
    print("-" * 70)
    
    print("   # Only save parameters (recommended):")
    print("   torch.save(model.state_dict(), 'model.pth')")
    print("   ✓ Saves only weights, not architecture")
    print("   ✓ Small file size")
    
    print("\n   # Save entire model:")
    print("   torch.save(model, 'full_model.pth')")
    print("   ✓ Saves everything")
    print("   ✓ Larger file size")
    
    # Actually save
    torch.save(model.state_dict(), 'temp_model.pth')
    print("\n   ✓ Model saved to 'temp_model.pth'")
    
    print("\n4️⃣ Load Model")
    print("-" * 70)
    
    print("   # Load parameters:")
    print("   model = nn.Sequential(nn.Linear(5, 3))")
    print("   model.load_state_dict(torch.load('model.pth'))")
    print("   model.eval()  # Set to evaluation mode")
    
    # Actually load
    new_model = nn.Sequential(nn.Linear(5, 3))
    new_model.load_state_dict(torch.load('temp_model.pth'))
    new_model.eval()
    print("\n   ✓ Model loaded successfully!")
    
    print("\n   # Load entire model:")
    print("   model = torch.load('full_model.pth')")
    print("   model.eval()")
    
    print("\n💡 Best Practices:")
    print("   • সবসময় state_dict() save করো (flexible)")
    print("   • Load করার পর model.eval() call করো")
    print("   • GPU তে train করলে CPU তে load করার আগে:")
    print("     model.load_state_dict(torch.load('model.pth', map_location='cpu'))")
    
    print("\n✓ Section 10 Complete!\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

sections = {
    1: ("Basic Imports", section_1_imports),
    2: ("Tensor Operations", section_2_tensors),
    3: ("Model Definition", section_3_model_definition),
    4: ("Layers", section_4_layers),
    5: ("Activation Functions", section_5_activations),
    6: ("Loss Functions", section_6_loss_functions),
    7: ("Optimizers", section_7_optimizers),
    8: ("Data Loading", section_8_data_loading),
    9: ("Training & Evaluation", section_9_training_evaluation),
    10: ("GPU & Save/Load", section_10_gpu_and_saving)
}

if CURRENT_SECTION in sections:
    name, func = sections[CURRENT_SECTION]
    print(f"\n▶ Running Section {CURRENT_SECTION}: {name}")
    func()
else:
    print(f"\n❌ Invalid section! Choose 1-10")
    print("\nAvailable Sections:")
    for num, (name, _) in sections.items():
        print(f"  {num}. {name}")

print("\n" + "="*70)
print("📚 সব sections দেখতে CURRENT_SECTION পরিবর্তন করো (1-10)")
print("="*70)