"""
Advanced PyTorch Problems - For Intellectual Assessment
========================================================
These problems test deep understanding, not just syntax memorization.

Topics covered:
1. Gradient Flow & Backpropagation Understanding
2. Custom Loss Functions & Optimization
3. Architecture Design Challenges
4. Batch Normalization vs Layer Normalization
5. Advanced Training Techniques
6. Memory & Computational Efficiency
7. Debugging & Analysis

Change CURRENT_PROBLEM (1-10) to run different challenges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

CURRENT_PROBLEM = 1

# ==============================================================================
# PROBLEM 1: Understanding Gradient Flow - Vanishing/Exploding Gradients
# ==============================================================================
"""
Challenge: Implement a deep network and demonstrate the vanishing gradient problem.
Then fix it using proper initialization and activation functions.

Intellectual Points:
- Why do gradients vanish in deep networks?
- How does activation function choice affect gradient flow?
- What is the role of weight initialization?
"""

def problem_1_vanishing_gradients():
    print("="*80)
    print("PROBLEM 1: Gradient Flow Analysis - Vanishing Gradient Problem")
    print("="*80)
    
    print("\n📊 LR Schedule Comparison:")
    print(f"{'Epoch':<10} {'StepLR':<12} {'Cosine':<12} {'Warmup+Cosine':<15}")
    print("-" * 50)
    for epoch in [0, 10, 30, 50, 70, 99]:
        print(f"{epoch:<10} {lrs_step[epoch]:<12.4f} {lrs_cosine[epoch]:<12.4f} {lrs_warmup[epoch]:<15.4f}")
    
    print("\n💡 Best Practices:")
    print("1. Use warmup for Transformers, large models")
    print("2. Cosine annealing for smooth convergence")
    print("3. StepLR when you know when to reduce LR")
    print("4. ReduceLROnPlateau: adaptive based on validation loss")
    
    print("\n🎓 Deep Question:")
    print("Q: Why does warmup help training?")
    print("\nA: At initialization:")
    print("  - Weights are random")
    print("  - Large LR + random weights = unstable gradients")
    print("  - Warmup allows network to 'stabilize' first")
    print("  - Then aggressive LR for fast learning")
    
    print("\n✓ Problem 5 Complete!\n")


# ==============================================================================
# PROBLEM 6: Weight Decay vs L2 Regularization (NOT the same!)
# ==============================================================================
"""
Challenge: Many people think weight decay = L2 regularization
They're different in Adam! Understand the subtle difference.
"""

def problem_6_weight_decay():
    print("="*80)
    print("PROBLEM 6: Weight Decay vs L2 Regularization")
    print("="*80)
    
    print("\n🎯 Challenge:")
    print("Understand the difference between weight decay and L2 regularization")
    print("They're the same in SGD, but DIFFERENT in Adam!")
    
    print("\n📚 Background:")
    print("Common misconception: Weight decay = L2 regularization")
    print("Truth: Only equivalent in SGD without momentum!")
    
    print("\n" + "-"*80)
    print("Theory: Weight Decay vs L2 Regularization")
    print("-"*80)
    
    print("\nL2 Regularization:")
    print("  Loss = Original_Loss + λ * ||w||²")
    print("  Gradient: ∇L = ∇Original_Loss + 2λw")
    print("  Update: w ← w - lr * (∇Original_Loss + 2λw)")
    
    print("\nWeight Decay:")
    print("  Update: w ← w - lr * ∇Original_Loss - lr * λ * w")
    print("          w ← (1 - lr*λ) * w - lr * ∇Original_Loss")
    
    print("\nFor SGD (no momentum):")
    print("  Both are equivalent! ✓")
    
    print("\nFor Adam (with adaptive LR):")
    print("  L2: Adds to gradient → adaptive LR applied")
    print("  Weight Decay: Directly scales weights → NOT adaptive")
    print("  Result: DIFFERENT behaviors! ✗")
    
    # Demonstration
    model_l2 = nn.Linear(10, 10)
    model_wd = nn.Linear(10, 10)
    
    # Copy weights to make them identical
    model_wd.load_state_dict(model_l2.state_dict())
    
    print("\n" + "-"*80)
    print("Experiment: Train with L2 vs Weight Decay")
    print("-"*80)
    
    # L2 regularization (manual)
    optimizer_l2 = optim.Adam(model_l2.parameters(), lr=0.01)
    
    # Weight decay (built-in)
    optimizer_wd = optim.AdamW(model_wd.parameters(), lr=0.01, weight_decay=0.01)
    
    # Generate dummy data
    X = torch.randn(100, 10)
    y = torch.randn(100, 10)
    
    # Training loop
    num_steps = 50
    print(f"\nTraining for {num_steps} steps...")
    
    for step in range(num_steps):
        # L2 Regularization
        optimizer_l2.zero_grad()
        output_l2 = model_l2(X)
        loss_l2 = F.mse_loss(output_l2, y)
        
        # Add L2 penalty manually
        l2_penalty = 0
        for param in model_l2.parameters():
            l2_penalty += torch.norm(param, 2) ** 2
        loss_l2_total = loss_l2 + 0.01 * l2_penalty
        
        loss_l2_total.backward()
        optimizer_l2.step()
        
        # Weight Decay (AdamW)
        optimizer_wd.zero_grad()
        output_wd = model_wd(X)
        loss_wd = F.mse_loss(output_wd, y)
        loss_wd.backward()
        optimizer_wd.step()
    
    # Compare final weights
    l2_weight_norm = torch.norm(model_l2.weight).item()
    wd_weight_norm = torch.norm(model_wd.weight).item()
    
    print(f"\nFinal weight norms:")
    print(f"  L2 Regularization: {l2_weight_norm:.4f}")
    print(f"  Weight Decay (AdamW): {wd_weight_norm:.4f}")
    print(f"  Difference: {abs(l2_weight_norm - wd_weight_norm):.4f}")
    
    print("\n💡 Key Insights:")
    print("1. For SGD: weight_decay = L2 regularization")
    print("2. For Adam: Use AdamW (weight decay), NOT L2!")
    print("3. AdamW decouples weight decay from gradient")
    print("4. AdamW generally performs better than Adam + L2")
    
    print("\n🎓 Why does this matter?")
    print("Papers that use 'Adam with L2 regularization' are technically wrong!")
    print("Should use AdamW for proper weight decay in Adam.")
    print("Reference: 'Decoupled Weight Decay Regularization' (Loshchilov & Hutter, 2019)")
    
    print("\n✓ Problem 6 Complete!\n")


# ==============================================================================
# PROBLEM 7: Memory Efficiency - Gradient Checkpointing
# ==============================================================================
"""
Challenge: Understand memory bottlenecks in deep learning
Implement gradient checkpointing concept
"""

def problem_7_memory_efficiency():
    print("="*80)
    print("PROBLEM 7: Memory Efficiency - Gradient Checkpointing")
    print("="*80)
    
    print("\n🎯 Challenge:")
    print("Understand memory usage in backpropagation")
    print("Learn trade-off between memory and computation")
    
    print("\n📚 Background:")
    print("During training, PyTorch stores ALL intermediate activations")
    print("Why? Needed for backward pass (chain rule)")
    print("Problem: Deep networks → huge memory usage!")
    
    print("\n" + "-"*80)
    print("Memory Analysis")
    print("-"*80)
    
    class DeepNetwork(nn.Module):
        def __init__(self, n_layers=100):
            super(DeepNetwork, self).__init__()
            self.layers = nn.ModuleList([
                nn.Linear(1000, 1000) for _ in range(n_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))
            return x
    
    print("\nConsider a network with 100 layers, input size 1000:")
    print("  Each layer output: 1000 floats = 4KB")
    print("  100 layers = 400KB per sample")
    print("  Batch size 32 = 12.8MB just for activations!")
    print("  Plus gradients = 25.6MB total")
    
    print("\n" + "-"*80)
    print("Solution: Gradient Checkpointing")
    print("-"*80)
    
    print("\nIdea: Don't store ALL activations")
    print("  1. Store only SOME activations (checkpoints)")
    print("  2. During backward, recompute others on-the-fly")
    print("  3. Trade: Less memory ↔ More computation")
    
    print("\nExample with 4 layers:")
    print("  Normal: Store [A1, A2, A3, A4]")
    print("  Checkpointing: Store [A1, A3] only")
    print("  During backward:")
    print("    - Need A2? Recompute from A1")
    print("    - Need A4? Recompute from A3")
    
    print("\n" + "-"*80)
    print("Memory Savings Calculation")
    print("-"*80)
    
    def calculate_memory(n_layers, checkpoint_every_k):
        """
        Calculate memory saved by checkpointing
        """
        # Normal: store all n_layers
        normal_memory = n_layers
        
        # Checkpointing: store every k-th layer
        checkpointed_memory = (n_layers // checkpoint_every_k) + 1
        
        savings = (1 - checkpointed_memory / normal_memory) * 100
        return normal_memory, checkpointed_memory, savings
    
    print(f"\n{'Layers':<10} {'Checkpoint Every':<18} {'Normal':<10} {'Checkpointed':<15} {'Savings':<10}")
    print("-" * 70)
    
    for n_layers in [50, 100, 200]:
        for k in [2, 5, 10]:
            normal, checkpointed, savings = calculate_memory(n_layers, k)
            print(f"{n_layers:<10} {k:<18} {normal:<10} {checkpointed:<15} {savings:.1f}%")
    
    print("\n💡 Real Implementation (PyTorch):")
    print("```python")
    print("from torch.utils.checkpoint import checkpoint")
    print("")
    print("def forward(self, x):")
    print("    x = self.layer1(x)")
    print("    x = checkpoint(self.layer2, x)  # Checkpointed!")
    print("    x = self.layer3(x)")
    print("    return x")
    print("```")
    
    print("\n🎓 When to use?")
    print("✓ Very deep networks (ResNet-152, Transformers)")
    print("✓ Limited GPU memory")
    print("✓ Large batch sizes needed")
    print("✗ Training time is critical (20-30% slower)")
    print("✗ Shallow networks (overhead not worth it)")
    
    print("\n✓ Problem 7 Complete!\n")


# ==============================================================================
# PROBLEM 8: Debugging Neural Networks - Systematic Approach
# ==============================================================================
"""
Challenge: Network not training? Apply systematic debugging
This is a CRITICAL skill that separates experts from beginners
"""

def problem_8_debugging():
    print("="*80)
    print("PROBLEM 8: Debugging Neural Networks - Systematic Approach")
    print("="*80)
    
    print("\n🎯 Challenge:")
    print("You have a network that's not learning (loss not decreasing)")
    print("Apply systematic debugging to find the problem")
    
    print("\n📚 The Debugging Checklist:")
    print("-"*80)
    
    debugging_steps = [
        ("1. Overfit a single batch", 
         "If you can't overfit 1 batch, there's a fundamental issue"),
        ("2. Check data pipeline",
         "Visualize inputs and labels - are they correct?"),
        ("3. Verify loss function",
         "Is it appropriate for your task?"),
        ("4. Check learning rate",
         "Too high → divergence, Too low → no learning"),
        ("5. Inspect gradients",
         "NaN? Zero? Exploding?"),
        ("6. Verify architecture",
         "Are shapes compatible? Any dimension mismatches?"),
        ("7. Check initialization",
         "Bad init → dead neurons or exploding activations"),
        ("8. Monitor activation statistics",
         "Mean, std, % of dead ReLUs"),
    ]
    
    for step, explanation in debugging_steps:
        print(f"\n{step}")
        print(f"  → {explanation}")
    
    print("\n" + "-"*80)
    print("Practical Example: Debugging a Broken Network")
    print("-"*80)
    
    # Intentionally broken network
    class BrokenNetwork(nn.Module):
        def __init__(self):
            super(BrokenNetwork, self).__init__()
            self.fc1 = nn.Linear(10, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 2)
            
            # BAD INITIALIZATION (too large)
            nn.init.normal_(self.fc1.weight, mean=0, std=10.0)
            nn.init.normal_(self.fc2.weight, mean=0, std=10.0)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = BrokenNetwork()
    X = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n🔍 Step 1: Try to overfit a single batch")
    print("-" * 70)
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print("\n❌ Loss not decreasing much! Let's debug...")
    
    print("\n🔍 Step 2: Check gradient magnitudes")
    print("-" * 70)
    
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name:<20} Gradient norm: {grad_norm:.2e}")
    
    print("\n⚠️  Gradients seem okay...")
    
    print("\n🔍 Step 3: Check activation statistics")
    print("-" * 70)
    
    # Hook to capture activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    model.fc1.register_forward_hook(get_activation('fc1'))
    model.fc2.register_forward_hook(get_activation('fc2'))
    
    with torch.no_grad():
        _ = model(X)
    
    for name, activation in activations.items():
        mean = activation.mean().item()
        std = activation.std().item()
        dead_relu = (activation == 0).float().mean().item() * 100
        
        print(f"\n{name}:")
        print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
        print(f"  Dead ReLUs: {dead_relu:.1f}%")
    
    print("\n🔎 Found it! High std → bad initialization!")
    
    print("\n" + "-"*80)
    print("Fix: Proper Initialization")
    print("-"*80)
    
    class FixedNetwork(nn.Module):
        def __init__(self):
            super(FixedNetwork, self).__init__()
            self.fc1 = nn.Linear(10, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 2)
            
            # GOOD INITIALIZATION
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    fixed_model = FixedNetwork()
    optimizer = optim.Adam(fixed_model.parameters(), lr=0.001)
    
    print("\nTraining fixed model:")
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = fixed_model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print("\n✓ Loss decreasing! Problem solved!")
    
    print("\n💡 Common Issues & Solutions:")
    print("-"*80)
    issues = [
        ("Loss is NaN", "→ Learning rate too high OR bad initialization"),
        ("Loss not decreasing", "→ Check LR, initialization, data pipeline"),
        ("Loss decreasing but slow", "→ Increase LR or use better optimizer"),
        ("Train acc high, test acc low", "→ Overfitting! Add regularization"),
        ("Both train & test acc low", "→ Underfitting! Increase capacity"),
    ]
    
    for issue, solution in issues:
        print(f"\n{issue}")
        print(f"  {solution}")
    
    print("\n✓ Problem 8 Complete!\n")


# ==============================================================================
# PROBLEM 9: Mixup Data Augmentation - Advanced Regularization
# ==============================================================================
"""
Challenge: Implement Mixup augmentation
Understand how mixing training examples improves generalization
"""

def problem_9_mixup():
    print("="*80)
    print("PROBLEM 9: Mixup Data Augmentation")
    print("="*80)
    
    print("\n🎯 Challenge:")
    print("Implement Mixup - a powerful data augmentation technique")
    print("Creates virtual training examples by mixing pairs")
    
    print("\n📚 Background:")
    print("Mixup: Mix two training examples")
    print("  x_mixed = λ * x_i + (1-λ) * x_j")
    print("  y_mixed = λ * y_i + (1-λ) * y_j")
    print("  λ ~ Beta(α, α), typically α=0.2 or α=1.0")
    
    def mixup_data(x, y, alpha=1.0):
        """
        YOUR TASK: Implement mixup
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """Loss function for mixup"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    print("\n" + "-"*80)
    print("Demonstration: Mixing Two Samples")
    print("-"*80)
    
    # Create two simple samples
    x1 = torch.ones(1, 3) * 1.0  # Sample 1
    x2 = torch.ones(1, 3) * 5.0  # Sample 2
    y1 = torch.tensor([0])  # Label: class 0
    y2 = torch.tensor([1])  # Label: class 1
    
    print(f"\nOriginal samples:")
    print(f"  x1 = {x1[0].tolist()}, y1 = {y1.item()}")
    print(f"  x2 = {x2[0].tolist()}, y2 = {y2.item()}")
    
    # Apply mixup
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
    
    print(f"\nAfter mixup (λ={lam:.3f}):")
    print(f"  mixed_x[0] = {mixed_x[0].tolist()}")
    print(f"  Mixed with: {lam:.3f} * sample_0 + {1-lam:.3f} * sample_1")
    
    print("\n" + "-"*80)
    print("Training with Mixup")
    print("-"*80)
    
    # Create toy dataset
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 3, (100,))
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("\nTraining with Mixup (alpha=1.0):")
    batch_size = 32
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Apply mixup
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=1.0)
            
            optimizer.zero_grad()
            outputs = model(mixed_x)
            
            # Mixup loss
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(X_train) // batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\n💡 Why does Mixup work?")
    print("1. Encourages linear behavior between training examples")
    print("2. Reduces overfitting to specific training samples")
    print("3. Network learns smoother decision boundaries")
    print("4. Improves calibration (confidence estimates)")
    
    print("\n🎓 Advanced Insight:")
    print("Mixup is a form of vicinal risk minimization:")
    print("  Traditional: Minimize risk at training points")
    print("  Mixup: Minimize risk in NEIGHBORHOOD of training points")
    print("  Result: Better generalization!")
    
    print("\n📊 Typical Results:")
    print("  Image Classification: +1-2% accuracy improvement")
    print("  Works best with: α=0.2 for images, α=1.0 for tabular")
    
    print("\n✓ Problem 9 Complete!\n")


# ==============================================================================
# PROBLEM 10: Multi-Task Learning - Shared Representations
# ==============================================================================
"""
Challenge: Build a multi-task network
Learn how to balance multiple losses
"""

def problem_10_multi_task():
    print("="*80)
    print("PROBLEM 10: Multi-Task Learning")
    print("="*80)
    
    print("\n🎯 Challenge:")
    print("Build a network that solves TWO tasks simultaneously")
    print("Task 1: Classification (3 classes)")
    print("Task 2: Regression (1 continuous value)")
    
    print("\n📚 Background:")
    print("Multi-task learning: Shared representation for multiple tasks")
    print("Benefits: Better generalization, reduced overfitting")
    print("Challenge: Balancing losses from different tasks")
    
    class MultiTaskNetwork(nn.Module):
        """
        YOUR TASK: Complete multi-task architecture
        """
        def __init__(self):
            super(MultiTaskNetwork, self).__init__()
            
            # Shared layers
            self.shared = nn.Sequential(
                nn.Linear(10, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU()
            )
            
            # Task 1: Classification head
            self.classifier = nn.Sequential(
                nn.Linear(50, 20),
                nn.ReLU(),
                nn.Linear(20, 3)  # 3 classes
            )
            
            # Task 2: Regression head
            self.regressor = nn.Sequential(
                nn.Linear(50, 20),
                nn.ReLU(),
                nn.Linear(20, 1)  # 1 continuous output
            )
        
        def forward(self, x):
            # Shared representation
            shared_features = self.shared(x)
            
            # Task-specific outputs
            class_output = self.classifier(shared_features)
            regr_output = self.regressor(shared_features)
            
            return class_output, regr_output
    
    print("\n" + "-"*80)
    print("Network Architecture")
    print("-"*80)
    
    model = MultiTaskNetwork()
    print(model)
    
    print("\n" + "-"*80)
    print("Generate Multi-Task Data")
    print("-"*80)
    
    # Generate synthetic data
    n_samples = 200
    X = torch.randn(n_samples, 10)
    
    # Task 1: Classification labels
    y_class = torch.randint(0, 3, (n_samples,))
    
    # Task 2: Regression targets (correlated with class)
    y_regr = y_class.float() + torch.randn(n_samples) * 0.5
    y_regr = y_regr.unsqueeze(1)
    
    print(f"Data shapes:")
    print(f"  X: {X.shape}")
    print(f"  y_class: {y_class.shape}")
    print(f"  y_regr: {y_regr.shape}")
    
    # Split train/test
    train_size = 150
    X_train, X_test = X[:train_size], X[train_size:]
    y_class_train, y_class_test = y_class[:train_size], y_class[train_size:]
    y_regr_train, y_regr_test = y_regr[:train_size], y_regr[train_size:]
    
    print("\n" + "-"*80)
    print("Training Multi-Task Network")
    print("-"*80)
    
    # Loss functions for each task
    criterion_class = nn.CrossEntropyLoss()
    criterion_regr = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Loss weights (important for balancing!)
    weight_class = 1.0
    weight_regr = 0.5
    
    print(f"\nLoss weights:")
    print(f"  Classification: {weight_class}")
    print(f"  Regression: {weight_regr}")
    
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        
        optimizer.zero_grad()
        
        # Forward pass
        class_out, regr_out = model(X_train)
        
        # Calculate both losses
        loss_class = criterion_class(class_out, y_class_train)
        loss_regr = criterion_regr(regr_out, y_regr_train)
        
        # Combined loss
        total_loss = weight_class * loss_class + weight_regr * loss_regr
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Class Loss: {loss_class.item():.4f}")
            print(f"  Regr Loss: {loss_regr.item():.4f}")
            print(f"  Total Loss: {total_loss.item():.4f}")
    
    print("\n" + "-"*80)
    print("Evaluation on Test Set")
    print("-"*80)
    
    model.eval()
    with torch.no_grad():
        class_out, regr_out = model(X_test)
        
        # Classification accuracy
        _, predicted = class_out.max(1)
        accuracy = (predicted == y_class_test).float().mean() * 100
        
        # Regression error
        mse = F.mse_loss(regr_out, y_regr_test)
        
        print(f"\nTest Results:")
        print(f"  Classification Accuracy: {accuracy:.2f}%")
        print(f"  Regression MSE: {mse.item():.4f}")
    
    print("\n💡 Key Challenges in Multi-Task Learning:")
    print("1. Loss Balancing: Different scales → one task dominates")
    print("2. Conflicting Gradients: Tasks may want different updates")
    print("3. Task Weighting: How to set α₁, α₂, ...?")
    
    print("\n🎓 Advanced Solutions:")
    print("1. Uncertainty Weighting (Kendall et al.)")
    print("   Learn task weights automatically")
    print("2. Gradient Normalization (GradNorm)")
    print("   Balance gradient magnitudes across tasks")
    print("3. Dynamic Weight Average (DWA)")
    print("   Adjust weights based on task learning speed")
    
    print("\n📊 When to use Multi-Task Learning?")
    print("✓ Tasks are related (share common features)")
    print("✓ Limited data for each task individually")
    print("✓ Want to reduce model size (shared parameters)")
    print("✗ Tasks are unrelated or conflicting")
    print("✗ Need maximum performance on single task")
    
    print("\n✓ Problem 10 Complete!\n")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

problems = {
    1: ("Vanishing Gradients", problem_1_vanishing_gradients),
    2: ("Custom Loss - Focal Loss", problem_2_custom_loss),
    3: ("Residual Networks", problem_3_residual_networks),
    4: ("BatchNorm vs LayerNorm", problem_4_normalization_comparison),
    5: ("LR Scheduling & Warmup", problem_5_lr_scheduling),
    6: ("Weight Decay vs L2", problem_6_weight_decay),
    7: ("Memory Efficiency", problem_7_memory_efficiency),
    8: ("Debugging Networks", problem_8_debugging),
    9: ("Mixup Augmentation", problem_9_mixup),
    10: ("Multi-Task Learning", problem_10_multi_task)
}

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ADVANCED PYTORCH PROBLEMS - INTELLECTUAL CHALLENGES")
    print("="*80)
    print("\nAvailable Problems:")
    print("-"*80)
    for num, (name, _) in problems.items():
        print(f"  {num:2d}. {name}")
    print("="*80)
    print(f"\n▶ Running Problem {CURRENT_PROBLEM}...\n")
    
    if CURRENT_PROBLEM in problems:
        name, func = problems[CURRENT_PROBLEM]
        func()
    else:
        print("❌ Invalid problem number! Set CURRENT_PROBLEM to 1-10")
    
    print("\n" + "="*80)
    print("💡 TIPS FOR YOUR EXAM:")
    print("="*80)
    print("1. These problems test UNDERSTANDING, not memorization")
    print("2. Focus on WHY things work, not just HOW")
    print("3. Be able to explain trade-offs and design choices")
    print("4. Practice debugging - it's a critical skill")
    print("5. Understand when to use which technique")
    print("="*80)
    print("\n🎓 Good luck with your intellectual teacher!")
    print("📚 Change CURRENT_PROBLEM (line 25) to practice different problems\n")🎯 Challenge:")
    print("Build a 10-layer deep network. Analyze gradient magnitudes at each layer.")
    print("Show how gradients vanish with bad initialization/activation.")
    print("Then fix it with proper techniques.")
    
    print("\n" + "-"*80)
    print("Part A: Bad Network (Vanishing Gradients)")
    print("-"*80)
    
    class BadDeepNetwork(nn.Module):
        """Network prone to vanishing gradients"""
        def __init__(self):
            super(BadDeepNetwork, self).__init__()
            layers = []
            # 10 layers with sigmoid activation (prone to vanishing gradients)
            for i in range(10):
                layers.append(nn.Linear(100, 100))
                layers.append(nn.Sigmoid())  # Sigmoid saturates → small gradients
            layers.append(nn.Linear(100, 10))
            self.network = nn.Sequential(*layers)
            
            # Bad initialization: standard normal
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1.0)  # Too large!
        
        def forward(self, x):
            return self.network(x)
    
    # Test bad network
    bad_model = BadDeepNetwork()
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(bad_model.parameters(), lr=0.01)
    
    # Forward and backward pass
    outputs = bad_model(x)
    loss = criterion(outputs, y)
    loss.backward()
    
    # Analyze gradients
    print("\n📊 Gradient Magnitudes in Bad Network:")
    layer_num = 0
    for name, param in bad_model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            print(f"Layer {layer_num}: {grad_mean:.2e}")
            layer_num += 1
    
    print("\n❌ Notice: Gradients become very small in early layers (vanishing!)")
    
    print("\n" + "-"*80)
    print("Part B: Good Network (Proper Initialization & Activation)")
    print("-"*80)
    
    class GoodDeepNetwork(nn.Module):
        """Network with proper gradient flow"""
        def __init__(self):
            super(GoodDeepNetwork, self).__init__()
            layers = []
            for i in range(10):
                layers.append(nn.Linear(100, 100))
                layers.append(nn.ReLU())  # ReLU doesn't saturate
                layers.append(nn.BatchNorm1d(100))  # Normalize activations
            layers.append(nn.Linear(100, 10))
            self.network = nn.Sequential(*layers)
            
            # Xavier/He initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
        def forward(self, x):
            return self.network(x)
    
    # Test good network
    good_model = GoodDeepNetwork()
    outputs = good_model(x)
    loss = criterion(outputs, y)
    loss.backward()
    
    print("\n📊 Gradient Magnitudes in Good Network:")
    layer_num = 0
    for name, param in good_model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            print(f"Layer {layer_num}: {grad_mean:.2e}")
            layer_num += 1
    
    print("\n✓ Notice: Gradients are more uniform across layers!")
    
    print("\n💡 Key Insights:")
    print("1. Sigmoid activation saturates → gradients near 0")
    print("2. ReLU doesn't saturate for positive values")
    print("3. Proper initialization (Kaiming/He) prevents exploding/vanishing")
    print("4. Batch Normalization helps maintain gradient flow")
    
    print("\n🎓 Theoretical Question:")
    print("Q: Why does sigmoid cause vanishing gradients?")
    print("A: Sigmoid derivative: σ'(x) = σ(x)(1-σ(x))")
    print("   Maximum value: 0.25 (at x=0)")
    print("   Chain rule: multiply many 0.25s → exponentially small!")
    
    print("\n✓ Problem 1 Complete!\n")


# ==============================================================================
# PROBLEM 2: Custom Loss Function - Focal Loss Implementation
# ==============================================================================
"""
Challenge: Implement Focal Loss from scratch (used in object detection)
Understand why standard CrossEntropy fails for imbalanced datasets

Focal Loss = -α(1-p_t)^γ * log(p_t)
where p_t is the probability of correct class
"""

def problem_2_custom_loss():
    print("="*80)
    print("PROBLEM 2: Custom Loss Function - Focal Loss")
    print("="*80)
    
    print("\n🎯 Challenge:")
    print("Implement Focal Loss for handling class imbalance.")
    print("Compare it with standard CrossEntropyLoss on imbalanced data.")
    
    print("\n📚 Background:")
    print("Focal Loss reduces loss for well-classified examples.")
    print("Formula: FL(p_t) = -α(1-p_t)^γ * log(p_t)")
    print("  - γ (gamma): focusing parameter (typically 2)")
    print("  - α (alpha): balancing parameter")
    print("  - p_t: probability of correct class")
    
    class FocalLoss(nn.Module):
        """
        YOUR TASK: Complete this implementation
        """
        def __init__(self, alpha=1.0, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            """
            inputs: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class labels
            """
            # Step 1: Convert logits to probabilities
            probs = F.softmax(inputs, dim=1)
            
            # Step 2: Get probability of correct class
            # Hint: Use torch.gather or index_select
            batch_size = inputs.size(0)
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
            p_t = (probs * targets_one_hot).sum(dim=1)  # Probability of true class
            
            # Step 3: Calculate focal loss
            # FL = -α(1-p_t)^γ * log(p_t)
            focal_weight = self.alpha * (1 - p_t) ** self.gamma
            ce_loss = -torch.log(p_t + 1e-8)  # Add epsilon for numerical stability
            focal_loss = focal_weight * ce_loss
            
            return focal_loss.mean()
    
    print("\n" + "-"*80)
    print("Testing on Imbalanced Dataset")
    print("-"*80)
    
    # Create highly imbalanced dataset
    # Class 0: 90%, Class 1: 9%, Class 2: 1%
    n_samples = 1000
    class_0_samples = int(0.9 * n_samples)
    class_1_samples = int(0.09 * n_samples)
    class_2_samples = n_samples - class_0_samples - class_1_samples
    
    X = torch.randn(n_samples, 10)
    y = torch.cat([
        torch.zeros(class_0_samples),
        torch.ones(class_1_samples),
        torch.full((class_2_samples,), 2)
    ]).long()
    
    print(f"\nDataset distribution:")
    print(f"  Class 0: {class_0_samples} samples (90%)")
    print(f"  Class 1: {class_1_samples} samples (9%)")
    print(f"  Class 2: {class_2_samples} samples (1%)")
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    )
    
    # Compare two loss functions
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    ce_loss = nn.CrossEntropyLoss()
    
    print("\n" + "-"*80)
    print("Comparing Loss Values")
    print("-"*80)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        
        # Calculate both losses
        fl = focal_loss(outputs, y)
        cel = ce_loss(outputs, y)
        
        print(f"\nCrossEntropy Loss: {cel.item():.4f}")
        print(f"Focal Loss: {fl.item():.4f}")
        
        # Check per-class performance
        probs = F.softmax(outputs, dim=1)
        _, preds = probs.max(1)
        
        for class_id in range(3):
            mask = (y == class_id)
            if mask.sum() > 0:
                acc = (preds[mask] == y[mask]).float().mean()
                print(f"Class {class_id} Accuracy: {acc.item()*100:.2f}%")
    
    print("\n💡 Key Insights:")
    print("1. Focal Loss downweights easy examples (well-classified)")
    print("2. γ=0 → Focal Loss = CrossEntropy")
    print("3. γ↑ → More focus on hard examples")
    print("4. Useful for: object detection, medical diagnosis, fraud detection")
    
    print("\n🎓 Theoretical Question:")
    print("Q: Why does standard CrossEntropy fail on imbalanced data?")
    print("A: Model learns to predict majority class (easy examples)")
    print("   Minority classes contribute little to loss → ignored")
    print("   Focal Loss fixes this by focusing on hard/rare examples")
    
    print("\n✓ Problem 2 Complete!\n")


# ==============================================================================
# PROBLEM 3: Architectural Challenge - Residual Networks
# ==============================================================================
"""
Challenge: Implement ResNet block and understand skip connections
Why do skip connections help training very deep networks?
"""

def problem_3_residual_networks():
    print("="*80)
    print("PROBLEM 3: Residual Networks - Skip Connections")
    print("="*80)
    
    print("\n🎯 Challenge:")
    print("Implement a ResNet block with skip connections.")
    print("Compare gradient flow with and without skip connections.")
    
    print("\n📚 Background:")
    print("ResNet introduced skip connections: y = F(x) + x")
    print("This solves degradation problem in very deep networks.")
    
    class ResidualBlock(nn.Module):
        """
        YOUR TASK: Complete the residual block implementation
        """
        def __init__(self, in_channels, out_channels):
            super(ResidualBlock, self).__init__()
            
            # Main path: two conv layers
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # Skip connection: match dimensions if needed
            self.shortcut = nn.Sequential()
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),  # 1x1 conv
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            # Main path
            identity = x
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            
            # Skip connection
            out += self.shortcut(identity)
            out = F.relu(out)
            
            return out
    
    class PlainBlock(nn.Module):
        """Regular block without skip connection for comparison"""
        def __init__(self, in_channels, out_channels):
            super(PlainBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return x
    
    # Build deep networks
    class DeepResNet(nn.Module):
        def __init__(self, num_blocks=10):
            super(DeepResNet, self).__init__()
            self.blocks = nn.ModuleList([
                ResidualBlock(64, 64) for _ in range(num_blocks)
            ])
            self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 8 * 8, 10)
        
        def forward(self, x):
            x = self.conv_in(x)
            for block in self.blocks:
                x = block(x)
            x = F.adaptive_avg_pool2d(x, (8, 8))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    class DeepPlainNet(nn.Module):
        def __init__(self, num_blocks=10):
            super(DeepPlainNet, self).__init__()
            self.blocks = nn.ModuleList([
                PlainBlock(64, 64) for _ in range(num_blocks)
            ])
            self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 8 * 8, 10)
        
        def forward(self, x):
            x = self.conv_in(x)
            for block in self.blocks:
                x = block(x)
            x = F.adaptive_avg_pool2d(x, (8, 8))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    print("\n" + "-"*80)
    print("Comparing Gradient Flow: ResNet vs Plain Network")
    print("-"*80)
    
    # Test both networks
    resnet = DeepResNet(num_blocks=10)
    plainnet = DeepPlainNet(num_blocks=10)
    
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()
    
    # ResNet gradients
    outputs = resnet(x)
    loss = criterion(outputs, y)
    loss.backward()
    
    resnet_grads = []
    for name, param in resnet.named_parameters():
        if 'conv1.weight' in name and param.grad is not None:
            resnet_grads.append(param.grad.abs().mean().item())
    
    # Plain network gradients
    plainnet.zero_grad()
    outputs = plainnet(x)
    loss = criterion(outputs, y)
    loss.backward()
    
    plain_grads = []
    for name, param in plainnet.named_parameters():
        if 'conv1.weight' in name and param.grad is not None:
            plain_grads.append(param.grad.abs().mean().item())
    
    print("\n📊 Gradient Magnitudes (first conv layer of each block):")
    print(f"\n{'Block':<10} {'PlainNet':<15} {'ResNet':<15}")
    print("-" * 40)
    for i, (pg, rg) in enumerate(zip(plain_grads, resnet_grads)):
        print(f"{i:<10} {pg:<15.2e} {rg:<15.2e}")
    
    print("\n💡 Key Observations:")
    print("1. Plain network: gradients diminish in early blocks")
    print("2. ResNet: gradients more uniform across all blocks")
    print("3. Skip connection provides direct gradient path")
    
    print("\n🎓 Theoretical Analysis:")
    print("Q: Why do skip connections help?")
    print("\nA: Mathematical proof:")
    print("   Without skip: y = F(x)")
    print("   Gradient: dy/dx = F'(x)")
    print("   Chain rule over n layers: ∏F'(x_i) → can vanish")
    print("")
    print("   With skip: y = F(x) + x")
    print("   Gradient: dy/dx = F'(x) + 1")
    print("   The '+1' ensures gradient >= 1 → no vanishing!")
    
    print("\n✓ Problem 3 Complete!\n")


# ==============================================================================
# PROBLEM 4: Batch Normalization vs Layer Normalization
# ==============================================================================
"""
Challenge: Understand the difference between BatchNorm and LayerNorm
When to use each? Implementation-level understanding required.
"""

def problem_4_normalization_comparison():
    print("="*80)
    print("PROBLEM 4: Batch Normalization vs Layer Normalization")
    print("="*80)
    
    print("\n🎯 Challenge:")
    print("Implement both normalizations from scratch (simplified version)")
    print("Understand when each is appropriate")
    
    print("\n📚 Background:")
    print("BatchNorm: Normalize across batch dimension")
    print("LayerNorm: Normalize across feature dimension")
    
    def manual_batch_norm(x, eps=1e-5):
        """
        x: (batch_size, features)
        Normalize across batch dimension
        """
        # Calculate mean and variance across batch
        mean = x.mean(dim=0, keepdim=True)  # (1, features)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        return x_normalized, mean, var
    
    def manual_layer_norm(x, eps=1e-5):
        """
        x: (batch_size, features)
        Normalize across feature dimension
        """
        # Calculate mean and variance across features
        mean = x.mean(dim=1, keepdim=True)  # (batch_size, 1)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        return x_normalized, mean, var
    
    print("\n" + "-"*80)
    print("Testing with Sample Data")
    print("-"*80)
    
    # Create sample data
    batch_size, features = 4, 3
    x = torch.randn(batch_size, features) * 10 + 5  # Mean≈5, Std≈10
    
    print(f"\nOriginal data (shape: {x.shape}):")
    print(x)
    print(f"\nMean per feature: {x.mean(dim=0)}")
    print(f"Std per feature: {x.std(dim=0)}")
    print(f"Mean per sample: {x.mean(dim=1)}")
    print(f"Std per sample: {x.std(dim=1)}")
    
    # Apply BatchNorm
    x_bn, bn_mean, bn_var = manual_batch_norm(x)
    print("\n" + "-"*80)
    print("After Batch Normalization:")
    print("-"*80)
    print(x_bn)
    print(f"\nMean per feature (should be ≈0): {x_bn.mean(dim=0)}")
    print(f"Std per feature (should be ≈1): {x_bn.std(dim=0, unbiased=False)}")
    
    # Apply LayerNorm
    x_ln, ln_mean, ln_var = manual_layer_norm(x)
    print("\n" + "-"*80)
    print("After Layer Normalization:")
    print("-"*80)
    print(x_ln)
    print(f"\nMean per sample (should be ≈0): {x_ln.mean(dim=1)}")
    print(f"Std per sample (should be ≈1): {x_ln.std(dim=1, unbiased=False)}")
    
    print("\n" + "-"*80)
    print("Visual Comparison:")
    print("-"*80)
    print("\nBatchNorm visualization (each column normalized):")
    print("  Sample 1 → | F1  F2  F3 |")
    print("  Sample 2 → | F1  F2  F3 |")
    print("  Sample 3 → | F1  F2  F3 |")
    print("             ↓   ↓   ↓")
    print("        Normalize each column")
    
    print("\nLayerNorm visualization (each row normalized):")
    print("  Sample 1 → | F1  F2  F3 | → Normalize")
    print("  Sample 2 → | F1  F2  F3 | → Normalize")
    print("  Sample 3 → | F1  F2  F3 | → Normalize")
    
    print("\n" + "-"*80)
    print("Testing with Small Batch (BatchNorm problem!)")
    print("-"*80)
    
    # Problem: BatchNorm with batch_size=1
    x_single = torch.randn(1, 3) * 10 + 5
    print(f"\nSingle sample: {x_single}")
    
    try:
        x_bn_single, _, _ = manual_batch_norm(x_single)
        print(f"BatchNorm result: {x_bn_single}")
        print("Warning: Variance is 0! (only one sample)")
    except:
        print("Error: Cannot compute variance with single sample!")
    
    x_ln_single, _, _ = manual_layer_norm(x_single)
    print(f"LayerNorm result: {x_ln_single}")
    print("✓ LayerNorm works fine!")
    
    print("\n💡 When to use each:")
    print("\nBatch Normalization:")
    print("  ✓ CNNs for images (large, consistent batches)")
    print("  ✓ Fixed batch size during training and inference")
    print("  ✗ Small batch sizes (statistics unreliable)")
    print("  ✗ RNNs (sequence length varies)")
    print("  ✗ Batch size = 1 at inference")
    
    print("\nLayer Normalization:")
    print("  ✓ RNNs, Transformers (variable sequence length)")
    print("  ✓ Batch size = 1 scenarios")
    print("  ✓ Online learning (single sample at a time)")
    print("  ✗ Generally slower than BatchNorm on CNNs")
    
    print("\n🎓 Deep Question:")
    print("Q: Why does BatchNorm improve training?")
    print("\nA: Multiple theories:")
    print("  1. Reduces internal covariate shift")
    print("  2. Smooths optimization landscape")
    print("  3. Acts as regularization (noise from batch statistics)")
    print("  4. Allows higher learning rates")
    
    print("\n✓ Problem 4 Complete!\n")


# ==============================================================================
# PROBLEM 5: Learning Rate Scheduling & Warmup
# ==============================================================================
"""
Challenge: Implement learning rate warmup and cosine annealing
Understanding why LR scheduling is crucial for convergence
"""

def problem_5_lr_scheduling():
    print("="*80)
    print("PROBLEM 5: Learning Rate Scheduling & Warmup")
    print("="*80)
    
    print("\n🎯 Challenge:")
    print("Implement and visualize different LR schedules")
    print("Understand the impact on training dynamics")
    
    print("\n📚 Background:")
    print("Learning rate is the most important hyperparameter!")
    print("Scheduling helps: Fast initial learning → Fine-tuning later")
    
    # Simple model for demonstration
    model = nn.Sequential(nn.Linear(10, 10))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    print("\n" + "-"*80)
    print("1. Step LR: Decrease LR at specific epochs")
    print("-"*80)
    
    from torch.optim.lr_scheduler import StepLR
    scheduler_step = StepLR(optimizer, step_size=30, gamma=0.1)
    
    print("StepLR(step_size=30, gamma=0.1)")
    print("LR reduced by 10x every 30 epochs")
    
    lrs_step = []
    for epoch in range(100):
        lrs_step.append(optimizer.param_groups[0]['lr'])
        scheduler_step.step()
    
    print(f"Epoch 0: LR = {lrs_step[0]:.4f}")
    print(f"Epoch 30: LR = {lrs_step[30]:.4f}")
    print(f"Epoch 60: LR = {lrs_step[60]:.4f}")
    
    # Reset
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.1
    
    print("\n" + "-"*80)
    print("2. Cosine Annealing: Smooth decrease")
    print("-"*80)
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    
    print("CosineAnnealingLR(T_max=100, eta_min=0.001)")
    print("Smoothly decrease from 0.1 → 0.001 following cosine curve")
    
    lrs_cosine = []
    for epoch in range(100):
        lrs_cosine.append(optimizer.param_groups[0]['lr'])
        scheduler_cosine.step()
    
    print(f"Epoch 0: LR = {lrs_cosine[0]:.4f}")
    print(f"Epoch 50: LR = {lrs_cosine[50]:.4f}")
    print(f"Epoch 99: LR = {lrs_cosine[99]:.4f}")
    
    # Reset
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.1
    
    print("\n" + "-"*80)
    print("3. Custom Warmup + Cosine Annealing")
    print("-"*80)
    
    print("Warmup: Gradually increase LR (0 → peak) in first few epochs")
    print("Why? Large LR at start can cause instability")
    
    def get_lr_with_warmup(epoch, warmup_epochs=10, max_epochs=100, 
                           base_lr=0.1, min_lr=0.001):
        """
        YOUR TASK: Implement warmup + cosine schedule
        """
        if epoch < warmup_epochs:
            # Linear warmup
            return base_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
    
    lrs_warmup = []
    for epoch in range(100):
        lr = get_lr_with_warmup(epoch)
        lrs_warmup.append(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    print(f"\nEpoch 0 (warmup): LR = {lrs_warmup[0]:.4f}")
    print(f"Epoch 5 (warmup): LR = {lrs_warmup[5]:.4f}")
    print(f"Epoch 10 (peak): LR = {lrs_warmup[10]:.4f}")
    print(f"Epoch 50: LR = {lrs_warmup[50]:.4f}")
    print(f"Epoch 99: LR = {lrs_warmup[99]:.4f}")
    
    print("\n💡 Key Insights:")
    print("1. Warmup prevents instability in early training")
    print("2. Cosine annealing helps fine-tune final weights")
    print("3. LR scheduling is crucial for convergence")