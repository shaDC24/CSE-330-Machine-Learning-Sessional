import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ১. কৃত্রিম ডেটাসেট তৈরি (Synthetic Data)
# আমরা y = 2x + 3 এই সরলরেখাটি শেখার চেষ্টা করব
X = torch.randn(100, 1) * 10
y = 2 * X + 3 + torch.randn(100, 1) * 2  # কিছুটা নয়েজ যোগ করা হয়েছে

# ২. সিম্পল লিনিয়ার মডেল
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

# ৩. বিভিন্ন অপ্টিমাইজার টেস্ট করার ফাংশন
def train_model(optimizer_type, learning_rate=0.01):
    model = LinearModel()
    criterion = nn.MSELoss()
    
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    for epoch in range(50):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
    return loss_history

# ৪. সব অপ্টিমাইজার চালানো এবং রেজাল্ট সংগ্রহ
lr = 0.001
losses_sgd = train_model('SGD', lr)
losses_momentum = train_model('Momentum', lr)
losses_adam = train_model('Adam', lr)

# ৫. গ্রাফের মাধ্যমে তুলনা (Visualization)
plt.figure(figsize=(10, 6))
plt.plot(losses_sgd, label='Simple SGD')
plt.plot(losses_momentum, label='SGD + Momentum')
plt.plot(losses_adam, label='Adam')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Comparison of Optimizers')
plt.legend()
plt.show()