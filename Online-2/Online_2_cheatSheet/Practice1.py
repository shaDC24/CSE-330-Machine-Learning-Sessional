import torch
import torch.nn as nn
import torch.optim as optim

class SimpleFNN(nn.Module):
    def __init__(self):
        super(SimpleFNN, self).__init__()

        # 1. Flatten Input
        # The input image is 28x28. We need to flatten it to a 1D vector.
        # TODO: Define a Flatten layer
        self.flatten = nn.Flatten()

        # 2. Hidden Layer
        # TODO: Define a Linear layer from 784 input features to 128 hidden units
        self.fc1 =nn.Linear(784, 128)

        # 3. Activation
        # TODO: Define ReLU activation
        self.relu =nn.ReLU()

        # 4. Output Layer
        # TODO: Define a Linear layer from 128 hidden units to 10 output classes
        self.fc2 =nn.Linear(128,10)

    def forward(self, x):
        # TODO: Implement the forward pass: Flatten -> FC1 -> ReLU -> FC2
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)

        return x

# ==========================================
# Training Setup
# ==========================================
model = SimpleFNN()

# TODO: Define the SGD Optimizer
# Parameters: model params, learning rate = 0.01, momentum = 0.9
optimizer =optim.SGD(model.parameters(),lr=0.01,momentum=0.9)




class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        # 1. Convolutional Layer
        # Input Image: 28x28, Channels: 1
        # TODO: Define Conv2d (In: 1, Out: 32, Kernel: 3, Stride: 1, Padding: 1)
        self.conv1 =nn.Conv2d(1,32,3,1,1)

        self.relu = nn.ReLU()

        # 2. Pooling Layer
        # TODO: Define MaxPool2d (Kernel: 2, Stride: 2)
        self.pool =nn.MaxPool2d(kernel_size=2,stride=2)

        # 3. Fully Connected Layer
        # Input: 28x28 -> Conv -> 28x28 -> Pool -> 14x14
        # TODO: Define Linear layer (Input Features = Calculated Size * 32, Output = 10)
        self.fc =nn.Linear(32*14*14,10)

    def forward(self, x):
        # TODO: Implement forward pass: Conv -> ReLU -> Pool -> Flatten -> FC
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

# ==========================================
# Training Setup
# ==========================================
model = BasicCNN()

# TODO: Define the Adam Optimizer
# Parameters: model params, learning rate = 0.001
optimizer =optim.Adam(model.parameters(), lr=0.001)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RobustCNN(nn.Module):
    def __init__(self):
        super(RobustCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # 1. Regularization
        # TODO: Define a Dropout layer with probability p=0.5
        self.dropout =nn.Dropout(0.5)

        # Input to FC is 16 * 14 * 14
        self.fc = nn.Linear(16 * 14 * 14, 10)

        # 2. Initialization
        # TODO: Initialize conv1 weights using Kaiming Normal
        # TODO: Initialize fc weights using Xavier Uniform
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # TODO: Apply Dropout before the FC layer
        x=self.dropout(x)

        x = self.fc(x)

        # TODO: Return LogSoftmax on dim=1
        return F.log_softmax(x, dim=1)


# ==========================================
# Training Setup
# ==========================================
model = RobustCNN()

# TODO: Define Negative Log Likelihood Loss
criterion = nn.NLLLoss()

