import torch
import torch.nn as nn

class MODEL(nn.Module):
    def __init__(self, num_features):
        super(MODEL, self).__init__()
        self.fc1 = nn.Linear(num_features, 3)
        self.relU=nn.ReLU()
        self.fc2=nn.Linear(3,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relU(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        return x
#creating datasets
features=torch.rand(10,5)
#creating models
model=MODEL(num_features=features.shape[1])
#forward pass
output=model(features)
print(output)    
print("1 no Layers' weights")
print(model.fc1.weight)   

class MODEL(nn.Module):
    def __init__(self, num_features):
        super(MODEL, self).__init__()
        self.fc1 = nn.Linear(num_features, 3)
        self.relU=nn.ReLU()
        self.fc2=nn.Linear(3,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relU(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        return x
#creating datasets
features=torch.rand(10,5)
#creating models
model=MODEL(num_features=features.shape[1])
#forward pass
output=model(features)
print(output)    
print("1 no Layers' weights")
print(model.fc1.weight)    





class MODEL_Seq(nn.Module):
    def __init__(self, num_features):
        super(MODEL_Seq, self).__init__()
        self.network=nn.Sequential(
            nn.Linear(num_features,3),
            nn.ReLU(),
            nn.Linear(3,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x
#creating datasets
features=torch.rand(10,5)
#creating models
model=MODEL_Seq(num_features=features.shape[1])
#forward pass
output=model(features)
print(output)    
print("1 no Layers' weights")
print(model.network[0].weight)    


class MySimpleNN(nn.Module):
    def __init__(self, num_features):
        super(MySimpleNN, self).__init__()
        self.fc1=nn.Linear(num_features, 1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, X):
        z = self.fc1(X)
        y_pred = self.sigmoid(z)
        return y_pred
    
    # def binary_CrossEntropy_loss(self, prediction, target):
    #     epsilon = 1e-8
    #     prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    #     loss = -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction)).mean()
    #     return loss

learning_rate = 0.01
num_epochs = 25 


#creating model
model = MySimpleNN(num_features=features.shape[1])
loss_function=nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Training the pipeline
for epo in range(num_epochs):
    #forward pass
    y_pred = model.forward(features)
    #loss calculation
    target=torch.randint(0,2,(10,1)).float()
    # loss = model.binary_CrossEntropy_loss(y_pred, target)
    loss = loss_function(y_pred, target)
    print(f"Initial Loss: {loss.item()}")
    #backward pass
    loss.backward()
    #parameter update
    # with torch.no_grad():   
    #     model.fc1.weight -= learning_rate * model.fc1.weight.grad
    #     model.fc1.bias -= learning_rate * model.fc1.bias.grad
    # #zero gradients
    # model.fc1.weight.grad.zero_()
    # model.fc1.bias.grad.zero_()
    # print(f"{epo} no Epoch Loss: {loss.item()}")
    optimizer.step()
    optimizer.zero_grad()
    print(f"{epo} no Epoch Loss: {loss.item()}")



#Testing the pipeline
with torch.no_grad():
    y_test_pred = model.forward(features)
    y_test_pred_label = (y_test_pred >= 0.5).float()
    accuracy = (y_test_pred_label == target).float().mean()
    print(f"Test Accuracy: {accuracy.item()}")    