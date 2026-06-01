import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
# print(df.columns)
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
print(df.shape)

#splitting features and labels
X_train,X_test,Y_train,Y_test = train_test_split(df.iloc[:, 1:],df.iloc[:,0],test_size=0.2)

#standardizing the features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#encoding the labels because they are categorical(M,B)
label_encoder=LabelEncoder()
Y_train=label_encoder.fit_transform(Y_train)
Y_test=label_encoder.transform(Y_test)

#converting to tensors from numpy arrays
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(Y_train)
y_test_tensor = torch.from_numpy(Y_test)
print(X_train_tensor.shape, y_train_tensor.shape)
print(X_train.dtype)

#Defining the model
class MySimpleNN():
    def __init__(self,X_train_tensor):
        self.weights=torch.rand(X_train_tensor.shape[1],1,dtype=torch.float64,requires_grad=True)
        self.bias=torch.rand(1,dtype=torch.float64,requires_grad=True)

    def forward(self,X):
        z=torch.matmul(X,self.weights)+self.bias    
        y_pred=torch.sigmoid(z)
        return y_pred
    
    def binary_CrossEntropy_loss(self,prediction,target):
        epsilon=1e-8
        prediction=torch.clamp(prediction,epsilon,1-epsilon)
        loss=-(target*torch.log(prediction)+(1-target)*torch.log(1-prediction)).mean()
        return loss

learning_rate=0.01
num_epochs=25

#Training the pipeline

#create the model
model=MySimpleNN(X_train_tensor)

for epoch in range(num_epochs):
    #forward pass
    y_pred=model.forward(X_train_tensor)    
    
    #loss calculation
    loss=model.binary_CrossEntropy_loss(y_pred,y_train_tensor)
    

    #backward pass  
    loss.backward()

    #parameter update
    with torch.no_grad():
        model.bias -= learning_rate * model.bias.grad
        model.weights -= learning_rate * model.weights.grad
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    #print loss in each epoch   
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")  

#Testing the pipeline
with torch.no_grad():
    y_test_pred=model.forward(X_test_tensor)
    y_test_pred_label=(y_test_pred>=0.8).float()
    accuracy=(y_test_pred_label==y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy.item()*100:.2f}%")