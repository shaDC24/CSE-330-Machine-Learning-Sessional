import torch

x=torch.tensor(3.0,requires_grad=True)
y=x**2 

print("X : ",x)
print("Y : ",y)
z=torch.sin(y)
print("Z : ",z)

# y.backward()
# print("dy/dx : ",x.grad)
z.backward()
print("dz/dx : ",x.grad)

x=torch.tensor(6.7)
y=torch.tensor(0.0) #true label = binary
w=torch.tensor(1.0,requires_grad=True) #weight
b=torch.tensor(0.0,requires_grad=True) #bias

def binary_CrossEntropy_loss(prediction,target):
    epsilon=1e-8
    prediction=torch.clamp(prediction,epsilon,1-epsilon)
    loss=-(target*torch.log(prediction)+(1-target)*torch.log(1-prediction))
    return loss 

#forward pass
z=w*x+b
y_pred=torch.sigmoid(z)
loss=binary_CrossEntropy_loss(y_pred,y)

#backward pass
loss.backward()
print("Loss : ",loss.item())
print("dLoss/dw : ",w.grad)
print("dLoss/db : ",b.grad) 



x=torch.tensor([1.0,2.0,3.0],requires_grad=True)
y=(x**2).mean()#y=func(x1,x2,x3)  
y.backward()
print("dy/dx : ",x.grad)
x.grad.zero_() #clear gradients
# x.requires_grad_(False) #disable gradient tracking
print("Requires grad : ",x.requires_grad)


z=x.detach()
print("X : ",x)
print("Z : ",z)

with torch.no_grad():
    y=x**2
    print("Y : ",y)