import torch
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Create a tensor
# using empty
tensor_empty = torch.empty(3, 4)
print("Empty Tensor:\n", tensor_empty)  
print("Type:", tensor_empty.dtype)  
print("Type of tensors using empty:", type(tensor_empty))

#using zeros
tensor_zeros = torch.zeros(3, 4)
print("\nZeros Tensor:\n", tensor_zeros)

#using ones
tensor_ones = torch.ones(3, 4)
print("\nOnes Tensor:\n", tensor_ones)

#using randn
tensor_randn = torch.randn(3, 4)
print("\nRandom Normal Tensor:\n", tensor_randn)

#using manual seed
torch.manual_seed(100)
tensor_randn_seeded = torch.randn(3, 4)
print("\nRandom Normal Tensor with Seed:\n", tensor_randn_seeded)

#using tensor
data=[[1, 2, 3], [4, 5, 6]]
tensor_from_data = torch.tensor(data)
print("\nTensor from Data:\n", tensor_from_data)

#using arange
tensor_arange = torch.arange(0, 10, step=2)
print("\nArange Tensor:\n", tensor_arange)

#using linspace
tensor_linspace = torch.linspace(0, 1, steps=5)
print("\nLinspace Tensor:\n", tensor_linspace)

#using eye
tensor_eye = torch.eye(3)
print("\nEye Tensor:\n", tensor_eye)

#using full
tensor_full= torch.full((2, 3), 7)
print("\nFull Tensor:\n", tensor_full)

#shape
x=torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\nShape of tensor_ones:", x.shape)

print("Empty tensor from x :",torch.empty_like(x))
print("One tensor from x :",torch.ones_like(x))
print("Random tensor from x :",torch.rand_like(x, dtype=torch.float))

#data type
x_int = torch.tensor([1, 2, 3], dtype=torch.int32)
x_float = x_int.to(torch.float32)
print("\nInteger Tensor:", x_int)
print("Converted Float Tensor:", x_float)

#writing all tha availabe dtypes
print("\nAvailable Data Types in PyTorch:")
for dtype in [torch.float16, torch.float32, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
    print(dtype)


#Mathematical Operations
print("Scalar Operations on Tensors:\n")
x=torch.randn(2, 3)
print("\nOriginal Tensor x:\n", x)
print("Addition:\n", x + 2)
print("Subtraction:\n", x - 2)
print("Multiplication:\n", x * 2)
print("Division:\n", x / 2)
print("Exponentiation:\n", x ** 2)
print("Int Division:\n", x // 2)
print("Mod ulus:\n", x % 2 )


print("\nTensor-Tensor Operations:\n")
y=torch.randn(2, 3)
print("\nTensor x:\n", x)
print("Tensor y:\n", y)
print("Addition:\n", x + y)
print("Subtraction:\n", x - y)
print("Multiplication:\n", x * y)
print("Division:\n", x / y)
print("Matrix Multiplication (using @):\n", x @ y.t())
print("Matrix Multiplication (using torch.matmul):\n", torch.matmul(x, y.t()))
print("Element-wise Power:\n", x ** y)
print("Modulus:\n", x % y)


x=torch.tensor([1,-2,3,-4])
print("\nMore element wise operations:\n")
print("Absolute:\n", torch.abs(x))
print("Square Root:\n", torch.sqrt(torch.abs(x)))
print("Round:\n", torch.round(torch.tensor([1.2, 2.5, 3.7])))
print("Ceil:\n", torch.ceil(torch.tensor([1.2, 2.5, 3.7])))
print("Floor:\n", torch.floor(torch.tensor([1.2, 2.5, 3.7])))
print("Clamp (0,2):\n", torch.clamp(torch.tensor([-1.0, 0.5, 2.5, 3.0]), min=0, max=2))
print("Negative:\n", torch.neg(x))

x=torch.tensor([[1,-2,3,
                 -4],[1,2,3,4]],dtype=torch.float32)
y=torch.tensor([[4,3,2,-1],[-4,-3,-2,-1]],dtype=torch.float32)
print("Reduction Operations:\n")
print("Sum:\n", torch.sum(x))
print("Column sum:\n", torch.sum(x, dim=0))
print("Row sum:\n", torch.sum(x, dim=1))
print("Mean:\n", torch.mean(x))
print("Column mean:\n", torch.mean(x, dim=0))
print("Row mean:\n", torch.mean(x, dim=1))
print("Max:\n", torch.max(x))
print("Min:\n", torch.min(x))
print("Product:\n", torch.prod(x))
print("Column Product:\n", torch.prod(x, dim=0))
print("Row Product:\n", torch.prod(x, dim=1))   
print("Standard Deviation:\n", torch.std(x))
print("Variance:\n", torch.var(x))
print("Highest index:\n", torch.argmax(x))
print("Highest index in Column:\n", torch.argmax(x, dim=0))
print("Highest index in Row:\n", torch.argmax(x, dim=1))
print("Lowest index:\n", torch.argmin(x))
print("Lowest index in Column:\n", torch.argmin(x, dim=0))
print("Lowest index in Row:\n", torch.argmin(x, dim=1))
print("Dot Product:\n", torch.dot(x[0], y[0]))
print("Transpose:\n", torch.t(x))
print("Determinant:\n", torch.det(torch.tensor([[1., 2.], [3., 4.]])))
print("Matrix Inverse:\n", torch.inverse(torch.tensor([[1., 2.], [3., 4.]])))

#Comparison Operations
print("\nComparison Operations:\n")
a = torch.tensor([[5,0,1],[1,1,3]])
b = torch.tensor([[2,6,3],[5,2,5]])
print("Less than:\n",a<b)
print("Less than or equal:\n",a<=b)
print("Greater than:\n",a>b)
print("Greater than or equal:\n",a>=b)
print("Equal:\n",a==b)
print("Not Equal:\n",a!=b)




k=torch.randint(size=(3,4),low=0,high=10)
print("\nLog Operations:\n", torch.log(k))
print("Exponential Operations:\n", torch.exp(k))
print("Logarithm base 10:\n", torch.log10(k))
print("Sigmoid Function:\n", torch.sigmoid(k))
print("Softmax along dim 0:\n", torch.nn.functional.softmax(k.float(), dim=0))
print("Softmax along dim 1:\n", torch.nn.functional.softmax(k.float(), dim=1))
print("ReLU Function:\n", torch.relu(k))



print("\nIn-place Operations:\n")
x=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)
print("Original Tensor x:\n", x)
x.add_(5)
print("After In-place Addition x.add_(5):\n", x)
x.mul_(2)
print("After In-place Multiplication x.mul_(2):\n", x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
x.relu_()
print("After In-place ReLU x.relu_():\n", x)




#copying tensors
print("\nCopying Tensors:\n")
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
y = x  # Shallow copy
z = x.clone()  # Deep copy  
print("Original Tensor x:\n", x)
y.add_(5)
print("After modifying y (shallow copy):\n", y)
print("Tensor x after modifying y:\n", x)
z.mul_(2)
print("After modifying z (deep copy):\n", z)
print("Tensor x after modifying z:\n", x)


#Tesor operations in GPU
if torch.cuda.is_available():
    print("\nTensor Operations on GPU:\n")
    x = torch.randn(3, 4)
    print("Original Tensor x on CPU:\n", x)
    x_gpu = x.to(device)
    print("Tensor x moved to GPU:\n", x_gpu)
    y_gpu = torch.randn(3, 4).to(device)
    print("Another Tensor y on GPU:\n", y_gpu)
    z_gpu = x_gpu + y_gpu
    print("Result of Addition on GPU (z = x + y):\n", z_gpu)
    z_cpu = z_gpu.to("cpu")
    print("Result moved back to CPU:\n", z_cpu)
else:
    print("\nCUDA is not available. Skipping GPU tensor operations.\n")


#reshaping tensors 
print("Reshaping Tensors:\n")
x = torch.randn(2, 3)
print("Original Tensor x:\n", x)
x_reshaped = x.view(3, 2)
print("Reshaped Tensor x to (3,2):\n", x_reshaped)
x_reshaped2 = x.reshape(6)
print("Reshaped Tensor x to (6,):\n", x_reshaped2)
x_reshaped3 = x.view(-1)
print("Reshaped Tensor x to (-1,):\n", x_reshaped3)
x_flattened = x.flatten()
print("Flattened Tensor x:\n", x_flattened)




