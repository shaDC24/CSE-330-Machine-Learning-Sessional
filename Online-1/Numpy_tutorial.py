import numpy as np;
import matplotlib.pyplot as plt;


arr_1d = np.array([1,2,3,4,5])
print("1D array : ",arr_1d)

arr_2d = np.array([[1,2,3],[4,5,6]])
print("2D array : ",arr_2d)


dataset = np.array([
    [1, 150000, 140000, 160000, 155000],
    [2, 145000, 135000, 165000, 150000],
    [3, 155000, 145000, 170000, 160000],
    [4, 148000, 138000, 168000, 152000],
    [5, 152000, 142000, 172000, 158000]
])

print("======Sales Analysis=====")
print("\nSales data shape : ",dataset.shape)
print("\nSample data for 1st 3 restaurants : \n",dataset[:3])
print("\nSample dataset for : \n",dataset[:,1:3])

print("\nTotal sales per year : \n")
print(np.sum(dataset[:,1:],axis=0))

print("\nMinimum sales per restaurant : \n")
min_sales=np.min(dataset[:,1:],axis=1)
print(min_sales)


print("\nMaximum sales per year : \n")
max_sales_per_year=np.max(dataset[:,1:],axis=0)
print(max_sales_per_year)

print("\nAverage sales per restuarant : \n")
avg_sales_per_restaurant = np.mean(dataset[:,1:],axis=1)
print(avg_sales_per_restaurant)

print("\nCumulative sales per restuarant : \n")
cumulative_sales_per_restaurant = np.cumsum(dataset[:,1:],axis=1)
print(cumulative_sales_per_restaurant)

plt.figure(figsize=(10,6))
plt.plot(np.mean(cumulative_sales_per_restaurant,axis=0))
plt.title("Average cumulative sales across restaurant")
plt.xlabel("Years")
plt.ylabel("Sales")
plt.grid(True)
# plt.show()


vector1=np.array([1,2,3,4,5])
vector2=np.array([6,7,8,9,10])
print("\nVector Addition : \n",vector1+vector2)
print("\nVector multiplication : \n",vector1*vector2)
print("\nDot product : \n",np.dot(vector1,vector2))
print("\nAngle : \n",np.arccos(np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))))

restaurant_types = np.array(['Italian', 'Chinese', 'Mexican', 'Indian', 'Japanese'])
vectorized_upper=np.vectorize(str.upper)
print("\nUpper cse the restaurant types : \n",vectorized_upper(restaurant_types))

monthly_avg=dataset[:,1:]/12
print("\nMonthly average: \n",monthly_avg)

array1=np.array([[1,2,3],[4,5,6]])
array2=np.random.rand(3,3)
array3=np.zeros((4,4))

np.save("ARRAY1.npy",array1)

print("\nLoaded from my file : \n")
loaded_array1=np.load("ARRAY1.npy")
print(loaded_array1)


try:
    logo=np.load("numpy-logo.npy")
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(logo)
    plt.title("Numpy Logo")
    plt.grid(False)
    
    dark_logo=1-logo
    plt.subplot(122)
    plt.imshow(dark_logo)
    plt.title("Numpy dark Logo")
    plt.grid(False)
    plt.show()

except FileNotFoundError:
    print("numpy logo file not found .")