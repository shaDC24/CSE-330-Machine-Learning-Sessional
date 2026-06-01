import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def train_test_split(X, y, test_size=0.2, random_state=42):
    
    np.random.seed(random_state)
    n_samples=X.shape[0]
    test_count=int(n_samples*test_size)
    indices=np.arange(n_samples)
    np.random.shuffle(indices)
    test_indices=indices[:test_count]
    train_indices=indices[test_count:]
    X_train=X[train_indices]
    y_train=y[train_indices]
    X_test=X[test_indices]
    y_test=y[test_indices]
    return X_train, X_test, y_train, y_test
    
# load the dataset
df= pd.read_csv('A1_19Batch\A1.csv')
print(df)
X = df[['X1', 'X2']].values
y = df['y'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot the dataset (optional for visualization)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', alpha=0.6)
plt.legend()
plt.show()