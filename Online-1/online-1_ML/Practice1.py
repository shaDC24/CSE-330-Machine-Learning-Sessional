import pandas as pd
import numpy as np

def load_and_fill_nulls(filename, fill_value=0):
    """
    Load data from a CSV file and fill null values with specified value.
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file to load
    fill_value : float or int, default=0
        The value to use for filling null/missing values
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with null values filled
    """
    df=pd.read_csv(filename)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print("\nNull values per each column before filling:")
    print(df.isnull().sum())
    print("\nTotal null values in the dataset : ")
    print(df.isnull().sum().sum())
    df=df.fillna(fill_value)
    print(f"Null values after : {df.isnull().sum().sum()}")
    return df
#Linear regression for continuous output

def linear_Regression_with_gradient(X,y,learning_rate=0.01,epochs=100,batch_size=32):
    n_samples,n_features=X.shape
    W=np.zeros((n_features,1))
    b=0.0
    print("Linear regression")
    for epoch in range(epochs):
        indices=np.arange(n_samples)
        np.random.shuffle(indices)
        X=X[indices]
        y=y[indices]
        n_batches=0
        epoch_loss=0
        for i in range(0,n_samples,batch_size):
            Xb=X[i:i+batch_size]
            Yb=y[i:,i+batch_size]
            y_pred=Xb@W+b
            loss=np.mean((y_pred-y)**2)
            epoch_loss+=loss
            n_batches+=1
            dW=(2/len(Xb))*Xb.T@(y_pred-Yb)
            db=(2/len(Xb))*np.sum(y_pred-Yb)
            W=W-learning_rate*dW
            b=b-learning_rate*db
        if((epoch+1)%20 ==0):
            print(f"Epoch {epoch+1}/{epochs} , Loss: {epoch_loss:.6f}, Avg loss : {(epoch_loss/n_batches):.6f}")    
            
    return W,b

#Logistic regression for binary classification
def sigmoid(z):
    return (1/(1+np.exp(-np.clip(z,-500,500))))
def logistic_regression_gradient_descent(X,y,learning_rate=0.1,epochs=100,batch_size=32):
    n_samples,n_features=X.shape
    W=np.zeros((n_features,1))
    b=0.0
    
    print("\nLogistic Regression : ")
    for epoch in range(epochs):
        indices=np.arange(n_samples)
        np.random.shuffle(indices)
        X=X[indices]
        y=y[indices]
        n_batches=0        
        epoch_loss=0
        for i in range(0,n_samples,batch_size):
            Xb=X[i:i+batch_size]
            Yb=y[i:i+batch_size]
            z=Xb@W+b
            y_pred=sigmoid(z)
            epsilon=1e-8
            loss=-np.mean(Yb*np.log(epsilon+y_pred)+(1-Yb)*np.log(1-y_pred+epsilon))
            epoch_loss+=loss
            n_batches+=1
            
            dW=(1/len(Xb))*Xb.T@(y_pred-y)
            db=(1/len(Xb))*np.sum(y_pred-y)
            W=W-learning_rate*dW
            b=b-learning_rate*db
        if((epoch+1)%20==0):
            print(f"Epoch : {epoch+1}/{epochs} , Loss : {epoch_loss : .6f}, Avg loss : {(epoch_loss/n_batches):.6f}") 
    return W,b


def evaluate_linear(X,y,W,b):
    y_pred=X@W+b
    mse=np.mean((y_pred-y)**2)
    r2=1-(np.sum((y-y_pred)**2)/np.sum((y-y.mean())**2)) 
    print(f"\nLinear Regression results : ")
    print(f"MSE : {mse : .6f}")
    print(f"R^2 score : {r2:.6f}")
    return y_pred

def evaluate_logistic(X,y,W,b):
    z=X@W+b
    y_pred_prob=sigmoid(z)
    y_pred_class=(y_pred_prob>=0.5).astype(int)
    accuracy=np.mean(y_pred_class==y)
    print("\nLogistic regression results : ")
    print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    return y_pred_class

if __name__  == "__main__":
    df=load_and_fill_nulls("sample.csv",fill_value=0)
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values.reshape(-1,1)
    
    X=(X-X.mean(axis=0))/(X.std(axis=0)+1e-8)#Normalize features
    
    unique_value=np.unique(y)
    is_binary=len(unique_value)==2 and set(unique_value).issubset({0,1})
    
    
    # ===== CHOOSE ONE BASED ON YOUR PROBLEM =====
    if is_binary:
        # Option 1: LINEAR REGRESSION (for continuous target)
        # Uncomment if your target is continuous (e.g., prices, scores)
        W, b = linear_Regression_with_gradient(X, y, learning_rate=0.01, epochs=100, batch_size=32)
        evaluate_linear(X, y, W, b)
    else:
        # Option 2: LOGISTIC REGRESSION (for binary classification)
        # Uncomment if your target is 0/1 (e.g., pass/fail, yes/no)
        W, b = logistic_regression_gradient_descent(X, y, learning_rate=0.1, epochs=100, batch_size=32)
        evaluate_logistic(X, y, W, b)
    
    print(f"\nFinal Parameters:")
    print(f"  W shape: {W.shape}")
    print(f"  b: {b}")      
    
# extra practices
def load_and_preprocess(filename):
    """
    Load a dataset and perform basic preprocessing.
    
    Parameters:
    -----------
    filename : str
        Path to the CSV file
    
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix (all columns except last)
    y : numpy.ndarray
        Target vector (last column)
    """
    
    # TODO: Load the CSV file using pandas
    df=pd.read_csv(filename)
    
    
    print(f"Dataset loaded!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    # TODO: Check how many null values exist in each column
    print("\nNull values in each column:")
    print(df.isnull().sum())
    
    
    # TODO: Fill all null values with the mean of each column
    df=df.fillna(df.mean(axis=0))
    
    
    print("\nNull values after filling:")
    print(df.isnull().sum())
    
    # TODO: Extract all columns except the last one as features (X)
    # Convert to numpy array
    X=df.iloc[:,:-1].values
    
    
    # TODO: Extract the last column as target (y)
    # Convert to numpy array
    y=df.iloc[:,-1].values
    
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y


def normalize_features(X):
    """
    Normalize features to have mean=0 and std=1.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    
    Returns:
    --------
    X_normalized : numpy.ndarray
        Normalized feature matrix
    """
    
    print("\n--- Normalization ---")
    print(f"Before normalization:")
    print(f"Mean: {X.mean(axis=0)[:3]}...")  # Show first 3
    print(f"Std: {X.std(axis=0)[:3]}...")
    
    # TODO: Calculate the mean of each column (axis=0)
    mean=X.mean(axis=0)
    
    
    # TODO: Calculate the standard deviation of each column (axis=0)
    std=X.std(axis=0)
    
    
    # TODO: Normalize X using the formula: (X - mean) / (std + 1e-8)
    X_normalised=(X - mean) / (std + 1e-8)
    
    
    print(f"\nAfter normalization:")
    print(f"Mean: {X_normalized.mean(axis=0)[:3]}...")
    print(f"Std: {X_normalized.std(axis=0)[:3]}...")
    
    return X_normalized


def split_train_test(X, y, test_size=0.2):
    """
    Split data into train and test sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    test_size : float
        Proportion of data to use for testing
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
    """
    
    # TODO: Calculate the number of test samples
    # Hint: multiply total samples by test_size and convert to int
    n_test = int(len(X)*test_size)
    
    # TODO: Calculate the number of training samples
    n_train = len(X)-n_test
    
    print(f"\nSplitting data:")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    
    # Split the data (first n_train for training, rest for testing)
    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("="*50)
    print("PRACTICE PROBLEM 1: Basic Data Pipeline")
    print("="*50)
    
    # Load and preprocess
    X, y = load_and_preprocess("dataset.csv")
    
    # Normalize
    X_normalized = normalize_features(X)
    
    # Split
    X_train, X_test, y_train, y_test = split_train_test(X_normalized, y)
    
    print("\n" + "="*50)
    print("Pipeline Complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("="*50)

def load_data_with_info(filename):
    """
    Load data and display detailed information.
    
    Parameters:
    -----------
    filename : str
        Path to CSV file
    
    Returns:
    --------
    df : pandas.DataFrame
    """
    
    # TODO: Load the CSV file
    df=pd.read_csv(filename)
    
    
    print("Dataset Information:")
    print("-" * 40)
    
    # TODO: Print the shape of the dataframe
    print(f"Shape: {df.shape}")
    
    # TODO: Print the column names
    print(f"\nColumns: {df.columns.tolist()}")
    
    # TODO: Print data types of each column
    print(f"\nData types:{df.dtypes}")
    
    
    return df


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values using different strategies.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    strategy : str
        'mean', 'median', 'zero', or 'drop'
    
    Returns:
    --------
    df_clean : pandas.DataFrame
        Cleaned dataframe
    """
    
    print(f"\n--- Handling Missing Values (strategy: {strategy}) ---")
    
    # TODO: Count total number of missing values
    total_nulls = df.isnull().sum().sum()
    print(f"Total missing values: {total_nulls}")
    
    # TODO: Print missing values per column
    print(f"\nMissing values per column:{df.isnull().sum()}")
    
    
    if strategy == 'mean':
        # TODO: Fill nulls with mean of each column
        df_clean=df.fillna(df.mean(axis=0))
        
        
    elif strategy == 'median':
        # TODO: Fill nulls with median of each column
        df_clean=df.fillna(df.median(axis=0))
        
        
    elif strategy == 'zero':
        # TODO: Fill nulls with 0
        df_clean=df.fillna(0)
        
        
    elif strategy == 'drop':
        # TODO: Drop rows with any null values
        df_clean=df.dropna()
        
        
    
    # TODO: Verify no nulls remain (or count remaining if strategy is 'drop')
    print(f"\nMissing values after handling: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def extract_statistics(df):
    """
    Extract basic statistics from the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    
    Returns:
    --------
    stats : dict
        Dictionary containing statistics
    """
    
    print("\n--- Computing Statistics ---")
    
    # Convert to numpy for calculations
    data = df.values
    
    # TODO: Calculate mean of all values
    overall_mean = data.mean()
    
    # TODO: Calculate standard deviation of all values
    overall_std = data.std()
    
    # TODO: Calculate minimum value
    min_val = data.min()
    
    # TODO: Calculate maximum value
    max_val = data.max()
    
    stats = {
        'mean': overall_mean,
        'std': overall_std,
        'min': min_val,
        'max': max_val
    }
    
    print(f"Overall Mean: {overall_mean:.4f}")
    print(f"Overall Std: {overall_std:.4f}")
    print(f"Min value: {min_val:.4f}")
    print(f"Max value: {max_val:.4f}")
    
    return stats


def reshape_and_split(df):
    """
    Reshape data and split into features and target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    
    Returns:
    --------
    X, y : numpy.ndarray
    """
    
    print("\n--- Reshaping Data ---")
    
    # TODO: Select first 3 columns as features
    X = df.iloc[:,:3]
    
    # TODO: Select 4th column (index 3) as target
    y = df.iloc[:,3]
    
    # TODO: Convert X to numpy array
    X = X.values
    
    # TODO: Convert y to numpy array
    y = y.values
    
    # TODO: Reshape y to be a column vector (shape: n, 1)
    y = y.reshape(-1,1)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y


if __name__ == "__main__":
    print("="*50)
    print("PRACTICE PROBLEM 2: Missing Data & Operations")
    print("="*50)
    
    # Step 1: Load data
    df = load_data_with_info("data_with_nulls.csv")
    
    # Step 2: Handle missing values
    df_clean = handle_missing_values(df, strategy='mean')
    
    # Step 3: Extract statistics
    stats = extract_statistics(df_clean)
    
    # Step 4: Reshape and split
    X, y = reshape_and_split(df_clean)
    
    print("\n" + "="*50)
    print("All operations completed successfully!")
    print("="*50)


def load_and_slice(filename):
    """
    Load data and practice different slicing operations.
    
    Parameters:
    -----------
    filename : str
        Path to CSV file
    
    Returns:
    --------
    df : pandas.DataFrame
    """
    
    # TODO: Load CSV file
    df=pd.read_csv(filename)
    
    
    print("Original DataFrame:")
    print(f"Shape: {df.shape}")
    print(df.head())
    
    print("\n--- Slicing Operations ---")
    
    # TODO: Select first 10 rows
    first_10 = df.iloc[:10]
    print(f"\nFirst 10 rows shape: {first_10.shape}")
    
    # TODO: Select last 5 rows
    last_5 = df.iloc[-5:]
    print(f"Last 5 rows shape: {last_5.shape}")
    
    # TODO: Select rows 20 to 30
    middle_rows = df.iloc[20:30]
    print(f"Middle rows (20-30) shape: {middle_rows.shape}")
    
    # TODO: Select columns at index 0, 2, and 4
    selected_cols = df.iloc[:, [0, 2, 4]]
    print(f"Selected columns shape: {selected_cols.shape}")
    
    return df


def create_features_target(df):
    """
    Create feature matrix and target vector with proper shapes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    
    Returns:
    --------
    X, y : numpy.ndarray
    """
    
    print("\n--- Creating X and y ---")
    
    # TODO: Get all rows, all columns except last → convert to numpy
    X = df.iloc[:,:-1].values
    
    # TODO: Get all rows, only last column → convert to numpy
    y = df.iloc[:,-1].values
    
    print(f"X shape before reshape: {X.shape}")
    print(f"y shape before reshape: {y.shape}")
    
    # TODO: Reshape y to column vector (n_samples, 1)
    y = y.reshape(-1,1)
    
    print(f"y shape after reshape: {y.shape}")
    
    return X, y


def normalize_data(X):
    """
    Normalize features using standardization.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    
    Returns:
    --------
    X_norm : numpy.ndarray
        Normalized features
    """
    
    print("\n--- Normalization ---")
    
    # TODO: Compute mean along axis=0 (column-wise)
    mean = X.mean(axis=0)
    
    # TODO: Compute std along axis=0 (column-wise)
    std = X.std(axis=0)
    
    print(f"Original mean (first 3 features): {mean[:3]}")
    print(f"Original std (first 3 features): {std[:3]}")
    
    # TODO: Normalize using (X - mean) / (std + 1e-8)
    X_norm = (X - mean) / (std + 1e-8)
    
    print(f"\nNormalized mean: {X_norm.mean(axis=0)[:3]}")
    print(f"Normalized std: {X_norm.std(axis=0)[:3]}")
    
    return X_norm


def process_in_batches(X, y, batch_size=32):
    """
    Process data in batches and compute statistics.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    batch_size : int
        Size of each batch
    
    Returns:
    --------
    batch_stats : list
        Statistics for each batch
    """
    
    print(f"\n--- Processing in Batches (batch_size={batch_size}) ---")
    
    # TODO: Get the number of samples
    n_samples = X.shape[0]
    
    batch_stats = []
    
    # TODO: Loop through data in batches
    # Hint: range(start, stop, step)
    for i in range(0,n_samples,batch_size):
        
        # TODO: Get batch of X (rows i to i+batch_size)
        X_batch = X[i:i+batch_size]
        
        # TODO: Get batch of y (rows i to i+batch_size)
        y_batch = y[i:i+batch_size]
        
        # TODO: Calculate mean of this batch
        batch_mean = X_batch.mean()
        
        # TODO: Calculate sum of this batch
        batch_sum = X_batch.sum()
        
        batch_stats.append({
            'batch_num': i // batch_size + 1,
            'size': X_batch.shape[0],
            'mean': batch_mean,
            'sum': batch_sum
        })
        
        print(f"Batch {i//batch_size + 1}: size={X_batch.shape[0]}, mean={batch_mean:.4f}")
    
    return batch_stats


def stack_arrays():
    """
    Practice stacking arrays vertically and horizontally.
    """
    
    print("\n--- Array Stacking ---")
    
    # Create sample arrays
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[7, 8, 9], [10, 11, 12]])
    
    print(f"arr1 shape: {arr1.shape}")
    print(f"arr2 shape: {arr2.shape}")
    
    # TODO: Stack arr1 and arr2 vertically (rows)
    # Hint: np.vstack()
    v_stacked = np.vstack(arr1,arr2)
    
    print(f"\nVertically stacked shape: {v_stacked.shape}")
    print(v_stacked)
    
    # Create arrays for horizontal stacking
    arr3 = np.array([[1, 2], [3, 4], [5, 6]])
    arr4 = np.array([[7, 8], [9, 10], [11, 12]])
    
    # TODO: Stack arr3 and arr4 horizontally (columns)
    # Hint: np.hstack()
    h_stacked = np.hstack(arr3,arr4)
    
    print(f"\nHorizontally stacked shape: {h_stacked.shape}")
    print(h_stacked)


if __name__ == "__main__":
    print("="*60)
    print("PRACTICE PROBLEM 3: Slicing & Batch Processing")
    print("="*60)
    
    # Step 1: Load and slice
    df = load_and_slice("data.csv")
    
    # Step 2: Create X and y
    X, y = create_features_target(df)
    
    # Step 3: Normalize
    X_norm = normalize_data(X)
    
    # Step 4: Process in batches
    stats = process_in_batches(X_norm, y, batch_size=32)
    
    # Step 5: Array stacking
    stack_arrays()
    
    print("\n" + "="*60)
    print("All operations completed!")
    print("="*60)

def load_dataset(filename):
    """
    Load dataset and perform initial inspection.
    
    Parameters:
    -----------
    filename : str
        CSV file path
    
    Returns:
    --------
    df : pandas.DataFrame
    """
    
    print("STEP 1: Loading Data")
    print("-" * 50)
    
    # TODO: Read CSV file
    df=pd.read_csv(filename)
    
    
    # TODO: Print dataset shape
    print(f"Dataset shape: {df.shape}")
    
    
    # TODO: Print first 5 rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    
    # TODO: Print information about columns and data types
    print("\nDataset info:")
    print(df.info())
    
    
    return df


def clean_data(df):
    """
    Clean the dataset by handling missing values and duplicates.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    
    Returns:
    --------
    df_clean : pandas.DataFrame
    """
    
    print("\n\nSTEP 2: Data Cleaning")
    print("-" * 50)
    
    # Check for nulls
    # TODO: Count null values in each column
    print("Null values per column:")
    print(df.isnull().sum())
    
    
    # TODO: Get total number of null values
    total_nulls = df.isnull().sum().sum()
    print(f"\nTotal null values: {total_nulls}")
    
    # TODO: Fill null values with column mean
    df_clean=df.fillna(df.mean(axis=0))
    
    
    # Check for duplicates
    # TODO: Count duplicate rows
    n_duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {n_duplicates}")
    
    # TODO: Remove duplicate rows if any exist
    df_clean=df.drop_duplicates()
    if n_duplicates > 0:
        
        print(f"Removed {n_duplicates} duplicate rows")
    
    # TODO: Verify cleaning
    print(f"\nAfter cleaning:")
    print(f"Shape: {df_clean.shape}")
    print(f"Nulls: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def prepare_features(df):
    """
    Prepare feature matrix and target vector.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    """
    
    print("\n\nSTEP 3: Feature Preparation")
    print("-" * 50)
    
    # TODO: Extract all columns except last as X
    X=df.iloc[:,:-1]
    
    
    # TODO: Extract last column as y
    y=df.iloc[:,-1]
    
    
    # TODO: Convert X to numpy array
    X = X.values
    
    # TODO: Convert y to numpy array and reshape to (n, 1)
    y = y.values
    
    print(f"Feature matrix X: {X.shape}")
    print(f"Target vector y: {y.shape}")
    
    return X, y


def normalize_features(X):
    """
    Normalize features to zero mean and unit variance.
    
    Parameters:
    -----------
    X : numpy.ndarray
    
    Returns:
    --------
    X_normalized : numpy.ndarray
    mean : numpy.ndarray
    std : numpy.ndarray
    """
    
    print("\n\nSTEP 4: Feature Normalization")
    print("-" * 50)
    
    # TODO: Calculate mean of each column
    mean = X.mean(axis=0)
    
    # TODO: Calculate std of each column
    std = X.std(axis=0)
    
    print("Before normalization:")
    print(f"Mean (first 3 features): {mean[:3]}")
    print(f"Std (first 3 features): {std[:3]}")
    
    # TODO: Normalize X
    X_normalized = (X-mean)/(std+1e-8)
    
    print("\nAfter normalization:")
    print(f"Mean (first 3 features): {X_normalized.mean(axis=0)[:3]}")
    print(f"Std (first 3 features): {X_normalized.std(axis=0)[:3]}")
    
    return X_normalized, mean, std


def initialize_parameters(n_features):
    """
    Initialize model parameters (weights and bias).
    
    Parameters:
    -----------
    n_features : int
        Number of features
    
    Returns:
    --------
    W : numpy.ndarray
        Weight matrix (n_features, 1)
    b : float
        Bias term
    """
    
    print("\n\nSTEP 5: Initialize Parameters")
    print("-" * 50)
    
    # TODO: Create weight matrix of zeros with shape (n_features, 1)
    W = np.zeros((n_features,1))
    
    # TODO: Initialize bias as 0.0
    b = 0.0
    
    print(f"Weights W: shape {W.shape}")
    print(f"Bias b: {b}")
    
    return W, b


def forward_pass_batched(X, y, W, b, batch_size=32):
    """
    Perform forward pass in batches and calculate predictions.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Normalized features
    y : numpy.ndarray
        Target values
    W : numpy.ndarray
        Weights
    b : float
        Bias
    batch_size : int
        Batch size
    
    Returns:
    --------
    predictions : numpy.ndarray
        All predictions
    """
    
    print("\n\nSTEP 6: Forward Pass (Batched)")
    print("-" * 50)
    
    # TODO: Get number of samples
    n_samples = X.shape[0]
    
    all_predictions = []
    
    # TODO: Loop through batches
    for i in range(0, n_samples, batch_size):
        
        # TODO: Get batch of X
        X_batch = X[i:i+batch_size]
        
        # TODO: Compute predictions: X_batch @ W + b
        predictions = X_batch@W+b
        
        # TODO: Append predictions to list
        all_predictions.append(predictions)
        
        
        print(f"Batch {i//batch_size + 1}: processed {X_batch.shape[0]} samples")
    
    # TODO: Stack all predictions vertically
    all_predictions = np.vstack(all_predictions)
    
    print(f"\nTotal predictions: {all_predictions.shape}")
    
    return all_predictions


def calculate_accuracy(predictions, y):
    """
    Calculate classification accuracy.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Model predictions
    y : numpy.ndarray
        True labels
    
    Returns:
    --------
    accuracy : float
        Classification accuracy
    """
    
    print("\n\nSTEP 7: Calculate Accuracy")
    print("-" * 50)
    
    # TODO: Convert predictions to binary (0 or 1)
    # If prediction >= 0.5, class = 1, else class = 0
    y_pred = (predictions>=0.5).astype(int)
    
    # TODO: Calculate accuracy (mean of correct predictions)
    accuracy = np.mean(y_pred==y)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


if __name__ == "__main__":
    print("="*50)
    print("PRACTICE PROBLEM 4: Complete ML Pipeline")
    print("="*50)
    
    # Complete pipeline
    df = load_dataset("ml_data.csv")
    df_clean = clean_data(df)
    X, y = prepare_features(df_clean)
    X_norm, mean, std = normalize_features(X)
    W, b = initialize_parameters(X.shape[1])
    predictions = forward_pass_batched(X_norm, y, W, b, batch_size=32)
    accuracy = calculate_accuracy(predictions, y)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)
    print(f"Final dataset shape: {X_norm.shape}")
    print(f"Model accuracy: {accuracy:.4f}")


"""
PRACTICE PROBLEM 5: Common Exam Scenarios
This file contains typical exam patterns you'll encounter
"""

# ============================================================
# SCENARIO 1: Quick Data Loading and Inspection
# ============================================================

def scenario_1():
    """Load CSV, check shape, display sample, and count nulls"""
    
    print("SCENARIO 1: Quick Data Check")
    print("="*50)
    
    # TODO: Load 'train.csv'
    df=pd.read_csv("train.csv")
    
    
    # TODO: Print shape (rows, columns)
    print(f"Shape: ")
    print(f"{df.shape}")
    
    # TODO: Show first 3 rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    
    # TODO: Count nulls in each column
    print("\nNull counts:")
    print(df.isnull().sum().sum())
    
    
    return df


# ============================================================
# SCENARIO 2: Fill Nulls with Different Strategies
# ============================================================

def scenario_2(df):
    """Practice different null-filling strategies"""
    
    print("\n\nSCENARIO 2: Handle Missing Data")
    print("="*50)
    
    # TODO: Fill nulls with 0
    df_zeros = df.fillna(0)
    print(f"After filling with 0: {df_zeros.isnull().sum().sum()} nulls")
    
    # TODO: Fill nulls with column mean
    df_mean = df.fillna(df.mean(axis=0))
    print(f"After filling with mean: {df_mean.isnull().sum().sum()} nulls")
    
    # TODO: Drop rows with any nulls
    df_dropped = df.dropna()
    print(f"After dropping nulls: {df_dropped.shape[0]} rows remain")
    
    return df_mean  # Return the mean-filled version


# ============================================================
# SCENARIO 3: Extract X and y Correctly
# ============================================================

def scenario_3(df):
    """Extract features and target in different ways"""
    
    print("\n\nSCENARIO 3: Feature & Target Extraction")
    print("="*50)
    
    # Method 1: All except last column as X
    # TODO: Get all columns except last, convert to numpy
    X1 = df.iloc[:,:-1].values
    
    # TODO: Get last column, convert to numpy
    y1 = df.iloc[:,-1].values
    
    print(f"Method 1 - X: {X1.shape}, y: {y1.shape}")
    
    # Method 2: Specific columns as features
    # TODO: Get columns 0, 1, 2 as features
    X2 = df.iloc[:, [0, 1, 2]].values
    
    # TODO: Get column 3 as target
    y2 = df.iloc[:,3].values
    
    print(f"Method 2 - X: {X2.shape}, y: {y2.shape}")
    
    # Method 3: First N-1 columns
    n_cols = df.shape[1]
    # TODO: Get first n_cols-1 columns
    X3 = df.iloc[:,:n_cols-1].values
    y3 = df.iloc[:, n_cols-1].values
    
    print(f"Method 3 - X: {X3.shape}, y: {y3.shape}")
    
    return X1, y1


# ============================================================
# SCENARIO 5: Array Reshaping
# ============================================================

def scenario_5():
    """Practice reshaping arrays"""
    
    print("\n\nSCENARIO 5: Array Reshaping")
    print("="*50)
    
    # Create sample array
    y = np.array([1, 2, 3, 4, 5])
    print(f"Original y shape: {y.shape}")
    
    # TODO: Reshape to column vector (5, 1)
    y_col = y.reshape(5,1)
    print(f"Column vector shape: {y_col.shape}")
    
    # TODO: Reshape to row vector (1, 5)
    y_row = y.reshape(1,5)
    print(f"Row vector shape: {y_row.shape}")
    
    # Create 1D array
    arr = np.arange(12)
    print(f"\nOriginal array shape: {arr.shape}")
    
    # TODO: Reshape to 3x4
    arr_2d = arr.reshape(3,4)
    print(f"Reshaped to 3x4: {arr_2d.shape}")
    
    # TODO: Reshape to 4x3
    arr_alt = arr.reshape(4,3)
    print(f"Reshaped to 4x3: {arr_alt.shape}")
    
    return y_col



# ============================================================
# SCENARIO 7: Matrix Operations
# ============================================================

def scenario_7():
    """Practice matrix multiplication and operations"""
    
    print("\n\nSCENARIO 7: Matrix Operations")
    print("="*50)
    
    # Create sample data
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    
    # TODO: Create weight matrix with shape (5, 1)
    W = np.zeros((5,1))
    
    # TODO: Initialize bias
    b = 0.0
    
    print(f"X shape: {X.shape}")
    print(f"W shape: {W.shape}")
    
    # TODO: Perform matrix multiplication: X @ W + b
    predictions = X@W+b
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Element-wise operations
    # TODO: Add 10 to all predictions
    pred_plus_10 = predictions+10
    
    # TODO: Square all predictions
    pred_squared = predictions**2
    
    print(f"All operations completed!")
    
    return predictions


# ============================================================
# SCENARIO 8: Calculating Accuracy
# ============================================================

def scenario_8():
    """Practice accuracy calculation"""
    
    print("\n\nSCENARIO 8: Accuracy Calculation")
    print("="*50)
    
    # Create sample predictions and true labels
    predictions = np.array([[0.2], [0.7], [0.8], [0.3], [0.6]])
    y_true = np.array([[0], [1], [1], [0], [1]])
    
    print("Predictions (continuous):", predictions.T)
    print("True labels:", y_true.T)
    
    # TODO: Convert predictions to binary (threshold = 0.5)
    y_pred = (predictions>=0.5).astype(int)
    
    print("Predictions (binary):", y_pred.T)
    
    # TODO: Calculate accuracy
    accuracy = np.mean(y_pred==y_true)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # TODO: Count correct predictions
    correct = np.sum(y_pred==y_true)
    print(f"Correct predictions: {correct} out of {len(y_true)}")
    
    return accuracy


# ============================================================
# SCENARIO 9: Stacking Arrays
# ============================================================

def scenario_9():
    """Practice stacking multiple arrays"""
    
    print("\n\nSCENARIO 9: Array Stacking")
    print("="*50)
    
    # Create sample arrays
    arr1 = np.array([[1, 2]])
    arr2 = np.array([[3, 4]])
    arr3 = np.array([[5, 6]])
    
    print(f"arr1: {arr1.shape}, arr2: {arr2.shape}, arr3: {arr3.shape}")
    
    # TODO: Stack vertically
    v_stack = np.vstack([arr1,arr2,arr3])
    
    print(f"Vertically stacked: {v_stack.shape}")
    print(v_stack)
    
    # TODO: Create list of arrays to stack
    arrays = [arr1, arr2, arr3]
    
    # TODO: Stack all at once
    combined = np.vstack(arrays)
    
    print(f"Combined: {combined.shape}")
    
    return combined


# ============================================================
# MAIN: Run All Scenarios
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PRACTICE PROBLEM 5: Common Exam Scenarios")
    print("Complete all TODO items in each scenario!")
    print("="*60 + "\n")
    
    # Run all scenarios
    try:
        df = scenario_1()
        df_clean = scenario_2(df)
        X, y = scenario_3(df_clean)
        y_reshaped = scenario_5()
        predictions = scenario_7()
        accuracy = scenario_8()
        combined = scenario_9()
        
        print("\n" + "="*60)
        print("ALL SCENARIOS COMPLETED! 🎉")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error encountered: {e}")
        print("Complete the TODO items to fix the errors!")  



"""
ADVANCED PROBLEM 1: Multi-File Data Processing
Load multiple CSV files, merge them, and process together
"""

def load_multiple_files(filenames):
    """
    Load multiple CSV files and return as a list of DataFrames.
    
    Parameters:
    -----------
    filenames : list of str
        List of CSV filenames to load
    
    Returns:
    --------
    dataframes : list of pandas.DataFrame
    """
    
    print("="*60)
    print("STEP 1: Loading Multiple Files")
    print("="*60)
    
    dataframes = []
    
    # TODO: Loop through filenames and load each CSV
    for filename in filenames:
        # TODO: Load the CSV file
        df=pd.read_csv(filename)
        
        
        # TODO: Append to dataframes list
        dataframes.append(df)
        
        
        print(f"Loaded {filename}: shape {df.shape}")
    
    return dataframes


def merge_dataframes(dataframes):
    """
    Merge multiple DataFrames vertically (stack rows).
    
    Parameters:
    -----------
    dataframes : list of pandas.DataFrame
    
    Returns:
    --------
    merged_df : pandas.DataFrame
    """
    
    print("\n" + "="*60)
    print("STEP 2: Merging DataFrames")
    print("="*60)
    
    # TODO: Use pd.concat to merge all dataframes vertically
    # Hint: pd.concat(list_of_dfs, axis=0)
    merged_df = pd.concat(dataframes,axis=0)
    
    # TODO: Reset the index after merging
    merged_df = merged_df.reset_index(drop=True)
    
    print(f"Merged DataFrame shape: {merged_df.shape}")
    
    # TODO: Check for and remove duplicate rows
    n_duplicates = merged_df.duplicated().sum()
    print(f"Duplicate rows found: {n_duplicates}")
    
    if n_duplicates > 0:
        # TODO: Drop duplicates
        merged_df=merged_df.drop_duplicates()
        
        print(f"After removing duplicates: {merged_df.shape}")
    
    return merged_df


def handle_mixed_nulls(df):
    """
    Handle nulls using different strategies for different columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    
    Returns:
    --------
    df_clean : pandas.DataFrame
    """
    
    print("\n" + "="*60)
    print("STEP 3: Smart Null Handling")
    print("="*60)
    
    df_clean = df.copy()
    
    # TODO: Count nulls per column
    null_counts = df_clean.isnull().sum()
    print("Null counts per column:")
    print(null_counts)
    
    # Strategy: Fill numeric columns with median, categorical with mode
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            # TODO: Check if column is numeric
            if df_clean[col].dtype in ['float64', 'int64']:
                # TODO: Fill with median
                df_clean[col]=df_clean[col].fillna(df_clean[col].median())
                
                print(f"{col}: filled with median")
            else:
                # TODO: Fill with mode (most frequent value)
                # Hint: df[col].mode()[0]
                df_clean[col]=df_clean[col].fillna(df_clean[col].mode()[0])
                
                print(f"{col}: filled with mode")
    
    # TODO: Verify no nulls remain
    print(f"\nTotal nulls remaining: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def feature_engineering(df):
    """
    Create new features from existing ones.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    
    Returns:
    --------
    df_enhanced : pandas.DataFrame
    """
    
    print("\n" + "="*60)
    print("STEP 4: Feature Engineering")
    print("="*60)
    
    df_enhanced = df.copy()
    
    # Assume df has columns: ['age', 'income', 'expenses', 'target']
    
    # TODO: Create 'savings' feature = income - expenses
    df_enhanced['savings']=df_enhanced['income']-df_enhanced['expenses']
    
    
    # TODO: Create 'savings_rate' feature = savings / income
    # Add small epsilon to avoid division by zero
    df_enhanced['savings_rate']=df_enhanced['savings']/(df_enhanced['income']+1e-8)
    
    
    # TODO: Create 'age_group' feature: 
    # 0 if age < 30, 1 if 30 <= age < 50, 2 if age >= 50
    # Hint: Use np.where or pd.cut
    df_enhanced['age_group'] = np.where(df_enhanced['age'] < 30, 0, np.where(df_enhanced['age'] < 50, 1, 2))
    
    print(f"Original features: {df.shape[1]}")
    print(f"Enhanced features: {df_enhanced.shape[1]}")
    print(f"New columns: {df_enhanced.columns.tolist()}")
    
    return df_enhanced


def advanced_preprocessing(df):
    """
    Advanced preprocessing with multiple steps.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    
    Returns:
    --------
    X_processed : numpy.ndarray
    y : numpy.ndarray
    feature_names : list
    """
    
    print("\n" + "="*60)
    print("STEP 5: Advanced Preprocessing")
    print("="*60)
    
    # TODO: Separate features and target
    # Assume last column is target
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    # TODO: Store feature names for later
    feature_names = X.columns.tolist()
    
    # TODO: Convert to numpy
    X = X.values
    y = y.values
    
    print(f"Feature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    
    # TODO: Remove outliers using IQR method
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(X,25,axis=0)
    Q3 = np.percentile(X,75,axis=0)
    IQR = Q3 - Q1
    
    # TODO: Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # TODO: Create mask for non-outliers (all features within bounds)
    # Hint: Check if all features are within bounds for each sample
    mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
    
    # TODO: Filter X and y using mask
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"Samples removed as outliers: {len(X) - len(X_clean)}")
    print(f"Remaining samples: {X_clean.shape[0]}")
    
    # TODO: Apply log transformation to positive features
    # Add 1 to handle zeros: log(x + 1)
    # Only apply to columns where all values are >= 0
    for i in range(X_clean.shape[1]):
        if np.all(X_clean[:, i] >= 0):
            # TODO: Apply log transformation
            X_clean[:,i]=np.log(X_clean[:,i]+1)
            
            print(f"Applied log transform to feature {i}")
    
    # TODO: Standardize features
    mean = X_clean.mean(axis=0)
    std = X_clean.std(axis=0)
    X_processed = (X_clean-mean)/(std+1e-8)
    
    print(f"Final processed shape: {X_processed.shape}")
    
    return X_processed, y_clean, feature_names


def create_polynomial_features(X, degree=2):
    """
    Create polynomial features (x^2, x1*x2, etc.).
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix (n_samples, n_features)
    degree : int
        Polynomial degree
    
    Returns:
    --------
    X_poly : numpy.ndarray
        Extended feature matrix with polynomial features
    """
    
    print("\n" + "="*60)
    print("STEP 6: Polynomial Features")
    print("="*60)
    
    n_samples, n_features = X.shape
    
    # Start with original features
    X_poly = X.copy()
    
    if degree >= 2:
        # TODO: Add squared features (x^2)
        # Hint: X ** 2
        X_squared = X**2
        
        # TODO: Stack horizontally with original features
        X_poly = np.hstack([X_poly,X_squared])
        
        print(f"Added {n_features} squared features")
        
        # TODO: Add interaction features (x1 * x2)
        # interaction=X[:i]*X[:,j]
        # For each pair of features
        interactions = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                # TODO: Multiply feature i with feature j
                interaction = interaction=X[:i]*X[:,j]
                
                # TODO: Reshape to column and append
                interactions.append(interaction.reshape(-1, 1))
        
        if interactions:
            # TODO: Stack all interactions horizontally
            interaction_matrix = np.hstack([X_poly,X_squared])
            
            # TODO: Add to X_poly
            X_poly = np.hstack([X_poly,interaction_matrix])
            
            print(f"Added {len(interactions)} interaction features")
    
    print(f"Original features: {n_features}")
    print(f"Polynomial features: {X_poly.shape[1]}")
    
    return X_poly


def stratified_split(X, y, test_size=0.2):
    """
    Split data maintaining class distribution (stratified split).
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    test_size : float
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    
    print("\n" + "="*60)
    print("STEP 7: Stratified Train-Test Split")
    print("="*60)
    
    # TODO: Get unique classes in y
    unique_classes = np.unique(y)
    
    print(f"Classes found: {unique_classes}")
    
    # TODO: Calculate class distribution
    for cls in unique_classes:
        
        count = cls.count()
        percentage = (count/len(unique_classes))*100
        print(f"Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Initialize lists for train and test
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    
    # TODO: For each class, split proportionally
    for cls in unique_classes:
        # TODO: Get indices where y equals cls
        indices = np.where(y==cls)[0]
        
        # TODO: Get X and y for this class
        X_cls = X[indices]
        y_cls = y[indices]
        
        # TODO: Calculate split point
        n_test = int(len(X_cls)*test_size)
        
        # TODO: Split this class
        X_train_list.append(X_cls[:len(X_cls)-n_test])
        X_test_list.append(X_cls[len(X_cls)-n_test:])
        y_train_list.append(y_cls[:len(y_cls)-n_test])
        y_test_list.append(y_cls[len(y_cls)-n_test:])
    
    # TODO: Concatenate all classes
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Verify stratification
    print("\nTrain set class distribution:")
    for cls in unique_classes:
        count = np.sum(y_train == cls)
        percentage = (count / len(y_train)) * 100
        print(f"Class {cls}: {count} ({percentage:.1f}%)")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED PROBLEM 1: Multi-File Processing Pipeline")
    print("="*60 + "\n")
    
    # Simulate multiple files
    filenames = ['data_part1.csv', 'data_part2.csv', 'data_part3.csv']
    
    try:
        # Step 1: Load multiple files
        dataframes = load_multiple_files(filenames)
        
        # Step 2: Merge dataframes
        merged_df = merge_dataframes(dataframes)
        
        # Step 3: Handle nulls smartly
        df_clean = handle_mixed_nulls(merged_df)
        
        # Step 4: Feature engineering
        df_enhanced = feature_engineering(df_clean)
        
        # Step 5: Advanced preprocessing
        X, y, feature_names = advanced_preprocessing(df_enhanced)
        
        # Step 6: Create polynomial features
        X_poly = create_polynomial_features(X, degree=2)
        
        # Step 7: Stratified split
        X_train, X_test, y_train, y_test = stratified_split(X_poly, y)
        
        print("\n" + "="*60)
        print("ADVANCED PIPELINE COMPLETED!")
        print("="*60)
        print(f"Final training data: {X_train.shape}")
        print(f"Final test data: {X_test.shape}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Complete all TODO items!")


"""
ADVANCED PROBLEM 2: Statistical Analysis & Cross-Validation
Perform detailed statistical analysis and implement k-fold cross-validation
"""

def load_and_analyze(filename):
    """
    Load data and perform comprehensive statistical analysis.
    
    Parameters:
    -----------
    filename : str
        CSV file path
    
    Returns:
    --------
    df : pandas.DataFrame
    stats_report : dict
    """
    
    print("="*60)
    print("STEP 1: Load and Statistical Analysis")
    print("="*60)
    
    # TODO: Load CSV
    df=pd.read_csv(filename)
    
    
    # TODO: Convert to numpy for analysis
    data = df.values
    
    stats_report = {}
    
    # TODO: Calculate basic statistics
    stats_report['mean'] = data.mean()
    stats_report['median'] = np.median(data)
    stats_report['std'] = data.std()
    stats_report['var'] = data.var()
    stats_report['min'] = data.min()
    stats_report['max'] = data.max()
    
    # TODO: Calculate quartiles (25th, 50th, 75th percentiles)
    stats_report['q25'] = np.percentile(data,25)
    stats_report['q50'] = np.percentile(data,50) 
    stats_report['q75'] = np.percentile(data,75) 
    
    # TODO: Calculate range and IQR
    stats_report['range'] = stats_report['max'] - stats_report['min']
    stats_report['iqr'] = stats_report['q75']-stats_report['q25']
    
    # TODO: Calculate coefficient of variation (std / mean * 100)
    stats_report['cv'] = (stats_report['std']/stats_report['mean'])*100
    
    print("\nStatistical Summary:")
    for key, value in stats_report.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value[:3]}...")  # Show first 3 values
        else:
            print(f"{key}: {value:.4f}")
    
    return df, stats_report


def detect_outliers_multiple_methods(X):
    """
    Detect outliers using multiple methods and compare results.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    
    Returns:
    --------
    outlier_report : dict
        Report of outliers detected by each method
    """
    
    print("\n" + "="*60)
    print("STEP 2: Multi-Method Outlier Detection")
    print("="*60)
    
    n_samples, n_features = X.shape
    outlier_report = {}
    
    # Method 1: IQR Method
    # TODO: Calculate Q1 and Q3 for each feature
    Q1 = np.percentile(X,25,axis=0)
    Q3 = np.percentile(X,75,axis=0)
    IQR = Q3 - Q1
    
    # TODO: Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # TODO: Find outliers (samples with ANY feature outside bounds)
    iqr_outliers = np.any((X<lower_bound)|(X>upper_bound))
    outlier_report['iqr'] = np.sum(iqr_outliers)
    
    # Method 2: Z-score Method (|z-score| > 3)
    # TODO: Calculate z-scores
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    z_scores = np.abs((X-mean)/(std+1e-8))
    
    # TODO: Find outliers (|z-score| > 3 for any feature)
    z_outliers = np.any((np.abs(z_scores))>3,axis=1)
    outlier_report['zscore'] = np.sum(z_outliers)
    
    # Method 3: Percentile Method (beyond 1st or 99th percentile)
    # TODO: Calculate 1st and 99th percentiles
    p1 = np.percentile(X,1,axis=0)
    p99 = np.percentile(X,99,axis=0)
    
    # TODO: Find outliers
    percentile_outliers = np.any((X<p1)|(X>p99))
    outlier_report['percentile'] = np.sum(percentile_outliers)
    
    print(f"\nOutlier Detection Results (out of {n_samples} samples):")
    print(f"IQR Method: {outlier_report['iqr']} outliers")
    print(f"Z-Score Method: {outlier_report['zscore']} outliers")
    print(f"Percentile Method: {outlier_report['percentile']} outliers")
    
    # TODO: Find consensus outliers (detected by at least 2 methods)
    consensus_outliers = (iqr_outliers.astype(int) + 
                         z_outliers.astype(int) + 
                         percentile_outliers.astype(int)) >= 2
    
    outlier_report['consensus'] = np.sum(consensus_outliers)
    print(f"Consensus (2+ methods): {outlier_report['consensus']} outliers")
    
    return outlier_report, consensus_outliers


def correlation_analysis(X, feature_names):
    """
    Analyze correlations between features.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    feature_names : list
        Names of features
    
    Returns:
    --------
    corr_matrix : numpy.ndarray
        Correlation matrix
    high_corr_pairs : list
        Pairs of highly correlated features
    """
    
    print("\n" + "="*60)
    print("STEP 3: Correlation Analysis")
    print("="*60)
    
    n_features = X.shape[1]
    
    # TODO: Calculate correlation matrix
    # Hint: For each pair of features, calculate correlation
    # Correlation = covariance(X, Y) / (std(X) * std(Y))
    
    corr_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # TODO: Calculate correlation between feature i and j
                # Hint: np.corrcoef(X[:, i], X[:, j])[0, 1]
                corr_matrix[i, j] =  np.corrcoef(X[:, i], X[:, j])[0, 1]
    
    print("\nCorrelation Matrix (first 3x3):")
    print(corr_matrix[:3, :3])
    
    # TODO: Find highly correlated pairs (|correlation| > 0.8, excluding diagonal)
    high_corr_pairs = []
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            # TODO: Check if absolute correlation > 0.8
            if np.abs(corr_matrix[i,j])>0.8 :
                high_corr_pairs.append({
                    'feature1': feature_names[i] if feature_names else f'F{i}',
                    'feature2': feature_names[j] if feature_names else f'F{j}',
                    'correlation': corr_matrix[i, j]
                })
    
    print(f"\nHighly correlated pairs (|r| > 0.8): {len(high_corr_pairs)}")
    for pair in high_corr_pairs[:5]:  # Show first 5
        print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.4f}")
    
    return corr_matrix, high_corr_pairs


def create_k_folds(X, y, k=5):
    """
    Split data into k folds for cross-validation.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    k : int
        Number of folds
    
    Returns:
    --------
    folds : list of tuples
        List of (train_indices, test_indices) for each fold
    """
    
    print("\n" + "="*60)
    print(f"STEP 4: Create {k}-Fold Cross-Validation Splits")
    print("="*60)
    
    # TODO: Get number of samples
    n_samples = X.shape[0]
    
    # TODO: Create array of indices
    indices = np.arange(n_samples)
    
    # TODO: Shuffle indices randomly
    np.random.shuffle(indices)
    
    # TODO: Calculate fold size
    fold_size = n_samples//k
    
    folds = []
    
    # TODO: Create k folds
    for i in range(k):
        # TODO: Calculate start and end indices for test set
        test_start = i*fold_size
        test_end = (i+1)*fold_size if i<k-1 else n_samples
        
        # TODO: Get test indices for this fold
        test_indices = indices[test_start:test_end]
        
        # TODO: Get train indices (all except test)
        train_indices = np.concatenate(indices[:test_start],indices[test_end:])
        
        folds.append((train_indices, test_indices))
        
        print(f"Fold {i+1}: Train={len(train_indices)}, Test={len(test_indices)}")
    
    return folds


def cross_validate(X, y, folds, batch_size=32):
    """
    Perform k-fold cross-validation with batched inference.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    folds : list
        List of (train_indices, test_indices)
    batch_size : int
    
    Returns:
    --------
    cv_results : dict
        Cross-validation results
    """
    
    print("\n" + "="*60)
    print("STEP 5: Cross-Validation with Batched Inference")
    print("="*60)
    
    n_features = X.shape[1]
    fold_scores = []
    
    # TODO: Loop through each fold
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        
        print(f"\n--- Fold {fold_idx + 1} ---")
        
        # TODO: Split data into train and test
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # TODO: Normalize based on training data
        train_mean = np.mean(X_train,axis=0)
        train_std = np.std(X_train,axis=0)
        
        X_train_norm = (X_train-train_mean)/(train_std+1e-8)
        X_test_norm = (X_test-train_mean)/(train_std+1e-8)
        
        # TODO: Initialize parameters
        W = np.zeros((n_features,1))
        b = 0.0
        
        # TODO: Batched inference on test set
        n_test = X_test_norm.shape[0]
        predictions = []
        
        for i in range(0, n_test, batch_size):
            # TODO: Get batch
            X_batch = X_test_norm[i:i+batch_size]
            
            # TODO: Forward pass
            pred = X_batch@W+b
            
            predictions.append(pred)
        
        # TODO: Combine predictions
        predictions = np.vstack(predictions)
        
        # TODO: Convert to binary and calculate accuracy
        y_pred = (predictions>=0.5).astype(int)
        accuracy = np.mean(y_pred==y_test.reshape(-1,1))
        
        fold_scores.append(accuracy)
        print(f"Fold {fold_idx + 1} Accuracy: {accuracy:.4f}")
    
    # TODO: Calculate cross-validation statistics
    cv_results = {
        'scores': np.array(fold_scores),
        'mean': np.mean(fold_scores),
        'std': np.std(fold_scores),
        'min': np.min(fold_scores),
        'max': np.max(fold_scores)
    }
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    print(f"Mean Accuracy: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
    print(f"Min Accuracy: {cv_results['min']:.4f}")
    print(f"Max Accuracy: {cv_results['max']:.4f}")
    print("="*60)
    
    return cv_results


def bootstrap_sampling(X, y, n_bootstrap=100, sample_size_ratio=1.0):
    """
    Perform bootstrap sampling for uncertainty estimation.
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    n_bootstrap : int
        Number of bootstrap samples
    sample_size_ratio : float
        Ratio of samples to draw (1.0 = same size as original)
    
    Returns:
    --------
    bootstrap_stats : dict
    """
    
    print("\n" + "="*60)
    print(f"STEP 6: Bootstrap Sampling (n={n_bootstrap})")
    print("="*60)
    
    n_samples = X.shape[0]
    # TODO: Calculate bootstrap sample size
    bootstrap_size = int(n_samples*sample_size_ratio)
    
    bootstrap_means = []
    bootstrap_stds = []
    
    # TODO: Perform bootstrap sampling
    for i in range(n_bootstrap):
        # TODO: Sample with replacement
        # Hint: np.random.choice(n_samples, size=bootstrap_size, replace=True)
        indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
        
        # TODO: Get bootstrap sample
        X_boot = X[indices]
        
        # TODO: Calculate statistics
        bootstrap_means.append(np.mean(X_boot))
        bootstrap_stds.append(np.std(X_boot))
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{n_bootstrap} bootstrap samples")
    
    # TODO: Convert to arrays
    bootstrap_means = np.array(bootstrap_means)
    bootstrap_stds = np.array(bootstrap_stds)
    
    # TODO: Calculate confidence intervals (95%)
    # Hint: Use np.percentile
    mean_ci_lower = np.percentile(bootstrap_means,2.5)
    mean_ci_upper = np.percentile(bootstrap_means,97.5)
    
    bootstrap_stats = {
        'mean_estimate': bootstrap_means.mean(),
        'mean_ci_lower': mean_ci_lower,
        'mean_ci_upper': mean_ci_upper,
        'std_estimate': bootstrap_stds.mean()
    }
    
    print(f"\nBootstrap Results:")
    print(f"Mean estimate: {bootstrap_stats['mean_estimate']:.4f}")
    print(f"95% CI: [{mean_ci_lower:.4f}, {mean_ci_upper:.4f}]")
    
    return bootstrap_stats


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED PROBLEM 2: Statistical Analysis & Validation")
    print("="*60 + "\n")
    
    try:
        # Load and analyze
        df, stats = load_and_analyze('dataset.csv')
        
        # Extract X and y
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = df.columns[:-1].tolist()
        
        # Outlier detection
        outlier_report, outlier_mask = detect_outliers_multiple_methods(X)
        
        # Remove outliers
        X_clean = X[~outlier_mask]
        y_clean = y[~outlier_mask]
        
        # Correlation analysis
        corr_matrix, high_corr = correlation_analysis(X_clean, feature_names)
        
        # Normalize
        X_norm = (X_clean - X_clean.mean(axis=0)) / (X_clean.std(axis=0) + 1e-8)
        
        # K-fold cross-validation
        folds = create_k_folds(X_norm, y_clean, k=5)
        cv_results = cross_validate(X_norm, y_clean, folds)
        
        # Bootstrap sampling
        bootstrap_stats = bootstrap_sampling(X_norm, y_clean, n_bootstrap=100)
        
        print("\n" + "="*60)
        print("ADVANCED ANALYSIS COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Complete all TODO items!")
import pandas as pd
import numpy as np

"""
ADVANCED PROBLEM 3: Complex Batch Operations & Data Augmentation
Advanced batching techniques and data augmentation strategies
"""

def load_and_prepare(filename):
    """Load data and prepare for advanced batch processing."""
    
    print("="*60)
    print("STEP 1: Load and Prepare Data")
    print("="*60)
    
    # TODO: Load CSV
    df=pd.read_csv(filename)
    
    
    # TODO: Handle nulls
    df=df.fillna(0)
    
    
    # TODO: Extract X and y
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    
    # TODO: Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    print(f"Prepared data: X={X.shape}, y={y.shape}")
    
    return X, y


def create_stratified_batches(X, y, batch_size=32):
    """
    Create batches that maintain class distribution.
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    batch_size : int
    
    Returns:
    --------
    batches : list of tuples
        List of (X_batch, y_batch)
    """
    
    print("\n" + "="*60)
    print("STEP 2: Create Stratified Batches")
    print("="*60)
    
    # TODO: Get unique classes
    classes = np.unique(y)
    
    # TODO: Separate data by class
    class_data = {}
    for cls in classes:
        # TODO: Get indices where y equals cls
        mask = (y==cls)
        class_data[cls] = {
            'X': X[mask],
            'y': y[mask],
            'count': np.sum(mask)
        }
        print(f"Class {cls}: {class_data[cls]['count']} samples")
    
    # TODO: Calculate samples per class per batch
    samples_per_class = batch_size // len(classes)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Samples per class per batch: {samples_per_class}")
    
    # TODO: Create batches
    batches = []
    class_indices = {cls: 0 for cls in classes}  # Track position in each class
    
    # TODO: Calculate number of batches
    min_class_count = min([class_data[cls]['count'] for cls in classes])
    n_batches = min_class_count//samples_per_class
    
    for batch_idx in range(n_batches):
        X_batch_list = []
        y_batch_list = []
        
        # TODO: Sample from each class
        for cls in classes:
            # TODO: Get start and end indices
            start = class_indices[cls]
            end = start+samples_per_class
            
            # TODO: Get samples for this class
            X_cls = class_data[cls]['X'][start:end]
            y_cls = class_data[cls]['y'][start:end]
            
            X_batch_list.append(X_cls)
            y_batch_list.append(y_cls)
            
            # TODO: Update index
            class_indices[cls] = end
        
        # TODO: Concatenate all classes for this batch
        X_batch = np.vstack(X_batch_list)
        y_batch = np.vstack(y_batch_list)
        
        # TODO: Shuffle within batch
        shuffle_idx = np.random.permutation(len(X_batch))
        X_batch = X_batch[shuffle_idx]
        y_batch = y_batch[shuffle_idx]
        
        batches.append((X_batch, y_batch))
        
        print(f"Batch {batch_idx+1}: {X_batch.shape[0]} samples")
    
    return batches


def augment_data(X, y, augmentation_factor=2):
    """
    Augment data by adding noise and creating variations.
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    augmentation_factor : int
        How many times to augment (1 = double the data)
    
    Returns:
    --------
    X_augmented, y_augmented : numpy.ndarray
    """
    
    print("\n" + "="*60)
    print(f"STEP 3: Data Augmentation (factor={augmentation_factor})")
    print("="*60)
    
    X_augmented = [X]
    y_augmented = [y]
    
    # TODO: Create augmented versions
    for aug_idx in range(augmentation_factor):
        
        # Method 1: Add Gaussian noise
        # TODO: Generate noise with mean=0, std=0.1
        noise = np.random.normal(0,0.1,X.shape)
        X_noisy = X+noise
        
        X_augmented.append(X_noisy)
        y_augmented.append(y.copy())
        
        # Method 2: Scale features randomly (0.9 to 1.1)
        # TODO: Generate random scaling factors
        scale_factors = np.random.uniform(0.9,1.1,X.shape[1])
        X_scaled = X*scale_factors
        
        X_augmented.append(X_scaled)
        y_augmented.append(y.copy())
        
        print(f"Augmentation {aug_idx+1}: Created {len(X)*2} new samples")
    
    # TODO: Concatenate all versions
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.concatenate(y_augmented)
    
    print(f"\nOriginal data: {X.shape[0]} samples")
    print(f"Augmented data: {X_augmented.shape[0]} samples")
    
    return X_augmented, y_augmented


def batch_inference_with_multiple_models(X, y, n_models=3, batch_size=32):
    """
    Perform inference with multiple models and ensemble predictions.
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    n_models : int
        Number of models in ensemble
    batch_size : int
    
    Returns:
    --------
    ensemble_accuracy : float
    """
    
    print("\n" + "="*60)
    print(f"STEP 4: Ensemble Inference ({n_models} models)")
    print("="*60)
    
    n_samples, n_features = X.shape
    
    # TODO: Initialize multiple models with different random weights
    models = []
    for i in range(n_models):
        # TODO: Create random weight matrix
        W = np.random.randn(n_features, 1) * 0.01
        b = np.random.randn()
        models.append({'W': W, 'b': b})
        print(f"Model {i+1} initialized")
    
    # TODO: Store predictions from each model
    all_model_predictions = []
    
    # TODO: Inference for each model
    for model_idx, model in enumerate(models):
        W = model['W']
        b = model['b']
        
        model_predictions = []
        
        # TODO: Batch inference
        for i in range(0, n_samples, batch_size):
            # TODO: Get batch
            X_batch = 
            
            # TODO: Forward pass
            pred = 
            
            model_predictions.append(pred)
        
        # TODO: Combine predictions
        model_predictions = 
        all_model_predictions.append(model_predictions)
        
        print(f"Model {model_idx+1} inference complete")
    
    # TODO: Stack predictions from all models
    # Shape: (n_models, n_samples, 1)
    all_model_predictions = 
    
    # TODO: Ensemble by averaging (majority voting for classification)
    # Take mean across models (axis=0)
    ensemble_predictions = 
    
    # TODO: Convert to binary
    y_pred = 
    
    # TODO: Calculate accuracy
    accuracy = 
    
    print(f"\nEnsemble Accuracy: {accuracy:.4f}")
    
    return accuracy


def weighted_batch_sampling(X, y, batch_size=32, n_batches=10):
    """
    Sample batches with weighted probabilities (oversample minority class).
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    batch_size : int
    n_batches : int
    
    Returns:
    --------
    batches : list
    """
    
    print("\n" + "="*60)
    print("STEP 5: Weighted Batch Sampling")
    print("="*60)
    
    # TODO: Calculate class distribution
    unique_classes, class_counts = 
    
    print("Class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # TODO: Calculate sampling weights (inverse of class frequency)
    # Minority class gets higher weight
    class_weights = {}
    total_samples = len(y)
    
    for cls, count in zip(unique_classes, class_counts):
        # TODO: Weight = total_samples / (n_classes * class_count)
        class_weights[cls] = 
    
    # TODO: Assign weight to each sample based on its class
    sample_weights = np.zeros(len(y))
    for i, label in enumerate(y):
        # TODO: Get weight for this sample's class
        sample_weights[i] = 
    
    # TODO: Normalize weights to sum to 1
    sample_weights = 
    
    print("\nClass weights:", class_weights)
    
    # TODO: Create weighted batches
    batches = []
    
    for batch_idx in range(n_batches):
        # TODO: Sample indices with weights
        # Hint: np.random.choice(n_samples, size=batch_size, replace=True, p=weights)
        indices = 
        
        # TODO: Get batch
        X_batch = 
        y_batch = 
        
        # TODO: Count class distribution in this batch
        batch_dist = {}
        for cls in unique_classes:
            count = 
            batch_dist[cls] = count
        
        batches.append((X_batch, y_batch))
        
        print(f"Batch {batch_idx+1}: {batch_dist}")
    
    return batches


def online_learning_simulation(X, y, initial_batch_size=100, 
                               update_batch_size=10, n_updates=20):
    """
    Simulate online learning with incremental batch updates.
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    initial_batch_size : int
    update_batch_size : int
    n_updates : int
    
    Returns:
    --------
    accuracy_history : list
    """
    
    print("\n" + "="*60)
    print("STEP 6: Online Learning Simulation")
    print("="*60)
    
    n_samples, n_features = X.shape
    
    # TODO: Initialize with first batch
    X_train = 
    y_train = 
    
    # TODO: Keep rest for updates
    X_remaining = 
    y_remaining = 
    
    print(f"Initial training: {X_train.shape[0]} samples")
    print(f"Remaining for updates: {X_remaining.shape[0]} samples")
    
    # TODO: Initialize parameters
    W = 
    b = 0.0
    
    accuracy_history = []
    current_idx = 0
    
    # TODO: Simulate online updates
    for update_idx in range(n_updates):
        
        # TODO: Compute current statistics
        mean = 
        std = 
        
        # TODO: Normalize current training data
        X_train_norm = 
        
        # TODO: Simple inference on current training data
        predictions = 
        
        # TODO: Calculate accuracy
        y_pred = 
        accuracy = 
        
        accuracy_history.append(accuracy)
        
        print(f"Update {update_idx+1}: Training size={X_train.shape[0]}, Accuracy={accuracy:.4f}")
        
        # TODO: Add new batch to training data
        if current_idx < len(X_remaining):
            end_idx = min(current_idx + update_batch_size, len(X_remaining))
            
            # TODO: Get new batch
            X_new = 
            y_new = 
            
            # TODO: Concatenate with existing training data
            X_train = 
            y_train = 
            
            current_idx = end_idx
    
    print(f"\nFinal training size: {X_train.shape[0]} samples")
    print(f"Final accuracy: {accuracy_history[-1]:.4f}")
    
    return accuracy_history


def parallel_batch_processing(X, y, n_workers=4, batch_size=32):
    """
    Simulate parallel batch processing (conceptual, not truly parallel).
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    n_workers : int
    batch_size : int
    
    Returns:
    --------
    results : list
    """
    
    print("\n" + "="*60)
    print(f"STEP 7: Parallel Batch Processing ({n_workers} workers)")
    print("="*60)
    
    n_samples = X.shape[0]
    
    # TODO: Divide data into chunks for each worker
    chunk_size = 
    
    worker_results = []
    
    # TODO: Simulate each worker processing its chunk
    for worker_id in range(n_workers):
        # TODO: Get this worker's data chunk
        start = 
        end = 
        
        X_chunk = 
        y_chunk = 
        
        print(f"\nWorker {worker_id+1} processing {len(X_chunk)} samples...")
        
        # TODO: Process this chunk in batches
        chunk_predictions = []
        
        W = np.zeros((X.shape[1], 1))
        b = 0.0
        
        for i in range(0, len(X_chunk), batch_size):
            # TODO: Get batch
            X_batch = 
            
            # TODO: Forward pass
            pred = 
            
            chunk_predictions.append(pred)
        
        # TODO: Combine chunk predictions
        chunk_predictions = 
        
        # TODO: Calculate accuracy for this chunk
        y_pred = 
        accuracy = 
        
        worker_results.append({
            'worker_id': worker_id + 1,
            'samples_processed': len(X_chunk),
            'accuracy': accuracy
        })
        
        print(f"Worker {worker_id+1} accuracy: {accuracy:.4f}")
    
    # TODO: Calculate overall accuracy
    total_accuracy = 
    print(f"\nOverall parallel accuracy: {total_accuracy:.4f}")
    
    return worker_results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED PROBLEM 3: Complex Batch Operations")
    print("="*60 + "\n")
    
    try:
        # Load data
        X, y = load_and_prepare('data.csv')
        
        # Stratified batches
        stratified_batches = create_stratified_batches(X, y, batch_size=32)
        
        # Data augmentation
        X_aug, y_aug = augment_data(X, y, augmentation_factor=2)
        
        # Ensemble inference
        ensemble_acc = batch_inference_with_multiple_models(X, y, n_models=3)
        
        # Weighted sampling
        weighted_batches = weighted_batch_sampling(X, y, batch_size=32, n_batches=10)
        
        # Online learning
        accuracy_hist = online_learning_simulation(X, y)
        
        # Parallel processing
        parallel_results = parallel_batch_processing(X, y, n_workers=4)
        
        print("\n" + "="*60)
        print("ADVANCED BATCH OPERATIONS COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Complete all TODO items!")
import pandas as pd
import numpy as np

"""
ADVANCED PROBLEM 4: Time Series & Sequential Data Processing
Handle temporal data, create sequences, and perform rolling operations
"""

def load_time_series(filename):
    """
    Load time series data with datetime index.
    
    Parameters:
    -----------
    filename : str
    
    Returns:
    --------
    df : pandas.DataFrame with datetime index
    """
    
    print("="*60)
    print("STEP 1: Load Time Series Data")
    print("="*60)
    
    # TODO: Load CSV
    
    
    # TODO: Convert first column to datetime if it exists
    # Assume first column is 'date' or 'timestamp'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # TODO: Sort by index (datetime)
    
    
    print(f"Time series loaded: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


def create_rolling_features(df, windows=[3, 7, 14]):
    """
    Create rolling mean and std features for different windows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
    windows : list of int
        Window sizes for rolling statistics
    
    Returns:
    --------
    df_enhanced : pandas.DataFrame
    """
    
    print("\n" + "="*60)
    print("STEP 2: Create Rolling Features")
    print("="*60)
    
    df_enhanced = df.copy()
    
    # Assume we're working with a 'value' column
    if 'value' not in df_enhanced.columns:
        # Use first numeric column
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            value_col = numeric_cols[0]
        else:
            print("No numeric column found!")
            return df_enhanced
    else:
        value_col = 'value'
    
    # TODO: Create rolling features for each window
    for window in windows:
        # TODO: Calculate rolling mean
        # Hint: df[col].rolling(window=window).mean()
        df_enhanced[f'rolling_mean_{window}'] = 
        
        # TODO: Calculate rolling std
        df_enhanced[f'rolling_std_{window}'] = 
        
        # TODO: Calculate rolling min
        df_enhanced[f'rolling_min_{window}'] = 
        
        # TODO: Calculate rolling max
        df_enhanced[f'rolling_max_{window}'] = 
        
        print(f"Created rolling features for window={window}")
    
    # TODO: Drop rows with NaN (from rolling operations)
    df_enhanced = 
    
    print(f"\nEnhanced data shape: {df_enhanced.shape}")
    print(f"New feature count: {df_enhanced.shape[1]}")
    
    return df_enhanced


def create_lag_features(df, lags=[1, 2, 3, 7]):
    """
    Create lagged versions of features (t-1, t-2, etc.).
    
    Parameters:
    -----------
    df : pandas.DataFrame
    lags : list of int
        Lag periods to create
    
    Returns:
    --------
    df_lagged : pandas.DataFrame
    """
    
    print("\n" + "="*60)
    print("STEP 3: Create Lag Features")
    print("="*60)
    
    df_lagged = df.copy()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # TODO: Create lag features for each numeric column
    for col in numeric_cols:
        for lag in lags:
            # TODO: Create lagged column
            # Hint: df[col].shift(lag)
            df_lagged[f'{col}_lag_{lag}'] = 
            
        print(f"Created {len(lags)} lag features for {col}")
    
    # TODO: Drop rows with NaN
    df_lagged = 
    
    print(f"\nData with lags shape: {df_lagged.shape}")
    
    return df_lagged


def create_sequences(X, y, sequence_length=10):
    """
    Create sequences for sequential models (e.g., LSTM input).
    
    Parameters:
    -----------
    X : numpy.ndarray (n_samples, n_features)
    y : numpy.ndarray (n_samples,)
    sequence_length : int
        Length of each sequence
    
    Returns:
    --------
    X_seq : numpy.ndarray (n_sequences, sequence_length, n_features)
    y_seq : numpy.ndarray (n_sequences,)
    """
    
    print("\n" + "="*60)
    print(f"STEP 4: Create Sequences (length={sequence_length})")
    print("="*60)
    
    n_samples, n_features = X.shape
    
    # TODO: Calculate number of sequences
    n_sequences = 
    
    print(f"Total samples: {n_samples}")
    print(f"Number of sequences: {n_sequences}")
    
    # TODO: Initialize sequence arrays
    X_seq = np.zeros((n_sequences, sequence_length, n_features))
    y_seq = np.zeros(n_sequences)
    
    # TODO: Create sequences
    for i in range(n_sequences):
        # TODO: Get sequence starting at index i
        # Sequence should be from i to i+sequence_length
        X_seq[i] = 
        
        # TODO: Target is the value at the end of the sequence
        y_seq[i] = 
    
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")
    
    return X_seq, y_seq


def sliding_window_statistics(X, window_size=5, stride=1):
    """
    Calculate statistics over sliding windows.
    
    Parameters:
    -----------
    X : numpy.ndarray (n_samples, n_features)
    window_size : int
    stride : int
        Step size for sliding
    
    Returns:
    --------
    window_stats : list of dict
    """
    
    print("\n" + "="*60)
    print(f"STEP 5: Sliding Window Statistics")
    print(f"Window size: {window_size}, Stride: {stride}")
    print("="*60)
    
    n_samples, n_features = X.shape
    window_stats = []
    
    # TODO: Slide window across data
    for i in range(0, n_samples - window_size + 1, stride):
        # TODO: Get window
        window = 
        
        # TODO: Calculate statistics for this window
        stats = {
            'start_idx': i,
            'end_idx': i + window_size,
            'mean': ,
            'std': ,
            'min': ,
            'max': ,
            'median': 
        }
        
        window_stats.append(stats)
        
        if len(window_stats) % 20 == 0:
            print(f"Processed {len(window_stats)} windows...")
    
    print(f"\nTotal windows: {len(window_stats)}")
    
    # TODO: Convert statistics to arrays for analysis
    means = np.array([s['mean'] for s in window_stats])
    stds = np.array([s['std'] for s in window_stats])
    
    print(f"Mean of window means: {means.mean():.4f}")
    print(f"Mean of window stds: {stds.mean():.4f}")
    
    return window_stats


def time_based_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split time series data temporally (no shuffling!).
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    train_ratio : float
    val_ratio : float
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    
    print("\n" + "="*60)
    print("STEP 6: Time-Based Split (No Shuffling)")
    print("="*60)
    
    n_samples = len(X)
    
    # TODO: Calculate split points
    train_end = 
    val_end = 
    
    # TODO: Split data sequentially
    X_train = 
    y_train = 
    
    X_val = 
    y_val = 
    
    X_test = 
    y_test = 
    
    print(f"Train: {X_train.shape[0]} samples (0 to {train_end})")
    print(f"Val: {X_val.shape[0]} samples ({train_end} to {val_end})")
    print(f"Test: {X_test.shape[0]} samples ({val_end} to {n_samples})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def expanding_window_validation(X, y, initial_train_size=100, 
                                step_size=20, n_steps=10):
    """
    Perform expanding window cross-validation for time series.
    
    Parameters:
    -----------
    X : numpy.ndarray
    y : numpy.ndarray
    initial_train_size : int
    step_size : int
    n_steps : int
    
    Returns:
    --------
    val_scores : list
    """
    
    print("\n" + "="*60)
    print("STEP 7: Expanding Window Validation")
    print("="*60)
    
    n_features = X.shape[1]
    val_scores = []
    
    # TODO: Perform expanding window validation
    for step in range(n_steps):
        
        # TODO: Calculate current train size
        current_train_size = 
        
        # TODO: Split data
        X_train = 
        y_train = 
        
        # TODO: Get next step_size samples for validation
        val_start = current_train_size
        val_end = 
        
        if val_end > len(X):
            break
        
        X_val = 
        y_val = 
        
        # TODO: Normalize based on training data
        train_mean = 
        train_std = 
        
        X_train_norm = 
        X_val_norm = 
        
        # TODO: Simple model inference
        W = 
        b = 0.0
        
        # TODO: Predictions on validation set
        predictions = 
        
        # TODO: Calculate accuracy
        y_pred = 
        accuracy = 
        
        val_scores.append(accuracy)
        
        print(f"Step {step+1}: Train size={current_train_size}, Val size={X_val.shape[0]}, Accuracy={accuracy:.4f}")
    
    # TODO: Calculate statistics
    mean_score = 
    std_score = 
    
    print(f"\nMean validation score: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return val_scores


def batch_sequence_processing(X_seq, y_seq, batch_size=16):
    """
    Process sequential data in batches.
    
    Parameters:
    -----------
    X_seq : numpy.ndarray (n_sequences, seq_length, n_features)
    y_seq : numpy.ndarray (n_sequences,)
    batch_size : int
    
    Returns:
    --------
    predictions : numpy.ndarray
    """
    
    print("\n" + "="*60)
    print("STEP 8: Batch Sequence Processing")
    print("="*60)
    
    n_sequences, seq_length, n_features = X_seq.shape
    
    # TODO: Flatten sequences for simple model
    # Shape: (n_sequences, seq_length * n_features)
    X_flat = 
    
    print(f"Flattened shape: {X_flat.shape}")
    
    # TODO: Initialize parameters
    W = 
    b = 0.0
    
    predictions = []
    
    # TODO: Batch processing
    for i in range(0, n_sequences, batch_size):
        # TODO: Get batch
        X_batch = 
        
        # TODO: Forward pass
        pred = 
        
        predictions.append(pred)
        
        print(f"Processed batch {i//batch_size + 1}")
    
    # TODO: Combine predictions
    predictions = 
    
    # TODO: Calculate accuracy
    y_pred = 
    accuracy = 
    
    print(f"\nSequence processing accuracy: {accuracy:.4f}")
    
    return predictions


def detect_anomalies_time_series(X, window_size=50, threshold=3.0):
    """
    Detect anomalies in time series using sliding window and z-score.
    
    Parameters:
    -----------
    X : numpy.ndarray (n_samples, n_features)
    window_size : int
    threshold : float
        Z-score threshold for anomaly
    
    Returns:
    --------
    anomalies : numpy.ndarray (boolean mask)
    """
    
    print("\n" + "="*60)
    print(f"STEP 9: Anomaly Detection")
    print(f"Window size: {window_size}, Threshold: {threshold}")
    print("="*60)
    
    n_samples = len(X)
    anomalies = np.zeros(n_samples, dtype=bool)
    
    # TODO: For each point, check if it's anomalous based on previous window
    for i in range(window_size, n_samples):
        # TODO: Get previous window
        window = 
        
        # TODO: Calculate window statistics
        window_mean = 
        window_std = 
        
        # TODO: Calculate z-score for current point
        current_point = X[i]
        z_score = 
        
        # TODO: Check if any feature exceeds threshold
        if :
            anomalies[i] = True
    
    n_anomalies = np.sum(anomalies)
    print(f"\nAnomalies detected: {n_anomalies} ({n_anomalies/n_samples*100:.2f}%)")
    print(f"Anomaly indices: {np.where(anomalies)[0][:10]}...")  # Show first 10
    
    return anomalies


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED PROBLEM 4: Time Series Processing")
    print("="*60 + "\n")
    
    try:
        # Load time series
        df = load_time_series('timeseries_data.csv')
        
        # Create rolling features
        df_rolling = create_rolling_features(df, windows=[3, 7, 14])
        
        # Create lag features
        df_lagged = create_lag_features(df_rolling, lags=[1, 2, 3, 7])
        
        # Convert to numpy
        X = df_lagged.iloc[:, :-1].values
        y = df_lagged.iloc[:, -1].values
        
        # Create sequences
        X_seq, y_seq = create_sequences(X, y, sequence_length=10)
        
        # Sliding window statistics
        window_stats = sliding_window_statistics(X, window_size=5)
        
        # Time-based split
        X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y)
        
        # Expanding window validation
        val_scores = expanding_window_validation(X, y)
        
        # Batch sequence processing
        predictions = batch_sequence_processing(X_seq, y_seq)
        
        # Anomaly detection
        anomalies = detect_anomalies_time_series(X)
        
        print("\n" + "="*60)
        print("TIME SERIES PROCESSING COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Complete all TODO items!")   
        
        
# Load and prepare
df = pd.read_csv(filename)
df = df.fillna(0)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# Stratified batches
classes = np.unique(y)
mask = (y == cls)
class_data[cls]['X'] = X[mask]
class_data[cls]['y'] = y[mask]
class_data[cls]['count'] = np.sum(mask)

samples_per_class = batch_size // len(classes)
n_batches = min_class_count // samples_per_class

start = class_indices[cls]
end = start + samples_per_class
X_cls = class_data[cls]['X'][start:end]
y_cls = class_data[cls]['y'][start:end]

X_batch = np.vstack(X_batch_list)
y_batch = np.concatenate(y_batch_list)
shuffle_idx = np.random.permutation(len(X_batch))
X_batch = X_batch[shuffle_idx]
y_batch = y_batch[shuffle_idx]

# Data augmentation
noise = np.random.normal(0, 0.1, X.shape)
X_noisy = X + noise

scale_factors = np.random.uniform(0.9, 1.1, X.shape[1])
X_scaled = X * scale_factors

X_augmented = np.vstack(X_augmented)
y_augmented = np.concatenate(y_augmented)

# Ensemble inference
W = np.random.randn(n_features, 1) * 0.01

X_batch = X[i:i+batch_size]
pred = X_batch @ W + b

model_predictions = np.vstack(model_predictions)
all_model_predictions = np.array(all_model_predictions)
ensemble_predictions = all_model_predictions.mean(axis=0)

y_pred = (ensemble_predictions >= 0.5).astype(int)
accuracy = np.mean(y_pred == y.reshape(-1, 1))

# Weighted sampling
unique_classes, class_counts = np.unique(y, return_counts=True)
class_weights[cls] = total_samples / (len(unique_classes) * count)
sample_weights[i] = class_weights[label]
sample_weights = sample_weights / sample_weights.sum()

indices = np.random.choice(n_samples, size=batch_size, replace=True, p=sample_weights)
X_batch = X[indices]
y_batch = y[indices]
count = np.sum(y_batch == cls)

# Online learning
X_train = X[:initial_batch_size]
y_train = y[:initial_batch_size]
X_remaining = X[initial_batch_size:]
y_remaining = y[initial_batch_size:]

W = np.zeros((n_features, 1))

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_norm = (X_train - mean) / (std + 1e-8)

predictions = X_train_norm @ W + b
y_pred = (predictions >= 0.5).astype(int)
accuracy = np.mean(y_pred == y_train.reshape(-1, 1))

X_new = X_remaining[current_idx:end_idx]
y_new = y_remaining[current_idx:end_idx]
X_train = np.vstack([X_train, X_new])
y_train = np.concatenate([y_train, y_new])

# Parallel processing
chunk_size = n_samples // n_workers
start = worker_id * chunk_size
end = (worker_id + 1) * chunk_size if worker_id < n_workers - 1 else n_samples
X_chunk = X[start:end]
y_chunk = y[start:end]

X_batch = X_chunk[i:i+batch_size]
pred = X_batch @ W + b
chunk_predictions = np.vstack(chunk_predictions)

y_pred = (chunk_predictions >= 0.5).astype(int)
accuracy = np.mean(y_pred == y_chunk.reshape(-1, 1))

total_accuracy = np.mean([r['accuracy'] for r in worker_results])

# Load time series
df = pd.read_csv(filename)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.sort_index()

# Rolling features
df_enhanced[f'rolling_mean_{window}'] = df_enhanced[value_col].rolling(window=window).mean()
df_enhanced[f'rolling_std_{window}'] = df_enhanced[value_col].rolling(window=window).std()
df_enhanced[f'rolling_min_{window}'] = df_enhanced[value_col].rolling(window=window).min()
df_enhanced[f'rolling_max_{window}'] = df_enhanced[value_col].rolling(window=window).max()
df_enhanced = df_enhanced.dropna()

# Lag features
df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
df_lagged = df_lagged.dropna()

# Create sequences
n_sequences = n_samples - sequence_length + 1
X_seq = np.zeros((n_sequences, sequence_length, n_features))
y_seq = np.zeros(n_sequences)

X_seq[i] = X[i:i+sequence_length]
y_seq[i] = y[i+sequence_length-1]

# Sliding window
window = X[i:i+window_size]
stats['mean'] = window.mean()
stats['std'] = window.std()
stats['min'] = window.min()
stats['max'] = window.max()
stats['median'] = np.median(window)

# Time-based split
train_end = int(n_samples * train_ratio)
val_end = int(n_samples * (train_ratio + val_ratio))

X_train = X[:train_end]
y_train = y[:train_end]
X_val = X[train_end:val_end]
y_val = y[train_end:val_end]
X_test = X[val_end:]
y_test = y[val_end:]

# Expanding window
current_train_size = initial_train_size + step * step_size
X_train = X[:current_train_size]
y_train = y[:current_train_size]
val_end = min(current_train_size + step_size, len(X))
X_val = X[val_start:val_end]
y_val = y[val_start:val_end]

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train_norm = (X_train - train_mean) / (train_std + 1e-8)
X_val_norm = (X_val - train_mean) / (train_std + 1e-8)

W = np.zeros((n_features, 1))
predictions = X_val_norm @ W + b
y_pred = (predictions >= 0.5).astype(int)
accuracy = np.mean(y_pred == y_val.reshape(-1, 1))

mean_score = np.mean(val_scores)
std_score = np.std(val_scores)

# Batch sequences
X_flat = X_seq.reshape(n_sequences, -1)
W = np.zeros((seq_length * n_features, 1))
X_batch = X_flat[i:i+batch_size]
pred = X_batch @ W + b
predictions = np.vstack(predictions)

y_pred = (predictions >= 0.5).astype(int)
accuracy = np.mean(y_pred == y_seq.reshape(-1, 1))

# Anomaly detection
window = X[i-window_size:i]
window_mean = window.mean(axis=0)
window_std = window.std(axis=0)
current_point = X[i]
z_score = np.abs((current_point - window_mean) / (window_std + 1e-8))

if np.any(z_score > threshold):
    anomalies[i] = True                                                           
            