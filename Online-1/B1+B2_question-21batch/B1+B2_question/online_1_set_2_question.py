import pandas as pd
import numpy as np

np.random.seed(42)

# 1. Windowing Average Imputation
def windowing_average_imputation(x, k=3):
    # TODO: Windowing average imputation
    n_samples,n_features=x.shape
    x_imputed=x.copy()
    b=k//2
    col_mean=np.nanmean(x,axis=0)
    for col in range(n_features):
        column=x_imputed[:,col]
        nan_features_in_col=np.where(np.isnan(column))[0]
        for i in nan_features_in_col:
            start_index=max(0,i-b)
            end_index=min(n_samples,i+b+1)
            window=column[start_index:end_index]
            full_window_available=((end_index-start_index)==k)
            if(full_window_available):
                valid_values=window[~np.isnan(window)]
                if(len(valid_values)>0):
                    x_imputed[i,col]=np.mean(valid_values)
            else:
                x_imputed[i,col]=col_mean[col]    
    
    # for col in range(n_features):
    #     column=x_imputed[:,col]
    #     nan_features_in_col=np.where(np.isnan(column))[0]
    #     col_mean=np.nanmean(column)
    #     for i in nan_features_in_col:
    #         x_imputed[i,col]=col_mean
            
    return x_imputed #of shape as x

# 2. Tanh
def tanh(x):
    # TODO: Tanh function
    a=np.exp(x)
    b=np.exp(-x)
    # value=(a-b)/(a+b)
    value=np.tanh(x)
    
    return value # of shape as x

# 3. Tanh gradient
def tanh_gradient(x, dout=1):
    # TODO: Tanh gradient
    a=tanh(x)
    grad=(1-a**2)*dout
    
    return grad # of shape as x

# 4. MAE
def mae(y_pred, y_true):
    # TODO: MAE loss
    
    loss=np.mean(np.abs(y_pred-y_true)) 
    
    return loss # only a scalar value

# def sign(z):
#     if z>0:
#         return 1
#     elif z<0:
#         return -1
#     else:
#         return 0
# 5. MAE gradient
def mae_gradient(y_pred, y_true):
    # TODO: MAE gradient
    n=y_pred.shape[0]
    z=y_pred-y_true
    
    grad=(1/n)*np.sign(z)
    
    
    return grad # of shape as y_pred

# 6. Inference on test data and evaluate MAE
def inference(df_test, W, b, X):
    # TODO: Load test data and evaluate MAE.
    
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values.reshape(-1, 1)
    print("NaN values in X_test before imputation:", np.isnan(X_test).sum())
    X_test = windowing_average_imputation(X_test, k=10)
    print("NaN values in X_test after imputation:", np.isnan(X_test).sum())
    outputs = X_test @ W + b
    y_pred=tanh(outputs)

    
    test_mae=mae(y_pred, y_test)
    return test_mae

def mse_loss(y_pred, y_true):
    n = len(y_pred)
    loss = (1/n) * np.sum((y_pred - y_true) ** 2)
    return loss

# ============================
# MAIN: MINIBATCH TRAINING + ACCURACY
# ============================
if __name__ == "__main__":
    df = pd.read_csv("B1+B2_question-21batch/B1+B2_question/train_data.csv", header=None)
    print("Data size:", df.shape)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    print("NaN values in X before imputation:", np.isnan(X).sum())
    print("Performing windowing average imputation for missing values...")
    X = windowing_average_imputation(X, k=10)
    print("NaN values in X after imputation:", np.isnan(X).sum())

    n_samples, n_features = X.shape

    # Initialize parameters
    W = np.zeros((n_features, 1))
    b = 0.0

    batch_size = 50
    learning_rate = 0.01
    num_epochs = 20

    print(f"\nTraining for {num_epochs} epochs with learning rate {learning_rate}...\n")

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data at the beginning of each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, n_samples, batch_size):
            Xb = X_shuffled[i:i+batch_size]
            yb = y_shuffled[i:i+batch_size]
            
            outputs = Xb @ W + b
            preds = tanh(outputs)
            
            # Compute loss
            batch_loss = mae(preds, yb)
            epoch_loss += batch_loss
            num_batches += 1
            
            # Gradient of loss w.r.t predictions
            dloss_dpreds = mae_gradient(preds, yb)
            
            # Gradient of tanh
            dpreds_doutputs = tanh_gradient(outputs, dout=dloss_dpreds)
            
            # Gradients w.r.t W and b
            dW = Xb.T @ dpreds_doutputs
            db = np.sum(dpreds_doutputs)
            
            # Update weights
            W -= learning_rate * dW
            b -= learning_rate * db
        
        avg_loss = epoch_loss / num_batches
        
        # Calculate accuracy on full dataset
        outputs_all = X @ W + b
        y_pred = tanh(outputs_all)

        mse = mse_loss(y_pred, y)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}, MSE: {mse:.4f}")

    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)

    # Final evaluation
    outputs_final = X @ W + b
    preds_final = tanh(outputs_final)
    final_loss = mae(preds_final, y)
    mse_final = mse_loss(preds_final, y)

    print(f"\nFinal MAE Loss: {final_loss:.6f}")
    print(f"Final MSE: {mse_final:.4f}")
    print("\nTrained weights (W):", W.ravel())
    print(f"Trained bias (b): {b:.6f}")

    # ============================
    # INFERENCE ON TEST DATA
    # ============================

    df_test = pd.read_csv("B1+B2_question-21batch/B1+B2_question/test_data.csv", header=None)

    test_mae = inference(df_test, W, b, X)
    
    print(f"Test MAE: {test_mae:.6f}")
