import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# any scikit-learn estimator will be used as the base learner (to be provided)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
import xgboost as xgb
import random
from scipy import stats

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# TODO: Load the dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()




# TODO: Separate features (X) and target (y)
print(data.items)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print('Dataset Shape : ',df.shape)
print('Target Distribution : ',df)
X=df.drop('target',axis=1)
y=df['target']



# TODO: Train-test split (80%-20%)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(f"\n Train size : {X_train.shape} , Test Size : {X_test.shape}")
# TODO: Scale the features if needed
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# -------------------- Base Learner --------------------

# TODO: Train the base learner and compute the accuracy

base_learner=DecisionTreeClassifier(max_depth=1,random_state=42)
base_learner.fit(X_train_scaled,y_train)
y_pred_base=base_learner.predict(X_test_scaled)
lr_acc=accuracy_score(y_test,y_pred_base)
print(f"\nBase Learner (Decision Stump) Accuracy: {lr_acc:.4f}")

# -------------------- bagging/stacking/adaboost etc. with the base learner --------------------

# -------------------- Bagging --------------------
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    bootstrap=True,        # with replacement = Bagging
    random_state=42
)
bagging_model.fit(X_train_scaled, y_train)
bagging_acc = accuracy_score(y_test, bagging_model.predict(X_test_scaled))
print(f"Bagging Accuracy: {bagging_acc:.4f}")

# -------------------- AdaBoost (Boosting) --------------------
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
adaboost_model.fit(X_train_scaled, y_train)
adaboost_acc = accuracy_score(y_test, adaboost_model.predict(X_test_scaled))
print(f"AdaBoost Accuracy: {adaboost_acc:.4f}")

# -------------------- Gradient Boosting --------------------
gbm_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbm_model.fit(X_train_scaled, y_train)
gbm_acc = accuracy_score(y_test, gbm_model.predict(X_test_scaled))
print(f"Gradient Boosting Accuracy: {gbm_acc:.4f}")

# ---- How AdaBoost works (step-by-step logic) ----
# Step 1: Initialize equal weights w_i = 1/n for all samples
# Step 2: Train weak learner h_t on weighted data
# Step 3: Compute weighted error: eps = sum(w_i * I(h_t(x_i) != y_i)) / sum(w_i)
# Step 4: Compute learner weight: alpha = 0.5 * ln((1-eps)/eps)
# Step 5: Update weights: w_i = w_i * exp(-alpha * y_i * h_t(x_i))
# Step 6: Normalize weights
# Step 7: Final prediction: H(x) = sign(sum(alpha_t * h_t(x)))





# -------------------- XGBoost --------------------

# XGBoost key parameters:
# n_estimators  → number of trees (more = better but slower, risk overfit)
# max_depth     → tree depth (higher = more complex = overfit risk)
# learning_rate → step size (lower = slower but better generalization)
# subsample     → fraction of samples per tree (< 1 adds randomness)
# colsample_bytree → fraction of features per tree (like Random Forest trick)
# reg_lambda    → L2 regularization (higher = more regularized)
# reg_alpha     → L1 regularization


# TODO: Experiment XGBoost with different combinations of hyperparameters (n_estimators, max_depth, learning_rate etc).
param_grid=[
    #poor configurations likely to underfit or overfit
    {
        "n_estimator" : 1,
        "max_depth" : 1,
        "learning_rate" : 0.001,
        "sub_sample" : 0.3,
        "colsample_bytree" : 0.3,
        "reg_lambda" : 100
    },
    {
        "n_estimator" : 2,
        "max_depth" : 1,
        "learning_rate" : 0.001,
        "sub_sample" : 0.2,
        "colsample_bytree" : 0.2,
        "reg_lambda" : 50
    },    
    # Medium configurations
    {
        "n_estimators": 50,  
        "max_depth": 3,  
        "learning_rate": 0.1,
        "subsample": 0.8,    
        "colsample_bytree": 0.8, 
        "reg_lambda": 1
    },

    {
        "n_estimators": 100, 
        "max_depth": 4,  
        "learning_rate": 0.05,
        "subsample": 0.9,    
        "colsample_bytree": 0.9, 
        "reg_lambda": 1
    },

    # Good configurations (balanced bias-variance)
    {
        "n_estimators": 200, 
        "max_depth": 4,  
        "learning_rate": 0.05,
        "subsample": 0.8,    
        "colsample_bytree": 0.8, 
        "reg_lambda": 1
    },

    {
        "n_estimators": 300, 
        "max_depth": 5,  
        "learning_rate": 0.01,
        "subsample": 0.9,    
        "colsample_bytree": 0.85,
        "reg_lambda": 0.5
    },


]
# TODO: Train and predict with each combination
# Find the worst(Lowest possible XGBoost accuracy) and best(Highest possible XGBoost accuracy) XGBoost configurations

results=[]
for i, params in enumerate(param_grid):
    model = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append((acc, params))
    print(f"Config {i+1}: Acc={acc:.4f} | {params}")



# -------------------- Final Output --------------------

ensemble_results = {
    "Bagging": bagging_acc,
    "AdaBoost": adaboost_acc,
    "Gradient Boosting": gbm_acc
}
best_ensemble_name = max(ensemble_results, key=ensemble_results.get)
print("Best ensemble model : ",best_ensemble_name)
bagging_acc = ensemble_results[best_ensemble_name] 

results_sorted = sorted(results, key=lambda x: x[0])
xgb_poor_acc  = results_sorted[0][0]
xgb_best_acc  = results_sorted[-1][0]

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Base Learner Accuracy: {round(lr_acc, 4)}")
print(f"Best Ensemble ({best_ensemble_name}): {round(bagging_acc, 4)}")
print("-" * 60)
print(f"XGBoost Worst Accuracy: {round(xgb_poor_acc, 4)}")
print(f"XGBoost Best Accuracy: {round(xgb_best_acc, 4)}")
print("=" * 60)