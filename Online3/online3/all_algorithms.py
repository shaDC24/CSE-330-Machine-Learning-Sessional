"""
============================================================
  ENSEMBLE LEARNING — All Algorithms Complete Code
  Dataset: Breast Cancer (Binary Classification)
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ============================================================
# SETUP — Data Loading & Preprocessing
# ============================================================
data = load_iris()#load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

import pandas as pd

# df = pd.read_csv("breast_cancer.csv")
# X = df.drop("target", axis=1)   # সব feature
# y = df["target"]                # শুধু target column
# #diagnosis is the target column
# X = df.drop("diagnosis", axis=1)
# y = df["diagnosis"]

# #diagnosis malignenet , benihgn chilo... so eta ke number e convert korlam

# le = LabelEncoder() #2 tar beshi hole OneHotEncoder use kora hoy 
# y = le.fit_transform(df["diagnosis"])   # M/B → 1/0 (বা উল্টা) 
# X = df.drop("diagnosis", axis=1)
# print(y[:5])



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train) # fit ----> mean and std calculate kore and transforms sei value diye scale kore
X_test_s  = scaler.transform(X_test)


results = {}

# ============================================================
# 1. BASE LEARNER — Single Decision Tree
# ============================================================
# Weak learner: max_depth=1 (decision stump)
# Strong single learner: max_depth=None (full tree)

# Weak learner (stump)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train_s, y_train)
stump_acc = accuracy_score(y_test, stump.predict(X_test_s))
results['Decision Stump (weak)'] = stump_acc


# Full Decision Tree
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train_s, y_train)
full_tree_acc = accuracy_score(y_test, full_tree.predict(X_test_s))
results['Full Decision Tree'] = full_tree_acc

print(f"🔸Decision Stump Accuracy : {stump_acc:.4f}")
print(f"🔹Full Decision Tree Acc  : {full_tree_acc:.4f}")


# ============================================================
# 2. BAGGING
# ============================================================
# Bootstrap Aggregating:
# - Draw T bootstrap samples (with replacement)
# - Train one model on each sample
# - Final prediction = majority vote (classification) / average (regression)

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,       # T = number of bootstrap samples
    max_samples=0.1,       # size of each bootstrap sample (1.0 = same as dataset)
    max_features=1.0,      # fraction of features to use per tree
    bootstrap=True,        # with replacement (True = Bagging, False = Pasting)
    random_state=42,
    n_jobs=-1              # use all CPU cores
)
bagging.fit(X_train_s, y_train)
bagging_acc = accuracy_score(y_test, bagging.predict(X_test_s))
results['Bagging'] = bagging_acc
print(f"\nBagging Accuracy        : {bagging_acc:.4f}")

#Pasting (No Replacement)

pasting = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,       # T = number of bootstrap samples
    max_samples=1.0,       # size of each bootstrap sample (1.0 = same as dataset)
    max_features=1.0,      # fraction of features to use per tree
    bootstrap=False,        # with replacement (True = Bagging, False = Pasting)
    random_state=42,
    n_jobs=-1              # use all CPU cores
)
pasting.fit(X_train_s, y_train)
pasting_acc = accuracy_score(y_test, pasting.predict(X_test_s))
results['Pasting'] = pasting_acc
print(f"Pasting Accuracy        : {pasting_acc:.4f}")

#Random Subspace

random_subspace = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,       # T = number of bootstrap samples
    max_samples=1.0,       # size of each bootstrap sample (1.0 = same as dataset)
    max_features=0.1,      # fraction of features to use per tree
    bootstrap=True,        # with replacement (True = Bagging, False = Pasting)
    random_state=42,
    n_jobs=-1              # use all CPU cores
)
random_subspace .fit(X_train_s, y_train)
random_subspace_acc = accuracy_score(y_test, random_subspace.predict(X_test_s))
results['random_subspace'] = random_subspace_acc 
print(f"Random Subspace  Accuracy        : {random_subspace_acc:.4f}")

#Random patches

random_patches = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,       # T = number of bootstrap samples
    max_samples=0.1,       # size of each bootstrap sample (1.0 = same as dataset)
    max_features=0.1,      # fraction of features to use per tree
    bootstrap=True,        # with replacement (True = Bagging, False = Pasting)
    random_state=42,
    n_jobs=-1              # use all CPU cores
)
random_patches.fit(X_train_s, y_train)
random_patches_acc = accuracy_score(y_test, random_patches.predict(X_test_s))
results['random_patches'] = random_patches_acc 
print(f"Random Patches Accuracy        : {random_patches_acc:.4f}")


# ============================================================
# 3. RANDOM FOREST
# ============================================================
# Bagging + Random Feature Subset per split
# At each node: randomly select k = sqrt(p) features, find best split among them
# This reduces correlation between trees → lower variance

rf = RandomForestClassifier(
    n_estimators=100,          # number of trees T
    max_features='sqrt',       # k = sqrt(p) features per split (default for classification)
    max_depth=None,            # grow full trees
    min_samples_split=2,       # min samples to split a node
    min_samples_leaf=1,        # min samples in a leaf
    bootstrap=True,            # use bootstrap sampling
    oob_score=True,            # use Out-Of-Bag samples for validation
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_s, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_s))
results['Random Forest'] = rf_acc
print(f"\nRandom Forest Accuracy  : {rf_acc:.4f}")
print(f"OOB Score (free val)    : {rf.oob_score_:.4f}")  # validation without test set!

# Feature Importance from Random Forest
feat_imp = pd.Series(rf.feature_importances_, index=data.feature_names)
top3 = feat_imp.nlargest(3)
print("Top 3 Important Features:")
for fname, imp in top3.items():
    print(f"  {fname:40s}: {imp:.4f}")


#=============================================================================================================================================================================================================================
"""
Implement bootstrap sampling without using sklearn.
Given a dataset of n samples, create m bootstrap samples.
Each bootstrap sample should be same size as original (with replacement).
"""

def create_bootstrap_samples(X, y, n_samples=5):
    """
    Create bootstrap samples
    
    Args:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        n_samples: number of bootstrap samples to create
    
    Returns:
        List of (X_boot, y_boot) tuples
    """
    # Your code here
    n=5
    bootstrap_samples=[]
    for _ in range(n_samples):
        indices=np.random.choice(n,size=n,replace=True)
        X_boot=X[indices]
        y_boot=y[indices]
        bootstrap_samples.append((X_boot,y_boot))
    return bootstrap_samples    



# Test
import numpy as np
X = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
y = np.array([0, 0, 1, 1, 0])

samples = create_bootstrap_samples(X, y, n_samples=3)
for i, (X_boot, y_boot) in enumerate(samples):
    print(f"Bootstrap {i+1}:")
    print(X_boot)
    print(y_boot)


# **Expected Output:**

# Bootstrap 1:
# [[1 2]
#  [1 2]  ← repeated!
#  [5 6]
#  [7 8]
#  [3 4]]
# [0 0 1 1 0]

# Bootstrap 2:
# [[3 4]
#  [9 10]
#  [5 6]
#  [5 6]  ← repeated!
#  [1 2]]
# [0 0 1 1 0]





# ### **❓ Question 2: Calculate OOB Score Manually**
# python

# Given bootstrap indices for each tree, calculate OOB score manually.
# For each sample, find trees where it was NOT used in training (OOB),
# use those trees to predict, and calculate accuracy.



def calculate_oob_score(X, y, trees, bootstrap_indices):
    """
    Calculate Out-of-Bag score
    
    Args:
        X: features (n_samples, n_features)
        y: true labels (n_samples,)
        trees: list of trained decision trees
        bootstrap_indices: list of arrays, bootstrap_indices[i] = indices used for tree i
    
    Returns:
        OOB accuracy score
    """
    # Your code here
    n_samples=(len(X))
    oob_predictions=np.zeros(n_samples)
    oob_counts=np.zeros(n_samples)
    for i in range(n_samples):
        predictions=[]
        for tree_index,index_value in enumerate(bootstrap_indices):
            if i not in index_value :
                pred=trees[tree_index].predict(X[i:i+1])[0]
                predictions.append(pred)
        if(len(predictions)>0):
            oob_predictions[i]=np.bincount(predictions).argmax()
            oob_counts[i]=len(predictions)  
    mask=(oob_counts>0)
    oob_accuracy=np.mean(oob_predictions[mask]==y[mask])   
    return oob_accuracy           
# Test setup
from sklearn.tree import DecisionTreeClassifier
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Train 10 trees on bootstrap samples
trees = []
bootstrap_indices = []

for _ in range(10):
    indices = np.random.choice(100, 100, replace=True)
    bootstrap_indices.append(indices)
    
    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(X[indices], y[indices])
    trees.append(tree)

# Calculate OOB score
oob = calculate_oob_score(X, y, trees, bootstrap_indices)
print(f"OOB Score: {oob:.4f}")



# ### **❓ Question 3: Feature Importance from Scratch**

# """
# Calculate feature importance for Random Forest manually.
# For each feature, sum the total decrease in impurity (Gini)
# across all trees and all nodes where that feature was used.
# """

def calculate_feature_importance(trees, n_features):
    """
    Calculate feature importance
    
    Args:
        trees: list of fitted sklearn DecisionTree objects
        n_features: number of features
    
    Returns:
        Array of feature importances (sums to 1)
    """
    # Your code here
    # Hint: Use tree.tree_.feature, tree.tree_.impurity, tree.tree_.n_node_samples
    total_importance=np.zeros(n_features)
    for tree in trees :
        tree_obj=tree.tree_
        for node in range(tree_obj.node_count):
            if tree_obj.feature[node]!=-2:
                feature=tree_obj.feature[node]
                n_samples=tree_obj.n_node_samples[node]
                impurity=tree_obj.impurity[node]
                left_child = tree_obj.children_left[node]
                right_child = tree_obj.children_right[node]
                
                left_impurity = tree_obj.impurity[left_child]
                left_samples = tree_obj.n_node_samples[left_child]
                
                right_impurity = tree_obj.impurity[right_child]
                right_samples = tree_obj.n_node_samples[right_child]
                # Weighted impurity decrease
                decrease = n_samples * impurity - (left_samples * left_impurity + 
                        right_samples * right_impurity)
                
                total_importance[feature] += decrease
    total_importance /= total_importance.sum()
    total_importance=sorted(total_importance)

    return total_importance                

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, 
                          n_informative=5, random_state=42)

# Train Random Forest
trees = []
for i in range(50):
    indices = np.random.choice(len(X), len(X), replace=True)
    tree = DecisionTreeClassifier(max_depth=10, max_features='sqrt')
    tree.fit(X[indices], y[indices])
    trees.append(tree)

# Calculate importance
importance = calculate_feature_importance(trees, n_features=10)
print("Feature Importances:")
for i, imp in enumerate(importance):
    print(f"Feature {i}: {imp:.4f}")


# Question 5: Hyperparameter Tuning

# Implement grid search to find best hyperparameters for Random Forest.
# Test different combinations of n_estimators, max_depth, max_features.
# Use cross-validation.

from sklearn.model_selection import cross_val_score
from itertools import product
def random_forest_grid_search(X, y, param_grid, cv=5):
    """
    Grid search for Random Forest
    
    Args:
        X, y: training data
        param_grid: dict like {'n_estimators': [50,100], 'max_depth': [10,20]}
        cv: number of CV folds
    
    Returns:
        best_params, best_score, all_results
    """
    # Your code here
    keys=param_grid.keys()
    values=param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    best_score=-1
    best_params=None
    all_results=[]

    for params in combinations:
        rf=RandomForestClassifier(**params,random_state=42,n_jobs=-1)
        scores=cross_val_score(rf,X,y,cv=cv,scoring='accuracy')
        mean_score=scores.mean()

        all_results.append({'params':params,'score':mean_score,'std':scores.std()})

        if mean_score>best_score:
            best_score=mean_score
            best_params=params

        print(f"Params: {params} → Score: {mean_score:.4f} ± {scores.std():.4f}")

    return best_params, best_score, all_results       
        


# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'max_features': ['sqrt', 'log2', 0.5]
}

best_params, best_score, results = random_forest_grid_search(X, y, param_grid, cv=5)
print(f"Best Parameters: {best_params}")
print(f"Best CV Score: {best_score:.4f}")


"""
Feature selection using random forest


Use Random Forest feature importance to select top K features.
Then train new model on only those features.
Compare performance.
"""
def select_top_k_features(X, y, k=10):
    """
    Select top k most important features using Random Forest
    
    Returns:
        selected_features: indices of top k features
        importances: feature importances
    """
    # Your code here
    rf=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
    rf.fit(X,y)
    importances=rf.feature_importances_
    selected_features=np.argsort(importances)[-k:][::-1]
    return selected_features,importances

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=50, 
                          n_informative=10, n_redundant=40, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Select top 10 features
selected_features, importances = select_top_k_features(X_train, y_train, k=10)

# Train on all features
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train, y_train)
acc_all = accuracy_score(y_test, rf_all.predict(X_test))

# Train on selected features
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train[:, selected_features], y_train)
acc_selected = accuracy_score(y_test, rf_selected.predict(X_test[:, selected_features]))

print(f"All features (50):      {acc_all:.4f}")
print(f"Selected features (10): {acc_selected:.4f}")



from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

# Create dataset with p=100 features
X, y = make_classification(n_samples=1000, n_features=100, 
                          n_informative=20, random_state=42)

# Test different max_features
k_values = [1, 5, 10, 'sqrt', 'log2', 0.5, None]  # sqrt ≈ 10, log2 ≈ 7

results = {}
for k in k_values:
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features=k,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(rf, X, y, cv=5)
    results[str(k)] = scores.mean()
    print(f"max_features={str(k):6s}: {scores.mean():.4f} ± {scores.std():.4f}")


results = {}
# ============================================================
# 4. ADABOOST
# ============================================================
# Sequential boosting:
# Round t:
#   1. Train weak learner h_t on weighted data
#   2. Compute error: eps_t = sum(w_i * I(h_t(x_i) != y_i)) / sum(w_i)
#   3. Compute learner weight: alpha_t = 0.5 * ln((1-eps_t)/eps_t)
#   4. Update sample weights: w_i = w_i * exp(-alpha_t * y_i * h_t(x_i))
#   5. Normalize weights
# Final: H(x) = sign(sum(alpha_t * h_t(x)))
data = load_iris()#load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train) # fit ----> mean and std calculate kore and transforms sei value diye scale kore
X_test_s  = scaler.transform(X_test)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # weak learner = stump
    n_estimators=50,        # T = number of weak learners
    learning_rate=0.8,      # shrinks contribution of each learner .........Controls how much each weak learner contributes to the final prediction
    # algorithm='SAMME',      # SAMME = discrete boosting (classic AdaBoost)
    random_state=42
)
#High Learning Rate (η = 1.0)
# ✓ Fast convergence (few steps needed)
# ✗ Might overshoot the goal
# ✗ Might oscillate around target
# ✗ Risk of overfitting
# → Strong corrections
# → Might overfit to training quirks
# Low Learning Rate
# ✓ Smooth convergence (careful approach)
# ✓ Less likely to overshoot
# ✓ Better generalization
# ✗ Slower (need more models)
# → More conservative
# → Better generalization

# Small η → More regularization → Less overfitting
# Large η → Less regularization → More overfitting

print("X train shape : ",X_train_s.shape)
print("Y train shape :",y_train.shape)
ada.fit(X_train_s, y_train)
ada_acc = accuracy_score(y_test, ada.predict(X_test_s))
results['AdaBoost'] = ada_acc
print(f"\nAdaBoost Accuracy       : {ada_acc:.4f}")

# Staged predictions — see how accuracy improves round by round
staged_accs = [
    accuracy_score(y_test, y_staged)
    for y_staged in ada.staged_predict(X_test_s)
]
print(staged_accs)
learning_rates = [0.1, 0.5, 1.0, 2.0]
results = {}

for lr in learning_rates:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )
    ada.fit(X_train, y_train)
    
    # Staged predictions (round by round)
    train_scores = []
    test_scores = []
    
    for train_pred, test_pred in zip(
        ada.staged_predict(X_train),
        ada.staged_predict(X_test)
    ):
        train_scores.append(accuracy_score(y_train, train_pred))
        test_scores.append(accuracy_score(y_test, test_pred))
    
    results[lr] = {'train': train_scores, 'test': test_scores}

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]
    rounds = range(1, 101)
    
    ax.plot(rounds, results[lr]['train'], label='Train', linewidth=2)
    ax.plot(rounds, results[lr]['test'], label='Test', linewidth=2)
    ax.set_title(f'Learning Rate = {lr}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Estimators')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Annotate final scores
    final_train = results[lr]['train'][-1]
    final_test = results[lr]['test'][-1]
    ax.text(50, 0.5, f'Final Train: {final_train:.3f}\nFinal Test: {final_test:.3f}',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('learning_rate_comparison.png', dpi=150)
plt.show()

# ============================================================
# 5. GRADIENT BOOSTING (GBM)
# ============================================================
# Fits each new tree on the RESIDUALS (pseudo-gradients) of the previous ensemble
# F_m(x) = F_{m-1}(x) + rho_m * h_m(x)
# Where h_m fits the negative gradient of the loss function

gbm = GradientBoostingClassifier(
    n_estimators=100,       # M = number of trees---- risk of overfitting ache 
    learning_rate=0.1,      # shrinkage / step size----slower learning , less overfitting ,need more trees
    max_depth=3,            # depth of each tree (shallow trees = weak learners)---this prevents overfitting
    subsample=0.8,          # stochastic GBM: use 80% of samples per tree------
    #--each tree uses 80% random data -----eta ke bole Stochastic Gradient Boosting
    #Reduces variance,Improves generalization,Adds randomness
    min_samples_split=2, #minimum samples needed to split a node 
    random_state=42
)
gbm.fit(X_train_s, y_train)
gbm_acc = accuracy_score(y_test, gbm.predict(X_test_s))
results['Gradient Boosting'] = gbm_acc
print(f"\nGradient Boosting Acc   : {gbm_acc:.4f}")

# ============================================================
# 6. XGBOOST
# ============================================================
# GBM + Regularization + Second-order Taylor expansion + Parallelization
# Obj = L(y, F(x)) + Omega(h)
# Omega(h) = gamma*T + (1/2)*lambda*||w||^2
# Uses Gain = S_L + S_R - S_P - gamma for splitting (not Gini/Entropy)
data = load_iris()#load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train) # fit ----> mean and std calculate kore and transforms sei value diye scale kore
X_test_s  = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,   # Random Forest trick: random feature subset per tree
    reg_lambda=1.0,         # L2 regularization
    reg_alpha=0.0,          # L1 regularization
    gamma=0,                # min gain required to split
    min_child_weight=1,     # min sum of hessians in leaf
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train_s, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test_s))
results['XGBoost'] = xgb_acc
print(f"\nXGBoost Accuracy        : {xgb_acc:.4f}")



# 1. Load data
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# 4. Set parameters
params = {
    'objective': 'binary:logistic',  # for binary classification
    'eval_metric': 'logloss',        # can also use 'error' or 'auc'
    'eta': 0.1,                      # learning rate
    'max_depth': 4,
    'seed': 42
}

# 5. Train with early stopping
evallist = [(dtrain, 'train'), (dval, 'eval')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,      # maximum number of boosting rounds
    evals=evallist,
    early_stopping_rounds=10,  # stop if eval metric doesn't improve for 10 rounds
    verbose_eval=True
)

# 6. Make predictions
y_pred_prob = bst.predict(dval)
y_pred = (y_pred_prob > 0.5).astype(int)

# 7. Evaluate
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

results['XGBoost + Early Stop'] = accuracy
print(f"XGBoost+EarlyStopping   : {accuracy:.4f} (stopped at iter {bst.best_iteration})")

# ============================================================
# 7. VOTING CLASSIFIER
# ============================================================
# Hard Voting: majority class vote
# Soft Voting: average probability → use class with highest avg prob (better!)

data = load_iris()#load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train) # fit ----> mean and std calculate kore and transforms sei value diye scale kore
X_test_s  = scaler.transform(X_test)


voting_hard = VotingClassifier(
    estimators=[
        ('dt',  DecisionTreeClassifier(max_depth=4, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('lr',  LogisticRegression(max_iter=1000, random_state=42))
    ],
    voting='hard'   # majority vote
)
voting_hard.fit(X_train_s, y_train)
voting_hard_acc = accuracy_score(y_test, voting_hard.predict(X_test_s))
results['Voting (Hard)'] = voting_hard_acc

voting_soft = VotingClassifier(
    estimators=[
        ('dt',  DecisionTreeClassifier(max_depth=4, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('lr',  LogisticRegression(max_iter=1000, random_state=42))
    ],
    voting='soft',  # average probabilities
    weights=[1, 1, 2]  # LogReg gets double weight
)
voting_soft.fit(X_train_s, y_train)
voting_soft_acc = accuracy_score(y_test, voting_soft.predict(X_test_s))
results['Voting (Soft, Weighted)'] = voting_soft_acc
print(f"\nVoting Hard Accuracy    : {voting_hard_acc:.4f}")
print(f"Voting Soft Accuracy    : {voting_soft_acc:.4f}")

# ============================================================
# 8. STACKING
# ============================================================
# Level 0 (Base learners): Train multiple diverse models
# Level 1 (Meta learner):  Train on the OUTPUT of base learners
# Meta learner learns HOW to best combine base learners

data = load_iris()#load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train) # fit ----> mean and std calculate kore and transforms sei value diye scale kore
X_test_s  = scaler.transform(X_test)


stacking = StackingClassifier(
    estimators=[   # Level 0: base learners (diverse!)
        ('dt',  DecisionTreeClassifier(max_depth=4, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(probability=True, kernel='rbf', random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000),  # Level 1: meta learner
    cv=5,           # cross-validation to generate meta features (prevent leakage!)
    passthrough=False  # True = also pass original features to meta learner
)
stacking.fit(X_train_s, y_train)
stacking_acc = accuracy_score(y_test, stacking.predict(X_test_s))
results['Stacking'] = stacking_acc
print(f"\nStacking Accuracy       : {stacking_acc:.4f}")

# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n" + "="*55)
print(f"{'Algorithm':<35} {'Accuracy':>10}")
print("="*55)
for name, acc in sorted(results.items(), key=lambda x: x[1]):
    bar = "█" * int(acc * 20)
    print(f"{name:<35} {acc:.4f}  {bar}")
print("="*55)




"""
====================================================
  6 BOOSTING MODELS — EXAM READY REFERENCE CODE
====================================================
Models covered:
  1. XGBoost          — Credit Card Fraud Detection
  2. AdaBoost         — Mushroom Edibility
  3. CatBoost         — Stroke Prediction
  4. GradientBoosting — Titanic Survival
  5. HistGradientBoosting — Unemployment (Regression)
  6. LightGBM         — Water Potability

NOTE: Each section uses sklearn's make_classification /
      make_regression to generate synthetic data so the
      code runs anywhere without Kaggle datasets.
      On Kaggle just swap the data-loading block with
      pd.read_csv(...) as shown in comments.
====================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.datasets import make_classification, make_regression

# ─────────────────────────────────────────────────────────────────
# 1. XGBoost — Credit Card Fraud Detection (Classification)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. XGBoost — Credit Card Fraud Detection")
print("="*60)

import xgboost as xgb
from sklearn import metrics

# --- Data Loading (Kaggle) ---
# train = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
# df = train.copy()
# df = df.drop(['Time', 'V13', 'V15', 'V22', 'V23', 'V24', 'V25', 'V26', 'V28', 'Amount'], axis=1)
# X = df.drop(['Class'], axis=1)
# Y = df['Class']

# --- Synthetic Data (runs anywhere) ---
X_raw, Y = make_classification(n_samples=1000, n_features=19,
                                n_informative=15, random_state=42)
X_raw = pd.DataFrame(X_raw, columns=[f"V{i}" for i in range(19)])
Y = pd.Series(Y)

# --- Preprocessing ---
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# --- Model ---
model = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.9,
    n_estimators=500,
    eval_metric='logloss'
)

# --- KFold Cross Validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X, Y, cv=kf, scoring='accuracy')
accuracy_values = cv_results
average_accuracy = cv_results.mean()
print("Accuracy for each fold:", accuracy_values)
print("Average Accuracy across all folds:", average_accuracy)

# --- Train/Test Split + Confusion Matrix ---
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, stratify=Y, random_state=42)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
cm = metrics.confusion_matrix(y_test, prediction)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", metrics.classification_report(y_test, prediction))


# ─────────────────────────────────────────────────────────────────
# 2. AdaBoost — Mushroom Edibility (Classification)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. AdaBoost — Mushroom Edibility")
print("="*60)

from sklearn.ensemble import AdaBoostClassifier

# --- Data Loading (Kaggle) ---
# train = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
# label_encoder = LabelEncoder()
# df = train.apply(lambda x: label_encoder.fit_transform(x) if x.dtype == 'O' else x)
# df = df.drop(['veil-type'], axis=1)
# X_train = df.drop(['class'], axis=1)
# Y_train = df['class']

# --- Synthetic Data ---
X_raw, Y_train = make_classification(n_samples=8000, n_features=21,
                                      n_informative=15, random_state=42)
X_raw = pd.DataFrame(X_raw)
Y_train = pd.Series(Y_train)

# --- Preprocessing ---
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# --- Model ---
model = AdaBoostClassifier()

# --- KFold Cross Validation ---
kf = KFold(n_splits=4, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy')
accuracy_values = cv_results
average_accuracy = cv_results.mean()
print("Accuracy for each fold:", accuracy_values)
print("Average Accuracy across all folds:", average_accuracy)


# ─────────────────────────────────────────────────────────────────
# 3. CatBoost — Stroke Prediction (Classification)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. CatBoost — Stroke Prediction")
print("="*60)

from catboost import CatBoostClassifier

# --- Data Loading (Kaggle) ---
# train = pd.read_csv("/kaggle/input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv")
# label_encoder = LabelEncoder()
# df = train.apply(lambda x: label_encoder.fit_transform(x) if x.dtype == 'O' else x)
# df = df.drop(['id', 'gender'], axis=1)
# X_train = df.drop(['stroke'], axis=1)
# Y_train = df['stroke']

# --- Synthetic Data ---
X_raw, Y_train = make_classification(n_samples=5000, n_features=8,
                                      n_informative=6, random_state=42)
X_raw = pd.DataFrame(X_raw, columns=['age', 'hypertension', 'heart_disease',
                                      'ever_married', 'work_type', 'residence_type',
                                      'avg_glucose_level', 'bmi'])
Y_train = pd.Series(Y_train)

# --- Preprocessing ---
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# --- Model ---
model = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.2,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    metric_period=100,   # suppresses per-iteration output
    verbose=False
)

# --- KFold Cross Validation ---
kf = KFold(n_splits=4, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy')
accuracy = cv_results
average_accuracy = cv_results.mean()
print("Accuracy for each fold:", accuracy)
print("Average Accuracy across all folds:", average_accuracy)


# ─────────────────────────────────────────────────────────────────
# 4. GradientBoosting — Titanic Survival (Classification)
#    (Notebook also tested LightGBM & XGBoost on same data;
#     all three are shown here for completeness)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. GradientBoosting — Titanic Survival")
print("="*60)

from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# --- Data Loading (Kaggle) ---
# traindf = pd.read_csv("/kaggle/input/titanic/train.csv")
# testdf  = pd.read_csv("/kaggle/input/titanic/test.csv")
#
# # Null handling
# traindf['Embarked'].fillna(traindf['Embarked'].mode()[0], inplace=True)
# testdf['Fare'].fillna(testdf['Fare'].mean(), inplace=True)
#
# # Feature Engineering — Age grouping
# def group_age(val):
#     if val < 1:    return 0   # infant
#     elif val < 11: return 1   # kid
#     elif val < 18: return 2   # teen
#     elif val < 52: return 3   # adult
#     else:          return 4   # senior
# traindf['Age'] = traindf['Age'].apply(group_age)
# testdf['Age']  = testdf['Age'].apply(group_age)
#
# # Feature Engineering — RelativeNum
# traindf['RelativeNum'] = traindf['SibSp'] + traindf['Parch']
# testdf['RelativeNum']  = testdf['SibSp']  + testdf['Parch']
# def group_rel(val):
#     if val < 1:   return 0   # solo
#     elif val < 4: return 1   # nuclear family
#     elif val < 7: return 2   # large family
#     else:         return 3   # group
# traindf['RelativeNum'] = traindf['RelativeNum'].apply(group_rel)
# testdf['RelativeNum']  = testdf['RelativeNum'].apply(group_rel)
#
# # Fill missing Age using RelativeNum
# def pos_age(row):
#     if pd.isnull(row['Age']):
#         if row['RelativeNum'] < 2:  return 3
#         elif row['RelativeNum']==2: return 1
#         elif row['RelativeNum']==3: return 4
#     return row['Age']
# traindf['Age'] = traindf.apply(pos_age, axis=1)
# testdf['Age']  = testdf.apply(pos_age, axis=1)
#
# # Cabin — has cabin or not
# traindf['Cabin'] = traindf['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
# testdf['Cabin']  = testdf['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
#
# # Fare grouping
# def determine_fare(row):
#     if row['Fare'] <= 10:             return 0
#     elif row['Fare'] <= 16:           return 1
#     elif row['Fare'] <= 55:           return 2
#     else:                             return 3
# traindf['Fare'] = traindf.apply(determine_fare, axis=1)
# testdf['Fare']  = testdf.apply(determine_fare, axis=1)
#
# # Encode + drop useless columns
# encoder = LabelEncoder()
# tren = traindf.apply(lambda i: encoder.fit_transform(i) if i.dtype == 'O' else i)
# tren = tren.drop(['PassengerId','Name','Ticket','SibSp','Parch'], axis=1)
# X_train = tren.drop(['Survived'], axis=1)
# Y_train = tren['Survived']
# testdf  = testdf.apply(lambda i: encoder.fit_transform(i) if i.dtype == 'O' else i)
# testdf  = testdf.drop(['Name','Ticket','SibSp','Parch'], axis=1)
# X_test  = testdf.drop(['PassengerId'], axis=1)
#
# scaler = StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# X_test  = pd.DataFrame(scaler.fit_transform(X_test),  columns=X_test.columns)

# --- Synthetic Data ---
X_raw, Y_train = make_classification(n_samples=891, n_features=7,
                                      n_informative=5, random_state=42)
X_raw = pd.DataFrame(X_raw, columns=['Pclass','Sex','Age','Fare',
                                      'Cabin','Embarked','RelativeNum'])
Y_train = pd.Series(Y_train)
scaler = StandardScaler()
X_train_t = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- LightGBM ---
modelL = LGBMClassifier(verbose=-1)
modelL.fit(X_train_t, Y_train)
cv_results = cross_val_score(modelL, X_train_t, Y_train, cv=kf)
print("LightGBM  — Accuracy per fold:", cv_results,
      " | Avg:", round(cv_results.mean(), 4))

# --- XGBoost ---
modelX = XGBClassifier(eval_metric='logloss')
modelX.fit(X_train_t, Y_train)
cv_results = cross_val_score(modelX, X_train_t, Y_train, cv=kf)
print("XGBoost   — Accuracy per fold:", cv_results,
      " | Avg:", round(cv_results.mean(), 4))

# --- GradientBoosting (main model used in notebook) ---
modelG = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01,
                                     random_state=42)
modelG.fit(X_train_t, Y_train)
cv_results = cross_val_score(modelG, X_train_t, Y_train, cv=kf)
print("GradBoost — Accuracy per fold:", cv_results,
      " | Avg:", round(cv_results.mean(), 4))


# ─────────────────────────────────────────────────────────────────
# 5. HistGradientBoosting — Unemployment 2024 (Regression)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. HistGradientBoosting — Unemployment Prediction (Regression)")
print("="*60)

from sklearn.experimental import enable_hist_gradient_boosting  # needed for older sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

# --- Data Loading (Kaggle) ---
# train = pd.read_csv("/kaggle/input/global-unemployment-data/global_unemployment_data.csv")
# label_encoder = LabelEncoder()
# df = train.apply(lambda x: label_encoder.fit_transform(x) if x.dtype == 'O' else x)
# df = df.dropna(subset=['2024'])
# df = df.drop(['indicator_name'], axis=1)
# X_train = df.drop(['2024'], axis=1)
# Y_train = df['2024']
# scaler = StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# --- Synthetic Data ---
X_raw, Y_train = make_regression(n_samples=1000, n_features=10,
                                  n_informative=7, noise=0.1, random_state=42)
X_raw = pd.DataFrame(X_raw)
Y_train = pd.Series(Y_train)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# --- Model ---
model = HistGradientBoostingRegressor()
model.fit(X_train, Y_train)

# --- KFold Cross Validation (R² Score) ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X_train, Y_train, cv=kf, scoring='r2')
print("R² scores for each fold:", cv_results)
average_r2 = cv_results.mean()
print("Average R² across all folds:", average_r2)


# ─────────────────────────────────────────────────────────────────
# 6. LightGBM — Water Potability (Classification)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. LightGBM — Water Potability")
print("="*60)

import lightgbm as lgb

# --- Data Loading (Kaggle) ---
# train = pd.read_csv("/kaggle/input/water-potability/water_potability.csv")
# X_train = train.drop(['Potability'], axis=1)   # ph & Turbidity kept here
# Y_train = train['Potability']
# scaler = StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# --- Synthetic Data ---
X_raw, Y_train = make_classification(n_samples=3276, n_features=8,
                                      n_informative=6, random_state=42)
X_raw = pd.DataFrame(X_raw, columns=['ph', 'Hardness', 'Solids', 'Chloramines',
                                      'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes'])
Y_train = pd.Series(Y_train)

# --- Preprocessing ---
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# --- Model ---
model = lgb.LGBMClassifier(verbose=-1)
model.fit(X_train, Y_train)

# --- KFold Cross Validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy')
accuracy_values = cv_results
average_accuracy = cv_results.mean()
print("Accuracy for each fold:", accuracy_values)
print("Average Accuracy across all folds:", average_accuracy)

print("\n" + "="*60)
print("All 6 models completed successfully!")
print("="*60)
