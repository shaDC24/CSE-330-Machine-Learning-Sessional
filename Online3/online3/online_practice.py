"""

Problem 1.1: Bootstrap Sampling Verification


Implement a function to verify that bootstrap sampling 
is working correctly.

Given a dataset of n samples:
1. Create a bootstrap sample
2. Calculate what percentage of original samples are included (should be ~63%)
3. Calculate what percentage are duplicated
4. Return statistics

Expected: ~63% samples included, ~37% left out (Out-of-Bag)
"""

import numpy as np

def bootstrap_statistics(n_samples=1000, n_iterations=100):
    """
    Calculate bootstrap sampling statistics
    
    Args:
        n_samples: size of dataset
        n_iterations: number of bootstrap samples to analyze
    
    Returns:
        dict with:
            - avg_unique_pct: average % of unique samples in bootstrap
            - avg_oob_pct: average % of OOB samples
            - avg_duplicates: average number of duplicated samples
    """
    # YOUR CODE HERE
    unique_pcts=[]
    oob_pcts=[]
    duplicate_counts=[]
    for i in range(n_iterations):
        indices=np.random.choice(n_samples,size=n_samples,replace=True)
        unique_sample=len(np.unique(indices))
        unique_pct=(unique_sample/n_samples)*100
        oob_pct=((n_samples-unique_sample)/n_samples)*100
        duplicates=(n_samples-unique_sample)

        unique_pcts.append(unique_pct)
        oob_pcts.append(oob_pct)
        duplicate_counts.append(duplicates)
    return {
        'avg_unique_pct':np.mean(unique_pcts),
        'avg_oob_pct':np.mean(oob_pcts),
        'avg_duplicates':np.mean(duplicate_counts)
    }    

# Test
stats = bootstrap_statistics(n_samples=1000, n_iterations=1000)
print(f"Average unique samples: {stats['avg_unique_pct']:.2f}%")
print(f"Average OOB samples: {stats['avg_oob_pct']:.2f}%")
print(f"Average duplicates: {stats['avg_duplicates']:.1f}")

# Expected output:
# Average unique samples: ~63.2%
# Average OOB samples: ~36.8%
# Average duplicates: ~368


"""
Problem 1.2: Simple Majority Voting


Implement majority voting for classification ensemble.
Given predictions from multiple models, return the majority vote.
Handle ties by returning the class with lowest index.
"""

def majority_vote(predictions):
    """
    Args:
        predictions: 2D array (n_models, n_samples)
                    Each row is predictions from one model
    
    Returns:
        1D array of majority votes (n_samples,)
    
    Example:
        predictions = [[0, 1, 1, 0],
                      [0, 1, 0, 0],
                      [1, 1, 0, 0]]
        
        Sample 0: [0,0,1] → 0 wins (2 votes)
        Sample 1: [1,1,1] → 1 wins (3 votes)
        Sample 2: [1,0,0] → 0 wins (2 votes)
        Sample 3: [0,0,0] → 0 wins (3 votes)
        
        Output: [0, 1, 0, 0]
    """
    n_models, n_samples=predictions.shape
    majority_votes=[]
    # YOUR CODE HERE
    for i in range(n_samples):
        sample_data=predictions[:,i]
        vote_count=np.bincount(sample_data)

        majority=np.argmax(vote_count)
        majority_votes.append(majority)
    return np.array(majority_votes)    

# Test
predictions = np.array([
    [0, 1, 1, 0, 1],
    [0, 1, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [0, 0, 0, 1, 1]
])

result = majority_vote(predictions)
print(f"Majority votes: {result}")
# Expected: [0, 1, 0, 0, 1]



"""
Problem 1.3: Bagging Single Round

Implement ONE round of bagging:
1. Create bootstrap sample
2. Train one decision tree
3. Make predictions
4. Return tree and predictions
"""

from sklearn.tree import DecisionTreeClassifier

def bagging_single_round(X_train, y_train, X_test, max_depth=None, random_state=None):
    """
    Perform one round of bagging
    
    Returns:
        tree: trained DecisionTree
        predictions: predictions on X_test
        oob_indices: indices of OOB samples
    """
    # YOUR CODE HERE
    n_samples=len(X_train)
    if random_state is not None:
        np.random.seed(random_state)
    #create bootstrap samples    
    indices=np.random.choice(n_samples,size=n_samples,replace=True)
    all_indices=set(range(n_samples))
    bootstrap_set=set(indices)
    oob_indices=list(all_indices-bootstrap_set)

    X_boot=X_train[indices]
    y_boot=y_train[indices]
    tree=DecisionTreeClassifier(max_depth=max_depth,random_state=random_state)
    tree.fit(X_boot,y_boot)
    y_pred=tree.predict(X_test)

    return tree,y_pred,oob_indices 

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

tree, preds, oob = bagging_single_round(X[:80], y[:80], X[80:], max_depth=5, random_state=42)
print(f"Predictions shape: {preds.shape}")
print(f"OOB samples: {len(oob)}")
print(f"First 5 predictions: {preds[:5]}")



"""
Problem 1.4: Calculate Model Weights (AdaBoost Style)

In AdaBoost, each model gets a weight based on its error.
Implement the weight calculation:

α = 0.5 * ln((1 - error) / error)

where error = weighted error rate
"""

def calculate_model_weight(error):
    """
    Calculate AdaBoost model weight
    
    Args:
        error: weighted error rate (0 < error < 1)
    
    Returns:
        alpha: model weight
    
    Note: If error = 0, return large value (e.g., 10)
          If error >= 0.5, return 0 (worse than random)
    """
    # YOUR CODE HERE
    if error == 0 :
        return 10.0
    elif error >= 0.5:
        return 0.0
    else :
        return (0.5*(np.log((1-error)/error)))

# Test cases
test_errors = [0.1, 0.2, 0.3, 0.4, 0.49, 0.5, 0.6]
for err in test_errors:
    alpha = calculate_model_weight(err)
    print(f"Error: {err:.2f} → Alpha: {alpha:.4f}")

# Expected:
# Error: 0.10 → Alpha: 2.1972
# Error: 0.20 → Alpha: 1.3863
# Error: 0.30 → Alpha: 0.8473
# Error: 0.40 → Alpha: 0.4055
# Error: 0.49 → Alpha: 0.0202
# Error: 0.50 → Alpha: 0.0000
# Error: 0.60 → Alpha: 0.0000

"""
Problem 1.5: Feature Importance Ranking


Given feature importances from Random Forest,
return indices sorted by importance (descending).
"""

def rank_features(importances, feature_names=None):
    """
    Rank features by importance
    
    Args:
        importances: array of feature importances
        feature_names: optional list of feature names
    
    Returns:
        list of tuples: [(feature_name/index, importance), ...]
        sorted by importance (descending)
    """
    # YOUR CODE HERE

    if feature_names is None :
        feature_names=[f"feature_{i}" for i in range(len(importances))]
    sorted_indices=np.argsort(importances)[::-1]
    ranked=[(feature_names[i],importances[i]) for i in sorted_indices]    
    return ranked

# Test
importances = np.array([0.05, 0.30, 0.10, 0.25, 0.15, 0.15])
feature_names = ['feature_A', 'feature_B', 'feature_C', 'feature_D', 'feature_E', 'feature_F']

ranked = rank_features(importances, feature_names)
print("Feature Ranking:")
for i, (name, imp) in enumerate(ranked, 1):
    print(f"{i}. {name}: {imp:.4f}")

# Expected:
# 1. feature_B: 0.3000
# 2. feature_D: 0.2500
# 3. feature_E: 0.1500
# 4. feature_F: 0.1500
# 5. feature_C: 0.1000
# 6. feature_A: 0.0500


"""
Problem 2.1: Implement Simple Bagging Classifier
Implement a simplified BaggingClassifier from scratch.
Should support:
- Bootstrap sampling
- Multiple base estimators
- Majority voting
- fit() and predict() methods
"""

class SimpleBagging:
    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        """
        Args:
            base_estimator: sklearn estimator (e.g., DecisionTreeClassifier)
            n_estimators: number of models in ensemble
            random_state: for reproducibility
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.bootstrap_indices_ = []
    
    def fit(self, X, y):
        """Train n_estimators on bootstrap samples"""
        # YOUR CODE HERE
        n_smaples=len(X)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        for i in range(self.n_estimators):
            indices=np.random.choice(n_smaples,size=n_smaples,replace=True)
            self.bootstrap_indices_.append(indices)
            X_boot=X[indices]
            y_boot=y[indices]


            from sklearn.base import clone
            estimator=clone(self.base_estimator)    
            estimator.fit(X_boot,y_boot)
            self.estimators_.append(estimator)
        return self    
    

    def predict(self, X):
        """Predict using majority vote"""
        # YOUR CODE HERE
        predictions=np.array([est.predict(X) for est in self.estimators_])
        n_samples=X.shape[0]
        final_predictions=[]
        for i in range(n_samples):
            sample_pred=predictions[:,i]
            majority=np.bincount(sample_pred).argmax()
            final_predictions.append(majority)
        return np.array(final_predictions)    

    
    def score(self, X, y):
        """Calculate accuracy"""
        # YOUR CODE HERE
        predictions=self.predict(X)
        return accuracy_score(y,predictions)

# Test
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Your implementation
bagging = SimpleBagging(
    base_estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=20,
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Simple Bagging Accuracy: {acc:.4f}")

# Compare with sklearn
from sklearn.ensemble import BaggingClassifier
sklearn_bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=20,
    random_state=42
)
sklearn_bag.fit(X_train, y_train)
sklearn_acc = sklearn_bag.score(X_test, y_test)

print(f"Sklearn Bagging Accuracy: {sklearn_acc:.4f}")
print(f"Difference: {abs(acc - sklearn_acc):.4f}")
# Should be very close (< 0.01 difference)


"""
Problem 2.2: OOB Score Calculator

Implement OOB (Out-of-Bag) score calculation for bagging ensemble.
For each sample, use only trees where it was NOT in training (OOB).
"""

def calculate_oob_score(X, y, estimators, bootstrap_indices):
    """
    Calculate Out-of-Bag accuracy
    
    Args:
        X: feature matrix (n_samples, n_features)
        y: true labels (n_samples,)
        estimators: list of trained models
        bootstrap_indices: list of arrays, indices used for each model
    
    Returns:
        oob_score: accuracy using OOB predictions
        oob_predictions: OOB predictions for each sample (or -1 if never OOB)
    """
    # YOUR CODE HERE
    n_samples=len(X)
    oob_predictions=np.full(n_samples,-1)
    oob_counts=np.zeros(n_samples)
    for sample_idx in range(n_samples):
        predictions_for_sample=[]
        for tree_idx , tree in enumerate(bootstrap_indices):
            if sample_idx not in tree:
                pred=estimators[tree_idx].predict(X[sample_idx:sample_idx+1])[0]
                predictions_for_sample.append(pred)
        if len(predictions_for_sample)>0:
            vote_counts=np.bincount(predictions_for_sample)
            oob_predictions[sample_idx]=np.argmax(vote_counts)
            oob_counts[sample_idx]=len(predictions_for_sample)
    mask= (oob_predictions!=-1)
    if np.sum(mask)==0:
        return 0.0,oob_predictions
    oob_score=accuracy_score(y[mask],oob_predictions[mask])
    return oob_score,oob_predictions                



# Test with your SimpleBagging from Problem 2.1
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Train bagging
bagging = SimpleBagging(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bagging.fit(X, y)

# Calculate OOB score
oob_score, oob_preds = calculate_oob_score(
    X, y, 
    bagging.estimators_, 
    bagging.bootstrap_indices_
)

print(f"OOB Score: {oob_score:.4f}")
print(f"Samples with OOB predictions: {np.sum(oob_preds != -1)}/{len(y)}")

# Compare with sklearn's OOB
from sklearn.ensemble import BaggingClassifier
sklearn_bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    oob_score=True,
    random_state=42
)
sklearn_bag.fit(X, y)
print(f"Sklearn OOB Score: {sklearn_bag.oob_score_:.4f}")



"""
Problem 2.3: Random Feature Selector


Implement random feature selection for Random Forest.
At each node split, randomly select k features (without replacement).
"""

class RandomFeatureSelector:
    def __init__(self, n_features, max_features='sqrt'):
        """
        Args:
            n_features: total number of features
            max_features: 'sqrt', 'log2', int, or float
        """
        self.n_features = n_features
        self.max_features = max_features
        self.k = self._calculate_k()
    
    def _calculate_k(self):
        """Calculate number of features to select"""
        # YOUR CODE HERE
        if self.max_features== 'sqrt' :
            return int(np.sqrt(self.n_features))
        elif self.max_features=='log2':
            return int(np.log2(self.n_features))
        elif isinstance(self.max_features,int):
            return min(self.max_features,self.n_features)
        elif isinstance(self.max_features,float):
            return min(self.max_features,self.n_features)
        
    
    def select_features(self, random_state=None):
        """
        Randomly select k features
        
        Returns:
            array of k feature indices
        """
        # YOUR CODE HERE
        
        if random_state is not None:
            np.random.seed(random_state)
        selected=np.random.choice(self.n_features,size=self.k,replace = False)
        return np.sort(selected)    

# Test
selector_sqrt = RandomFeatureSelector(n_features=100, max_features='sqrt')
print(f"Total features: 100")
print(f"max_features='sqrt' → k = {selector_sqrt.k}")

for i in range(3):
    features = selector_sqrt.select_features(random_state=i)
    print(f"Selection {i+1}: {features[:5]}... (showing first 5)")

# Test other options
selector_log2 = RandomFeatureSelector(n_features=100, max_features='log2')
print(f"\nmax_features='log2' → k = {selector_log2.k}")

selector_int = RandomFeatureSelector(n_features=100, max_features=20)
print(f"max_features=20 → k = {selector_int.k}")

selector_float = RandomFeatureSelector(n_features=100, max_features=0.3)
print(f"max_features=0.3 → k = {selector_float.k}")

"""
Problem 2.4: AdaBoost Sample Weight Updater

Implement sample weight update for AdaBoost.
After each round, increase weights of misclassified samples.
"""

def update_sample_weights(y_true, y_pred, sample_weights, alpha, learning_rate=1.0):
    """
    Update sample weights for next AdaBoost round
    
    Args:
        y_true: true labels (n_samples,)
        y_pred: predictions (n_samples,)
        sample_weights: current weights (n_samples,)
        alpha: model weight for this round
        learning_rate: shrinkage parameter
    
    Returns:
        new_weights: updated and normalized weights
    
    Formula:
        w_i = w_i * exp(learning_rate * alpha * I(y_i != ŷ_i))
        then normalize so sum = 1
    """
    # YOUR CODE HERE
    incorrect=(y_true!=y_pred).astype(int)
    new_weights=sample_weights*np.exp(learning_rate*alpha*incorrect)
    new_weights/=np.sum(new_weights)
    return new_weights

# Test
np.random.seed(42)
y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])  # 2 mistakes (indices 2, 5)

sample_weights = np.ones(8) / 8  # Initial uniform weights
alpha = 0.65  # Model weight

print("Initial weights:", sample_weights)
print("Mistakes at indices:", np.where(y_true != y_pred)[0])

# Update with different learning rates
for lr in [1.0, 0.5, 0.1]:
    new_weights = update_sample_weights(y_true, y_pred, sample_weights, alpha, learning_rate=lr)
    print(f"\nLearning rate = {lr}")
    print(f"New weights: {new_weights}")
    print(f"Weight on mistakes: {new_weights[[2,5]]}")
    print(f"Weight on correct: {new_weights[[0,1,3,4,6,7]][:3]}...")




"""
Problem 2.5: Voting Classifier (Hard & Soft)

Implement a voting classifier that supports both hard and soft voting.
"""
class VotingClassifier:
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
        self.estimators_ = []

    def fit(self, X, y):
        from sklearn.base import clone
        self.estimators_ = []
        for name, estimator in self.estimators:
            est = clone(estimator)
            est.fit(X, y)
            self.estimators_.append((name, est))
        return self

    def predict(self, X):

        if self.voting == 'hard':
            predictions = np.array([
                est.predict(X) for name, est in self.estimators_
            ])

            final_preds = []
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                majority = np.bincount(votes).argmax()
                final_preds.append(majority)

            return np.array(final_preds)

        elif self.voting == 'soft':
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        probas = np.array([
            est.predict_proba(X) for name, est in self.estimators_
        ])
        return np.mean(probas, axis=0)

# Test
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# Define estimators
estimators = [
    ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('nb', GaussianNB())
]

# Hard voting
voting_hard = VotingClassifier(estimators, voting='hard')
voting_hard.fit(X_train, y_train)
hard_acc = accuracy_score(y_test, voting_hard.predict(X_test))
# Soft voting
voting_soft = VotingClassifier(estimators, voting='soft')
voting_soft.fit(X_train, y_train)

soft_acc = accuracy_score(y_test, voting_soft.predict(X_test))

print(f"Hard Voting Accuracy: {hard_acc:.4f}")
print(f"Soft Voting Accuracy: {soft_acc:.4f}")

# Individual model accuracies
# Individual model accuracies
from sklearn.base import clone
for name, est in estimators:
    est_clone = clone(est)
    est_clone.fit(X_train, y_train)
    acc = accuracy_score(y_test, est_clone.predict(X_test))
    print(f"{name} alone: {acc:.4f}")



"""
Problem 2.6: Feature Importance Aggregation


Calculate aggregated feature importance across multiple Random Forests.
Train multiple forests with different random seeds, then average importances.
"""
from sklearn.ensemble import RandomForestClassifier
def aggregate_feature_importance(X, y, n_forests=10, n_estimators=100):
    """
    Train multiple Random Forests and aggregate feature importances
    
    Returns:
        mean_importance: mean importance for each feature
        std_importance: standard deviation
        cv_importance: coefficient of variation (std/mean)
    """
    n_features=X.shape[1]
    all_importances=[]
    for i in range(n_forests):
        rf=RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=i,
            n_jobs=-1
        )
        rf.fit(X,y)
        all_importances.append(rf.feature_importances_)

        mean_importance=np.mean(all_importances,axis=0)
        std_importance=np.mean(all_importances,axis=0)

        cv_importance=std_importance/(mean_importance+1e-10)
    return mean_importance,std_importance,cv_importance    
     

# Test
from sklearn.datasets import load_wine
wine = load_wine()
X, y = wine.data, wine.target

mean_imp, std_imp, cv_imp = aggregate_feature_importance(X, y, n_forests=20, n_estimators=50)

# Print results
print("Feature Importances (aggregated over 20 forests):\n")
for i, (mean, std, cv) in enumerate(zip(mean_imp, std_imp, cv_imp)):
    print(f"Feature {i} ({wine.feature_names[i]:25s}): "
          f"{mean:.4f} ± {std:.4f} (CV={cv:.2f})")

# Find most stable features (low CV)
stable_idx = np.argsort(cv_imp)[:3]
print(f"\nMost stable features (low CV):")
for idx in stable_idx:
    print(f"  {wine.feature_names[idx]}: CV={cv_imp[idx]:.3f}")


"""
Problem 2.7: Early Stopping for Boosting


Implement early stopping for boosting algorithm.
Stop training when validation score stops improving.
"""
from sklearn.ensemble import AdaBoostClassifier

def train_boosting_with_early_stopping(X_train, y_train, X_val, y_val, 
                                       max_estimators=200, patience=10):
    """
    Train AdaBoost with early stopping
    
    Args:
        patience: number of rounds to wait for improvement
    
    Returns:
        best_estimator: model with best validation score
        best_n_estimators: optimal number of estimators
        train_scores: list of training scores
        val_scores: list of validation scores
    """
    # YOUR CODE HERE
    # Hint: Use staged_predict() or train incrementally
    ada=AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=max_estimators,
        random_state=42

                        
    )
    ada.fit(X_train,y_train)
    train_scores=[]
    val_scores=[]
    for train_pred,val_pred in zip(ada.staged_predict(X_train),ada.staged_predict(X_val)):
        train_scores.append(accuracy_score(y_train,train_pred))
        val_scores.append(accuracy_score(y_val,val_pred))

    best_score=0
    best_n=0
    patience_counter=0
    for i,score in enumerate(val_scores):
        if score>best_score:
            best_score=score
            best_n=i+1
            patience_counter=0
        else:
            patience_counter+=1
        if patience_counter>=patience :
            break
    best_estimator=AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=best_n,
        random_state=42
    )                
    best_estimator.fit(X_train,y_train)
    return best_estimator,best_n,train_scores,val_scores

# Test
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

best_model, best_n, train_scores, val_scores = train_boosting_with_early_stopping(
    X_train, y_train, X_val, y_val, max_estimators=200, patience=10
)

print(f"Best n_estimators: {best_n}")
print(f"Best validation score: {max(val_scores):.4f}")
print(f"Test score: {best_model.score(X_test, y_test):.4f}")

# Plot learning curves
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(train_scores, label='Train')
plt.plot(val_scores, label='Validation')
plt.axvline(x=best_n, color='r', linestyle='--', label=f'Best ({best_n})')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Early Stopping in Boosting')
plt.show()    



"""
Problem 2.8: Ensemble Diversity Measurement


Measure diversity in an ensemble using multiple metrics:
1. Q-statistic (pairwise diversity)
2. Disagreement measure
3. Double-fault measure
"""

def calculate_ensemble_diversity(predictions):
    """
    Calculate diversity metrics for ensemble
    
    Args:
        predictions: 2D array (n_models, n_samples)
                    Binary predictions from each model
    
    Returns:
        dict with:
            - q_statistic: average pairwise Q-statistic
            - disagreement: average pairwise disagreement
            - double_fault: average pairwise double-fault
    
    Q-statistic formula for models i, j:
        Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)
        
    where:
        N11 = both correct
        N00 = both wrong
        N01 = i correct, j wrong
        N10 = i wrong, j correct
    """
    # YOUR CODE HERE
    n_models,n_samples=predictions.shape
    q_state=[]
    disagreements=[]
    double_faults=[]

    for i in range(n_models):
        for j in range(i+1,n_models):
            pred_i=predictions[i]
            pred_j=predictions[j]
            both_correct=np.sum((pred_i==1) & (pred_j==1))
            both_wrong=np.sum((pred_i==0)&(pred_j==0))
            i_correct_j_wrong=np.sum((pred_i==1) & (pred_j==0))
            i_cwrong_j_correct=np.sum((pred_i==0) & (pred_j==1))


            numerator=both_correct*both_wrong-i_correct_j_wrong*i_cwrong_j_correct
            denominator=both_correct*both_wrong+i_correct_j_wrong*i_cwrong_j_correct


            if denominator!=0:
                q=numerator/denominator
            else :
                q=0
            q_state.append(q)
            disagr=(i_correct_j_wrong+i_cwrong_j_correct)/n_samples
            disagreements.append(disagr)
            df=both_wrong/n_samples
            double_faults.append(df)    
    return {
        'q_statistic':np.mean(q_state),
        'disagreement':np.mean(disagreements),
        'double_fault':np.mean(double_faults)

    }            



# Test
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train different ensembles
rf = RandomForestClassifier(n_estimators=20, random_state=42)
bag = BaggingClassifier(n_estimators=20, random_state=42)
ada = AdaBoostClassifier(n_estimators=20, random_state=42)

rf.fit(X_train, y_train)
bag.fit(X_train, y_train)
ada.fit(X_train, y_train)

# Get predictions from individual estimators
rf_preds = np.array([est.predict(X_test) for est in rf.estimators_])
bag_preds = np.array([est.predict(X_test) for est in bag.estimators_])
ada_preds = np.array([est.predict(X_test) for est in ada.estimators_])

# Calculate diversity
rf_div = calculate_ensemble_diversity(rf_preds)
bag_div = calculate_ensemble_diversity(bag_preds)
ada_div = calculate_ensemble_diversity(ada_preds)

print("Ensemble Diversity Metrics:\n")
for name, div in [('Random Forest', rf_div), ('Bagging', bag_div), ('AdaBoost', ada_div)]:
    print(f"{name}:")
    print(f"  Q-statistic: {div['q_statistic']:.4f}")
    print(f"  Disagreement: {div['disagreement']:.4f}")
    print(f"  Double-fault: {div['double_fault']:.4f}")
    print()




"""
Problem 2.9: Weighted Soft Voting

Implement weighted soft voting where each model has different weight
based on its validation performance.
"""

def weighted_soft_voting(models, X_val, y_val, X_test):
    """
    Perform weighted soft voting
    
    Args:
        models: list of trained classifiers
        X_val, y_val: validation data for weight calculation
        X_test: test data for prediction
    
    Returns:
        predictions: weighted averaged predictions
        weights: calculated weight for each model
    
    Weight calculation:
        w_i = accuracy_i / sum(accuracies)
    """
    # YOUR CODE HERE
    accuracies=[]
    
   
    for model in models:
        acc=accuracy_score(y_val,model.predict(X_val))
        accuracies.append(acc)
    weights=(np.array(accuracies)/np.sum(accuracies))*1.0
    all_probabilities=np.array([model.predict_proba(X_test) for model in models])
    weightedd_probs=np.tensordot(weights,all_probabilities,axes=([0],[0]))
    predictions=np.argmax(weightedd_probs,axis=1)
    print(predictions.shape)
    print(weights.shape)
    return predictions,weights    

# Test
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
X_train, X_temp, y_train, y_temp = train_test_split(
    digits.data, digits.target, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Train diverse models
models = [
    DecisionTreeClassifier(max_depth=20, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
    SVC(probability=True, random_state=42),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=5)
]

print(X_train.shape)
print(y_train.shape)

for model in models:
    model.fit(X_train, y_train)

# Weighted voting
predictions, weights = weighted_soft_voting(models, X_val, y_val, X_test)

print("Model Weights (based on validation accuracy):")
model_names = ['DecisionTree', 'LogisticRegression', 'SVC', 'GaussianNB', 'KNN']
for name, weight in zip(model_names, weights):
    print(f"  {name:20s}: {weight:.4f}")

# Calculate accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)
print(f"\nWeighted Voting Accuracy: {acc:.4f}")

# Compare with equal weights
equal_preds = np.array([model.predict(X_test) for model in models])
equal_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=equal_preds)
equal_acc = accuracy_score(y_test, equal_vote)
print(f"Equal Voting Accuracy: {equal_acc:.4f}")  



"""
Problem 2.10: Custom Bagging with Feature Sampling

Implement bagging with BOTH sample and feature sampling (Random Patches).
"""

class RandomPatchesClassifier:
    def __init__(self, base_estimator, n_estimators=10, 
                 max_samples=0.8, max_features=0.8, random_state=None):
        """
        Random Patches: Bootstrap both samples AND features
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []
        self.feature_subsets_ = []
    
    def fit(self, X, y):
        """
        Train models on random patches (sample + feature subsets)
        """
        # YOUR CODE HERE
        n_samples,n_features=X.shape
        n_sample_subset=int(self.max_samples*n_samples)
        n_feature_subset=int(self.max_features*n_features)

        for i in range(self.n_estimators):
            sample_indices=np.random.choice(n_samples,size=n_sample_subset,replace=True)
            feature_indices=np.random.choice(n_features,size=n_feature_subset,replace=False)
            X_subset=X[np.ix_(sample_indices,feature_indices)]
            y_subset=y[sample_indices]
            estimator=clone(self.base_estimator)
            estimator.fit(X_subset,y_subset)
            self.estimators_.append(estimator)
            self.feature_subsets_.append(feature_indices
                                         )
        return self    
    
    def predict(self, X):
        """
        Predict using all estimators
        """
        predictions = []
        
        for estimator, features in zip(self.estimators_, self.feature_subsets_):
            # Use only the features this estimator was trained on
            X_subset = X[:, features]
            pred = estimator.predict(X_subset)
            predictions.append(pred)
        
        # Majority vote
        predictions = np.array(predictions)
        n_samples = X.shape[0]
        final_preds = []
        
        for i in range(n_samples):
            votes = predictions[:, i]
            majority = np.bincount(votes).argmax()
            final_preds.append(majority)
        
        return np.array(final_preds)

# Test
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=1000, n_features=50, 
                          n_informative=30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Random Patches
rp = RandomPatchesClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=50,
    max_samples=0.7,
    max_features=0.6,
    random_state=42
)
rp.fit(X_train, y_train)
rp_acc = accuracy_score(y_test, rp.predict(X_test))

# Compare with standard bagging (no feature sampling)
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=50,
    max_samples=0.7,
    random_state=42
)
bag.fit(X_train, y_train)
bag_acc = bag.score(X_test, y_test)

print(f"Random Patches Accuracy: {rp_acc:.4f}")
print(f"Standard Bagging Accuracy: {bag_acc:.4f}")
print(f"Improvement: {(rp_acc - bag_acc)*100:.2f}%")

















