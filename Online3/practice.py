"""
============================================================
  ENSEMBLE LEARNING — Practice Questions with Answers
  Beginner → Intermediate → Advanced + Intellectual
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Shared setup
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

print("="*60)
print("  BEGINNER QUESTIONS")
print("="*60)

# -------------------------------------------------------
# Q1. Train a Decision Tree. Print train & test accuracy.
#     Why might train acc > test acc?
# -------------------------------------------------------
print("\nQ1: Decision Tree — Train vs Test Accuracy")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_tr, y_train)
train_acc = accuracy_score(y_train, dt.predict(X_tr))
test_acc  = accuracy_score(y_test,  dt.predict(X_te))
print(f"  Train Accuracy : {train_acc:.4f}")
print(f"  Test Accuracy  : {test_acc:.4f}")
print(f"  Gap            : {train_acc - test_acc:.4f}")
# Answer: Full tree memorizes training data → OVERFITTING
# High train acc + low test acc = HIGH VARIANCE model

# -------------------------------------------------------
# Q2. Apply Bagging on the same Decision Tree.
#     Does test accuracy improve? Why?
# -------------------------------------------------------
print("\nQ2: Bagging — Does it help?")
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,
    bootstrap=True,   # with replacement = Bagging
    random_state=42
)
bag.fit(X_tr, y_train)
bag_acc = accuracy_score(y_test, bag.predict(X_te))
print(f"  Bagging Accuracy : {bag_acc:.4f}")
print(f"  Improved by      : {bag_acc - test_acc:.4f}")
# Answer: Yes! Bagging reduces VARIANCE by averaging many trees.
# Each tree sees different bootstrap sample → diversity → lower error.

# -------------------------------------------------------
# Q3. Train a Random Forest. What is OOB score?
#     How is it different from test accuracy?
# -------------------------------------------------------
print("\nQ3: Random Forest + OOB Score")
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,    # use out-of-bag samples for free validation
    random_state=42
)
rf.fit(X_tr, y_train)
rf_test_acc = accuracy_score(y_test, rf.predict(X_te))
print(f"  RF Test Accuracy : {rf_test_acc:.4f}")
print(f"  OOB Score        : {rf.oob_score_:.4f}")
# Answer: OOB score uses the ~37% samples NOT in each bootstrap sample
# as a "free" validation set. No need for separate test split!
# It's approximately equal to cross-validation score.

# -------------------------------------------------------
# Q4. Train AdaBoost with n_estimators=1 vs 50 vs 200.
#     What do you observe?
# -------------------------------------------------------
print("\nQ4: AdaBoost — Effect of n_estimators")
for n in [1, 10, 50, 200]:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n, random_state=42
    )
    ada.fit(X_tr, y_train)
    acc = accuracy_score(y_test, ada.predict(X_te))
    print(f"  n_estimators={n:3d} → Accuracy: {acc:.4f}")
# Answer: More estimators = better accuracy (up to a point).
# AdaBoost is resistant to overfitting with more rounds
# because each new learner focuses on remaining hard examples.

print("\n" + "="*60)
print("  INTERMEDIATE QUESTIONS")
print("="*60)

# -------------------------------------------------------
# Q5. Implement Bagging FROM SCRATCH (no sklearn BaggingClassifier)
#     Using only resample + DecisionTreeClassifier
# -------------------------------------------------------
print("\nQ5: Bagging from Scratch")
from sklearn.utils import resample
from scipy import stats

def manual_bagging(X_train, y_train, X_test, n_estimators=50, random_state=42):
    np.random.seed(random_state)
    predictions = []

    for i in range(n_estimators):
        # Step 1: Bootstrap sample (with replacement)
        X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=i)

        # Step 2: Train base learner on bootstrap sample
        tree = DecisionTreeClassifier(random_state=i)
        tree.fit(X_boot, y_boot)

        # Step 3: Predict on test set
        predictions.append(tree.predict(X_test))

    # Step 4: Majority voting
    predictions = np.array(predictions)   # shape: (n_estimators, n_test)
    final_pred, _ = stats.mode(predictions, axis=0)
    return final_pred.flatten()

manual_pred = manual_bagging(X_tr, y_train.values, X_te, n_estimators=50)
manual_acc  = accuracy_score(y_test, manual_pred)
print(f"  Manual Bagging Accuracy  : {manual_acc:.4f}")
print(f"  Sklearn Bagging Accuracy : {bag_acc:.4f}")

# -------------------------------------------------------
# Q6. Implement AdaBoost weight update logic manually.
#     Show how weights change after first round.
# -------------------------------------------------------
print("\nQ6: AdaBoost — Manual Weight Update (Round 1)")

n = len(X_tr)

# Step 1: Initialize equal weights
w = np.ones(n) / n

# Step 2: Train weak learner
h = DecisionTreeClassifier(max_depth=1, random_state=42)
h.fit(X_tr, y_train, sample_weight=w)  # weighted training!

# Step 3: Compute weighted error
y_pred_train = h.predict(X_tr)
y_arr = y_train.values
misclassified = (y_pred_train != y_arr).astype(float)
eps = np.sum(w * misclassified) / np.sum(w)
print(f"  Weighted Error (eps)  : {eps:.4f}")

# Step 4: Compute learner weight alpha
alpha = 0.5 * np.log((1 - eps) / (eps + 1e-10))
print(f"  Learner Weight (alpha): {alpha:.4f}")
# High alpha = good learner (low error)

# Step 5: Update sample weights
# y in {-1, +1} for AdaBoost math
y_signed = 2 * y_arr - 1        # convert 0/1 → -1/+1
h_signed = 2 * y_pred_train - 1
w_new = w * np.exp(-alpha * y_signed * h_signed)
w_new = w_new / w_new.sum()     # normalize

print(f"  Avg weight before     : {w.mean():.6f}")
print(f"  Avg weight after      : {w_new.mean():.6f}")
print(f"  Max weight (wrong)    : {w_new.max():.6f}")
print(f"  Min weight (correct)  : {w_new.min():.6f}")
# Wrong samples get higher weight → next learner focuses on them

# -------------------------------------------------------
# Q7. Compare XGBoost hyperparameter configurations.
#     Find best and worst. Explain WHY each config performs as it does.
# -------------------------------------------------------
print("\nQ7: XGBoost — Best vs Worst Config")

configs = {
    "Underfit (1 tree, slow lr)": {
        "n_estimators": 1, "max_depth": 1,
        "learning_rate": 0.001, "reg_lambda": 100
    },
    "Overfit (deep, no reg)": {
        "n_estimators": 500, "max_depth": 15,
        "learning_rate": 0.5, "reg_lambda": 0
    },
    "Balanced (good config)": {
        "n_estimators": 200, "max_depth": 4,
        "learning_rate": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 1
    },
}

for name, params in configs.items():
    m = xgb.XGBClassifier(**params, random_state=42, verbosity=0, eval_metric='logloss')
    m.fit(X_tr, y_train)
    tr_acc = accuracy_score(y_train, m.predict(X_tr))
    te_acc = accuracy_score(y_test, m.predict(X_te))
    print(f"  {name}")
    print(f"    Train: {tr_acc:.4f} | Test: {te_acc:.4f} | Gap: {tr_acc-te_acc:.4f}")

# -------------------------------------------------------
# Q8. Use cross_val_score to compare all methods fairly.
#     Why is cross-validation better than single train/test split?
# -------------------------------------------------------
print("\nQ8: Cross-Validation Comparison (5-fold)")
from sklearn.pipeline import Pipeline

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Bagging':       BaggingClassifier(n_estimators=50, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost':      AdaBoostClassifier(n_estimators=50, random_state=42),
    'GBM':           GradientBoostingClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    print(f"  {name:<20}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# Answer: CV uses ALL data for both train and test (in turns)
# Single split depends heavily on random_state → unreliable

print("\n" + "="*60)
print("  ADVANCED QUESTIONS")
print("="*60)

# -------------------------------------------------------
# Q9. STACKING from scratch — train base learners,
#     use their predictions as features for meta learner
# -------------------------------------------------------
print("\nQ9: Stacking from Scratch")

from sklearn.model_selection import KFold

def manual_stacking(X_train, y_train, X_test, base_models, meta_model, cv=5):
    """
    1. For each base model: use K-fold CV to generate out-of-fold predictions
       (these are the meta-features for training)
    2. Train each base model on FULL train data → predict on test
    3. Train meta model on meta-features
    4. Final prediction: meta model on base model test predictions
    """
    n_train = len(X_train)
    n_test  = len(X_test)
    n_models = len(base_models)

    # Meta-features for training (out-of-fold predictions)
    meta_train = np.zeros((n_train, n_models))
    # Meta-features for test (average of K fold predictions)
    meta_test  = np.zeros((n_test, n_models))

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for i, model in enumerate(base_models):
        test_preds_all_folds = np.zeros((n_test, cv))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_tr, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_tr             = y_train[train_idx]

            model.fit(X_fold_tr, y_fold_tr)

            # Out-of-fold predictions → meta training features
            meta_train[val_idx, i] = model.predict(X_fold_val)
            # Test predictions for this fold
            test_preds_all_folds[:, fold] = model.predict(X_test)

        # Average test predictions across folds
        meta_test[:, i] = test_preds_all_folds.mean(axis=1).round()

        # Retrain on full training data
        model.fit(X_train, y_train)

    # Train meta learner on meta-features
    meta_model.fit(meta_train, y_train)
    final_pred = meta_model.predict(meta_test)
    return final_pred

base_models = [
    DecisionTreeClassifier(max_depth=4, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
]
meta_model = LogisticRegression(max_iter=1000)

stack_pred = manual_stacking(X_tr, y_train.values, X_te, base_models, meta_model)
stack_acc  = accuracy_score(y_test, stack_pred)
print(f"  Manual Stacking Accuracy : {stack_acc:.4f}")

# Compare with sklearn Stacking
sk_stack = StackingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(max_depth=4, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
sk_stack.fit(X_tr, y_train)
sk_stack_acc = accuracy_score(y_test, sk_stack.predict(X_te))
print(f"  Sklearn Stacking Accuracy: {sk_stack_acc:.4f}")

# -------------------------------------------------------
# Q10. Prove Bagging reduces variance experimentally.
#      Train 30 different Decision Trees on different subsets.
#      Show variance of single tree vs ensemble.
# -------------------------------------------------------
print("\nQ10: Experimental Variance Reduction Proof")

N_EXPERIMENTS = 30
single_accs = []
ensemble_preds = []

for i in range(N_EXPERIMENTS):
    X_b, y_b = resample(X_tr, y_train.values, replace=True, random_state=i)
    t = DecisionTreeClassifier(random_state=i)
    t.fit(X_b, y_b)
    single_accs.append(accuracy_score(y_test, t.predict(X_te)))
    ensemble_preds.append(t.predict(X_te))

# Ensemble prediction = majority vote of all 30 trees
ensemble_preds = np.array(ensemble_preds)
majority_vote, _ = stats.mode(ensemble_preds, axis=0)
ensemble_acc = accuracy_score(y_test, majority_vote.flatten())

print(f"  Single Tree Mean Acc   : {np.mean(single_accs):.4f}")
print(f"  Single Tree Variance   : {np.var(single_accs):.6f}  ← HIGH")
print(f"  Ensemble (30 trees) Acc: {ensemble_acc:.4f}  ← MORE STABLE")
# This PROVES: aggregating trees reduces variance!

# -------------------------------------------------------
# Q11. XGBoost Feature Importance — 3 types comparison
#      Which feature is most important by each metric?
# -------------------------------------------------------
print("\nQ11: XGBoost Feature Importance — 3 Types")

xgb_fi = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    random_state=42, verbosity=0, eval_metric='logloss'
)
xgb_fi.fit(X_tr, y_train)
booster = xgb_fi.get_booster()

for imp_type in ['gain', 'weight', 'cover']:
    scores = booster.get_score(importance_type=imp_type)
    if scores:
        top_feat = max(scores, key=scores.get)
        print(f"  {imp_type.upper():<8}: {top_feat} ({scores[top_feat]:.2f})")

# Answer: Gain = best feature for accuracy improvement (most meaningful)
# Weight = just counts usage, can be misleading
# Cover = shows reach/influence

print("\n" + "="*60)
print("  INTELLECTUAL / TRICKY QUESTIONS")
print("="*60)

# -------------------------------------------------------
# Q12. INTELLECTUAL: If you use bootstrap=False in BaggingClassifier,
#      what happens? What is this called? Is it still "Bagging"?
# -------------------------------------------------------
print("\nQ12: Bagging vs Pasting")
pasting = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,
    bootstrap=False,   # WITHOUT replacement = PASTING
    max_samples=0.7,   # use 70% of data each time
    random_state=42
)
pasting.fit(X_tr, y_train)
pasting_acc = accuracy_score(y_test, pasting.predict(X_te))
print(f"  Pasting (no replacement) Acc : {pasting_acc:.4f}")
print(f"  Bagging (with replacement)Acc: {bag_acc:.4f}")
# Answer: bootstrap=False = "Pasting" (random subsets WITHOUT replacement)
# Bagging has more diversity (duplicates create different distributions)
# Pasting has less diversity but each sample is "purer"

# -------------------------------------------------------
# Q13. INTELLECTUAL: Why does AdaBoost use exponential loss?
#      What happens to weights when eps = 0.5 exactly?
# -------------------------------------------------------
print("\nQ13: AdaBoost — What if eps = 0.5?")
eps_values = [0.1, 0.3, 0.5, 0.7, 0.9]
print("  eps   → alpha (learner weight)")
for eps in eps_values:
    if eps == 0.5:
        print(f"  {eps} → alpha = 0 (random guess, useless!)")
    elif eps > 0.5:
        alpha = 0.5 * np.log((1-eps)/(eps+1e-10))
        print(f"  {eps} → alpha = {alpha:.4f} (NEGATIVE! predictions FLIPPED)")
    else:
        alpha = 0.5 * np.log((1-eps)/eps)
        print(f"  {eps} → alpha = {alpha:.4f}")
# Answer: eps=0.5 → alpha=0 → this learner contributes NOTHING
# eps>0.5 → alpha<0 → predictions FLIPPED (this learner is worse than random)
# AdaBoost fails if weak learner can't beat random guess!

# -------------------------------------------------------
# Q14. INTELLECTUAL: Random Forest with max_features=n_features
#      (all features). Is it still Random Forest? What's it equal to?
# -------------------------------------------------------
print("\nQ14: RF with all features = Bagging?")
rf_all_feat = RandomForestClassifier(
    n_estimators=50,
    max_features=X_tr.shape[1],  # ALL features → no random subset
    bootstrap=True,
    random_state=42
)
rf_all_feat.fit(X_tr, y_train)
rf_all_acc = accuracy_score(y_test, rf_all_feat.predict(X_te))
print(f"  RF (all features) Acc : {rf_all_acc:.4f}")
print(f"  Bagging Acc           : {bag_acc:.4f}")
# Answer: RF with max_features=all features ≈ Bagging!
# The only randomness comes from bootstrap, not feature selection.
# This proves Random Forest's KEY advantage is the RANDOM FEATURE SUBSET.

# -------------------------------------------------------
# Q15. INTELLECTUAL: XGBoost vs GBM — show regularization effect
#      Same config but XGBoost has reg, GBM doesn't.
#      On purpose OVERFIT GBM and show XGBoost resists.
# -------------------------------------------------------
print("\nQ15: XGBoost Regularization vs Plain GBM")

# Deliberately overfit-prone config (deep trees, many estimators)
gbm_overfit = GradientBoostingClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.3,
    subsample=1.0, random_state=42
)
gbm_overfit.fit(X_tr, y_train)
gbm_tr = accuracy_score(y_train, gbm_overfit.predict(X_tr))
gbm_te = accuracy_score(y_test,  gbm_overfit.predict(X_te))

xgb_reg = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.3,
    subsample=1.0, reg_lambda=5.0, gamma=1.0,  # regularization!
    random_state=42, verbosity=0, eval_metric='logloss'
)
xgb_reg.fit(X_tr, y_train)
xgb_tr = accuracy_score(y_train, xgb_reg.predict(X_tr))
xgb_te = accuracy_score(y_test,  xgb_reg.predict(X_te))

print(f"  GBM (no reg) → Train: {gbm_tr:.4f} | Test: {gbm_te:.4f} | Gap: {gbm_tr-gbm_te:.4f}")
print(f"  XGBoost(reg) → Train: {xgb_tr:.4f} | Test: {xgb_te:.4f} | Gap: {xgb_tr-xgb_te:.4f}")
# Answer: XGBoost has smaller train-test gap → less overfitting
# Regularization (gamma, lambda) penalizes complex trees

print("\n" + "="*60)
print("  SUMMARY — Key Things to Remember for Exam")
print("="*60)
print("""
  1. Bagging     → Parallel, reduces VARIANCE, bootstrap with replacement
  2. RF          → Bagging + random sqrt(p) features per split
  3. AdaBoost    → Sequential, reduces BIAS, exponential loss
                   alpha = 0.5 * ln((1-eps)/eps)
                   Fails if eps >= 0.5
  4. GBM         → Fits residuals (negative gradients) sequentially
  5. XGBoost     → GBM + L1/L2 reg + Gain-based splits + fast
  6. Stacking    → Base learners → meta learner (use CV to avoid leakage!)
  7. Voting      → Hard (majority) vs Soft (avg prob, usually better)

  OOB Score      → Free validation using ~37% unused bootstrap samples
  Early Stopping → Stop when val loss increases (prevents overfit)
  Gain > 0       → XGBoost's split condition (not Gini/Entropy!)
  bootstrap=False → "Pasting" (not Bagging)
  eps=0.5        → AdaBoost alpha=0, useless learner
  max_features=all → RF becomes Bagging
""")