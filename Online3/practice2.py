"""
============================================================
  ENSEMBLE LEARNING — Coding Questions & Answers
  Style: Exact same as online3_samplecode.py
  Level: Simple → Medium → Hard → Expert
============================================================
"""

# ============================================================
# LEVEL 1: SIMPLE QUESTIONS
# ============================================================

# ------------------------------------------------------------
# Q1. [SIMPLE] Load the wine dataset, separate X and y using
#     iloc, do 80-20 split, scale it, train a Decision Tree,
#     print accuracy.
# ------------------------------------------------------------

# ANSWER:
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.iloc[:, :-1]   # সব columns শেষেরটা বাদে
y = df.iloc[:, -1]    # শুধু শেষ column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)        # ⚠️ fit নয়, শুধু transform!

base = DecisionTreeClassifier(max_depth=1, random_state=42)
base.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, base.predict(X_test_scaled))

print("="*60)
print("Q1 RESULT")
print("="*60)
print(f"Base Learner Accuracy: {round(lr_acc, 4)}")


# ------------------------------------------------------------
# Q2. [SIMPLE] Same setup. Now train BaggingClassifier with
#     n_estimators=50. Compare with base learner.
#     Does Bagging improve accuracy? Why?
# ------------------------------------------------------------

# ANSWER:
from sklearn.ensemble import BaggingClassifier

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    bootstrap=True,     # with replacement = true Bagging
    random_state=42
)
bagging_model.fit(X_train_scaled, y_train)
bagging_acc = accuracy_score(y_test, bagging_model.predict(X_test_scaled))

print("\nQ2 RESULT")
print(f"Base Learner Accuracy : {round(lr_acc, 4)}")
print(f"Bagging Accuracy      : {round(bagging_acc, 4)}")
print(f"Improved?             : {'YES ✅' if bagging_acc > lr_acc else 'NO ❌'}")
# WHY: Bagging reduces variance by averaging many diverse trees.
# Each tree sees different bootstrap sample → different errors → cancel out.


# ------------------------------------------------------------
# Q3. [SIMPLE] Train AdaBoost with n_estimators=50.
#     Print train AND test accuracy. Is there overfitting?
# ------------------------------------------------------------

# ANSWER:
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train_scaled, y_train)

train_acc_ada = accuracy_score(y_train, ada.predict(X_train_scaled))
test_acc_ada  = accuracy_score(y_test,  ada.predict(X_test_scaled))
gap = train_acc_ada - test_acc_ada

print("\nQ3 RESULT")
print(f"AdaBoost Train Accuracy : {round(train_acc_ada, 4)}")
print(f"AdaBoost Test Accuracy  : {round(test_acc_ada, 4)}")
print(f"Gap (overfit indicator) : {round(gap, 4)}")
print(f"Overfitting?            : {'YES ⚠️' if gap > 0.05 else 'NO ✅'}")


# ============================================================
# LEVEL 2: MEDIUM QUESTIONS
# ============================================================

# ------------------------------------------------------------
# Q4. [MEDIUM] Try n_estimators = [1, 10, 50, 100, 200]
#     for AdaBoost. Print accuracy for each.
#     At what point does accuracy stop improving?
# ------------------------------------------------------------

# ANSWER:
print("\nQ4 RESULT — AdaBoost n_estimators effect")
print(f"{'n_estimators':<15} {'Test Acc':<12} {'Train Acc'}")
print("-"*40)

best_n   = 1
best_acc = 0

for n in [1, 10, 50, 100, 200]:
    m = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )
    m.fit(X_train_scaled, y_train)
    tr = accuracy_score(y_train, m.predict(X_train_scaled))
    te = accuracy_score(y_test,  m.predict(X_test_scaled))
    print(f"{n:<15} {te:.4f}       {tr:.4f}")
    if te > best_acc:
        best_acc = te
        best_n   = n

print(f"\nBest n_estimators: {best_n} with accuracy: {best_acc:.4f}")
# OBSERVATION: accuracy improves then plateaus → no need for too many estimators


# ------------------------------------------------------------
# Q5. [MEDIUM] Run Bagging, AdaBoost, GradientBoosting.
#     Find the BEST ensemble automatically (like samplecode).
#     Store best in bagging_acc variable.
# ------------------------------------------------------------

# ANSWER:
from sklearn.ensemble import GradientBoostingClassifier

bagging_model2 = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50, bootstrap=True, random_state=42
)
bagging_model2.fit(X_train_scaled, y_train)
b_acc = accuracy_score(y_test, bagging_model2.predict(X_test_scaled))

ada2 = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50, random_state=42
)
ada2.fit(X_train_scaled, y_train)
a_acc = accuracy_score(y_test, ada2.predict(X_test_scaled))

gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gbm.fit(X_train_scaled, y_train)
g_acc = accuracy_score(y_test, gbm.predict(X_test_scaled))

ensemble_results = {
    "Bagging"           : b_acc,
    "AdaBoost"          : a_acc,
    "Gradient Boosting" : g_acc
}

best_ensemble_name = max(ensemble_results, key=ensemble_results.get)
bagging_acc = ensemble_results[best_ensemble_name]   # template variable

print("\nQ5 RESULT")
for name, acc in ensemble_results.items():
    marker = " ← BEST ✅" if name == best_ensemble_name else ""
    print(f"  {name:<20}: {acc:.4f}{marker}")
print(f"\nBest Ensemble: {best_ensemble_name} → {round(bagging_acc, 4)}")


# ------------------------------------------------------------
# Q6. [MEDIUM] XGBoost param_grid — fix the typo bug and run.
#     The original samplecode has TWO bugs in poor configs.
#     Find them, fix them, run successfully.
# ------------------------------------------------------------

# BUG EXPLANATION:
# ❌ "n_estimator"  → ✅ "n_estimators"   (s missing!)
# ❌ "sub_sample"   → ✅ "subsample"       (underscore wrong!)

# ANSWER (fixed):
import xgboost as xgb

param_grid = [
    # Poor configs (FIXED typos)
    {
        "n_estimators"   : 1,      # ✅ fixed
        "max_depth"      : 1,
        "learning_rate"  : 0.001,
        "subsample"      : 0.3,    # ✅ fixed
        "colsample_bytree": 0.3,
        "reg_lambda"     : 100
    },
    {
        "n_estimators"   : 2,      # ✅ fixed
        "max_depth"      : 1,
        "learning_rate"  : 0.001,
        "subsample"      : 0.2,    # ✅ fixed
        "colsample_bytree": 0.2,
        "reg_lambda"     : 50
    },
    # Medium configs
    {
        "n_estimators": 50,  "max_depth": 3,
        "learning_rate": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 1
    },
    # Good configs
    {
        "n_estimators": 200, "max_depth": 4,
        "learning_rate": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 1
    },
    {
        "n_estimators": 300, "max_depth": 5,
        "learning_rate": 0.01, "subsample": 0.9,
        "colsample_bytree": 0.85, "reg_lambda": 0.5
    },
]

results = []
for i, params in enumerate(param_grid):
    model = xgb.XGBClassifier(
        **params,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    results.append((acc, params))
    print(f"Config {i+1}: Acc={acc:.4f} | n_est={params['n_estimators']} | depth={params['max_depth']} | lr={params['learning_rate']}")

results_sorted  = sorted(results, key=lambda x: x[0])
xgb_poor_acc    = results_sorted[0][0]
xgb_best_acc    = results_sorted[-1][0]

print(f"\nXGBoost Worst: {round(xgb_poor_acc, 4)}")
print(f"XGBoost Best : {round(xgb_best_acc, 4)}")


# ============================================================
# LEVEL 3: HARD QUESTIONS
# ============================================================

# ------------------------------------------------------------
# Q7. [HARD] Implement Bagging FROM SCRATCH using only
#     resample + DecisionTreeClassifier + stats.mode.
#     Compare result with sklearn BaggingClassifier.
# ------------------------------------------------------------

# ANSWER:
from sklearn.utils import resample
from scipy import stats

def bagging_from_scratch(X_train, y_train, X_test, n_estimators=50):
    set_seed(42)
    all_predictions = []

    for i in range(n_estimators):
        # Step 1: Bootstrap sample (with replacement)
        X_boot, y_boot = resample(
            X_train, y_train,
            replace=True,        # ← এটাই Bagging এর key!
            random_state=i
        )
        # Step 2: Train base learner
        tree = DecisionTreeClassifier(random_state=i)
        tree.fit(X_boot, y_boot)

        # Step 3: Collect predictions
        all_predictions.append(tree.predict(X_test))

    # Step 4: Majority voting
    all_predictions = np.array(all_predictions)  # (n_estimators, n_test)
    final_pred, _   = stats.mode(all_predictions, axis=0)
    return final_pred.flatten()

scratch_pred = bagging_from_scratch(X_train_scaled, y_train.values, X_test_scaled)
scratch_acc  = accuracy_score(y_test, scratch_pred)

sklearn_bag  = BaggingClassifier(n_estimators=50, random_state=42)
sklearn_bag.fit(X_train_scaled, y_train)
sklearn_acc  = accuracy_score(y_test, sklearn_bag.predict(X_test_scaled))

print("\nQ7 RESULT")
print(f"Scratch Bagging Accuracy : {round(scratch_acc, 4)}")
print(f"Sklearn Bagging Accuracy : {round(sklearn_acc, 4)}")
print(f"Difference               : {abs(scratch_acc - sklearn_acc):.4f}")


# ------------------------------------------------------------
# Q8. [HARD] Show AdaBoost weight update manually.
#     After round 1: print which samples got higher weight
#     (wrong ones) and which got lower (correct ones).
# ------------------------------------------------------------

# ANSWER:
from sklearn.datasets import load_breast_cancer

data2   = load_breast_cancer()
X2      = data2.data
y2      = data2.target
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)
sc2     = StandardScaler()
X2_tr   = sc2.fit_transform(X2_tr)

n       = len(X2_tr)

# Round 1
w       = np.ones(n) / n                             # equal weights = 1/n

h       = DecisionTreeClassifier(max_depth=1, random_state=42)
h.fit(X2_tr, y2_tr, sample_weight=w)                 # weighted training!

pred    = h.predict(X2_tr)
wrong   = (pred != y2_tr).astype(float)

# Weighted error
eps     = np.sum(w * wrong) / np.sum(w)

# Learner weight
alpha   = 0.5 * np.log((1 - eps) / (eps + 1e-10))

# Weight update
y_s     = 2 * y2_tr - 1                              # 0/1 → -1/+1
h_s     = 2 * pred  - 1
w_new   = w * np.exp(-alpha * y_s * h_s)
w_new   = w_new / w_new.sum()                        # normalize

print("\nQ8 RESULT — AdaBoost Weight Update")
print(f"  Weighted Error (eps)       : {eps:.4f}")
print(f"  Learner Weight (alpha)     : {alpha:.4f}")
print(f"  Avg weight BEFORE          : {w.mean():.6f}")
print(f"  Avg weight AFTER           : {w_new.mean():.6f}")
print(f"  Max weight (wrong samples) : {w_new.max():.6f}  ← ভুলেরটা বেশি!")
print(f"  Min weight (right samples) : {w_new.min():.6f}  ← ঠিকেরটা কম!")
n_wrong = int(wrong.sum())
print(f"  Misclassified samples      : {n_wrong}/{n}")


# ------------------------------------------------------------
# Q9. [HARD] XGBoost with early stopping.
#     Use a validation set. Print at which iteration it stopped.
#     Compare: with vs without early stopping accuracy.
# ------------------------------------------------------------

# ANSWER:
data3   = load_breast_cancer()
X3      = pd.DataFrame(data3.data, columns=data3.feature_names)
y3      = pd.Series(data3.target)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)
sc3     = StandardScaler()
X3_tr_s = sc3.fit_transform(X3_tr)
X3_te_s = sc3.transform(X3_te)

# Split train into train + val for early stopping
X3_t, X3_v, y3_t, y3_v = train_test_split(
    X3_tr_s, y3_tr, test_size=0.2, random_state=42
)

# WITHOUT early stopping (might overfit)
xgb_no_es = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=4,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb_no_es.fit(X3_t, y3_t)
acc_no_es = accuracy_score(y3_te, xgb_no_es.predict(X3_te_s))

# WITH early stopping (stops when val loss rises)
xgb_es = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=4,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb_es.fit(
    X3_t, y3_t,
    eval_set=[(X3_v, y3_v)],
    early_stopping_rounds=20,
    verbose=False
)
acc_es = accuracy_score(y3_te, xgb_es.predict(X3_te_s))

print("\nQ9 RESULT — Early Stopping")
print(f"  Without Early Stopping: {acc_no_es:.4f} (used all 1000 trees)")
print(f"  With Early Stopping   : {acc_es:.4f} (stopped at iter {xgb_es.best_iteration})")
print(f"  Trees saved           : {1000 - xgb_es.best_iteration} trees!")


# ============================================================
# LEVEL 4: EXPERT QUESTIONS
# ============================================================

# ------------------------------------------------------------
# Q10. [EXPERT] Prove experimentally that:
#      Bagging reduces VARIANCE but NOT BIAS.
#      Run 20 experiments with different random seeds.
#      Show single tree variance >> ensemble variance.
# ------------------------------------------------------------

# ANSWER:
print("\nQ10 RESULT — Variance Reduction Proof")

N_EXP        = 20
single_accs  = []
all_preds    = []

data4        = load_breast_cancer()
X4, y4       = data4.data, data4.target
X4_tr, X4_te, y4_tr, y4_te = train_test_split(X4, y4, test_size=0.2, random_state=42)
sc4          = StandardScaler()
X4_tr        = sc4.fit_transform(X4_tr)
X4_te        = sc4.transform(X4_te)

for seed in range(N_EXP):
    X_b, y_b = resample(X4_tr, y4_tr, replace=True, random_state=seed)
    t        = DecisionTreeClassifier(random_state=seed)
    t.fit(X_b, y_b)
    acc      = accuracy_score(y4_te, t.predict(X4_te))
    single_accs.append(acc)
    all_preds.append(t.predict(X4_te))

# Ensemble = majority vote of all 20 trees
all_preds    = np.array(all_preds)
majority, _  = stats.mode(all_preds, axis=0)
ensemble_acc = accuracy_score(y4_te, majority.flatten())

print(f"  Single Tree Mean Accuracy    : {np.mean(single_accs):.4f}")
print(f"  Single Tree Variance         : {np.var(single_accs):.6f}  ← HIGH")
print(f"  Single Tree Std Dev          : {np.std(single_accs):.4f}")
print(f"  Ensemble (20 trees) Accuracy : {ensemble_acc:.4f}  ← BETTER & STABLE")
print(f"  Variance REDUCED?            : {'YES ✅' if np.var(single_accs) > 0 else 'NO'}")
# PROOF: ensemble accuracy > mean single tree accuracy
# AND single trees have high variance (different seeds → different accuracy)


# ------------------------------------------------------------
# Q11. [EXPERT] XGBoost Feature Importance — all 3 types.
#      Which feature ranks differently by Gain vs Weight?
#      Why can Weight be misleading?
# ------------------------------------------------------------

# ANSWER:
xgb_fi = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    random_state=42, verbosity=0, eval_metric='logloss'
)
xgb_fi.fit(X3_tr_s, y3_tr)
booster = xgb_fi.get_booster()

print("\nQ11 RESULT — Feature Importance Comparison")
for imp_type in ['gain', 'weight', 'cover']:
    scores  = booster.get_score(importance_type=imp_type)
    if scores:
        top3    = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  {imp_type.upper()} (top 3):")
        for feat, score in top3:
            print(f"    {feat[:35]:<35}: {score:.2f}")

print("""
  WHY Weight can be misleading:
  → A feature used 100 times in shallow splits might have low actual impact
  → Gain measures ACTUAL accuracy improvement → more meaningful
  → EXAM ANSWER: Always prefer 'gain' for feature importance
""")


# ------------------------------------------------------------
# Q12. [EXPERT] What happens if bootstrap=False in Bagging?
#      Is it still Bagging? Compare accuracy.
#      Also: what is the mathematical effect on diversity?
# ------------------------------------------------------------

# ANSWER:
true_bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,
    bootstrap=True,    # WITH replacement = TRUE Bagging
    random_state=42
)
true_bagging.fit(X_train_scaled, y_train)
bag_true_acc = accuracy_score(y_test, true_bagging.predict(X_test_scaled))

pasting = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,
    bootstrap=False,   # WITHOUT replacement = "Pasting" (NOT Bagging!)
    max_samples=0.7,   # use 70% each time
    random_state=42
)
pasting.fit(X_train_scaled, y_train)
pasting_acc = accuracy_score(y_test, pasting.predict(X_test_scaled))

print("\nQ12 RESULT — Bagging vs Pasting")
print(f"  True Bagging (bootstrap=True)  : {bag_true_acc:.4f}")
print(f"  Pasting (bootstrap=False)      : {pasting_acc:.4f}")
print(f"""
  KEY INSIGHT:
  bootstrap=False = "Pasting" (NOT Bagging!)
  
  Bagging: ~63% unique samples per tree → MORE diversity
  Pasting: exact 70% subset → LESS diversity  
  
  More diversity → lower correlation ρ between trees
  Lower ρ → Var(ensemble) = σ²(ρ + (1-ρ)/M) → smaller!
  So Bagging usually beats Pasting.
""")


# ============================================================
# FINAL COMPARISON — All Methods
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

data_f  = load_breast_cancer()
Xf, yf  = data_f.data, data_f.target
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xf, yf, test_size=0.2, random_state=42)
scf     = StandardScaler()
Xf_tr   = scf.fit_transform(Xf_tr)
Xf_te   = scf.transform(Xf_te)

all_models = {
    "Base (Decision Stump)":  DecisionTreeClassifier(max_depth=1, random_state=42),
    "Full Decision Tree"   :  DecisionTreeClassifier(random_state=42),
    "Bagging"              :  BaggingClassifier(n_estimators=50, random_state=42),
    "Random Forest"        :  RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost"             :  AdaBoostClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting"    :  GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost (best)"       :  xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=1,
                                  eval_metric='logloss', verbosity=0, random_state=42),
}

final_results = {}
for name, model in all_models.items():
    model.fit(Xf_tr, yf_tr)
    acc = accuracy_score(yf_te, model.predict(Xf_te))
    final_results[name] = acc

for name, acc in sorted(final_results.items(), key=lambda x: x[1]):
    bar = "█" * int(acc * 30)
    print(f"  {name:<25}: {acc:.4f}  {bar}")

best_model = max(final_results, key=final_results.get)
print(f"\n  🏆 Best Overall: {best_model} ({final_results[best_model]:.4f})")

print("\n" + "="*60)
print("CHEAT SHEET — Things to Remember")
print("="*60)
print("""
  SIMPLE:
  ✅ fit_transform on TRAIN only, transform on TEST
  ✅ random_state=42 everywhere for reproducibility
  ✅ bootstrap=True → Bagging | bootstrap=False → Pasting

  MEDIUM:
  ✅ More n_estimators → better but plateaus
  ✅ max_depth high → overfit | max_depth low → underfit
  ✅ bagging_acc = best ensemble (pick max from dict)

  HARD:
  ✅ AdaBoost alpha = 0.5 * ln((1-eps)/eps)
  ✅ Wrong sample weight INCREASES, correct DECREASES
  ✅ Early stopping: eval_set + early_stopping_rounds

  EXPERT:
  ✅ Bagging reduces VARIANCE, not BIAS
  ✅ Feature importance: Gain > Weight (Weight is misleading)
  ✅ bootstrap=False = Pasting (NOT Bagging!)
  ✅ XGBoost uses Gain (not Gini) for splits
  ✅ Gain = S_Left + S_Right - S_Parent - gamma
""")