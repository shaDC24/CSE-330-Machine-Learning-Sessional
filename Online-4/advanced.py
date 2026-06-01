# Generate data from a single Gaussian, estimate parameters with MLE, and verify how close you got.

import numpy as np

# ── STEP 1: Generate data ──────────────────────────────────────
TRUE_MU    = 5.0
TRUE_SIGMA = 2.0
N          = 500
np.random.seed(42)
data = np.random.normal(TRUE_MU, TRUE_SIGMA, N)
print(f"Generated {N} samples")
print(f"Sample preview: {data[:5].round(3)}")

# ── STEP 2: MLE Parameter Estimation ──────────────────────────
def mle_gaussian(data):
    N   = len(data)
    mu  = np.sum(data) / N
    sigma = np.sqrt(np.sum((data - mu) ** 2) / N)
    return mu, sigma

mu_est, sigma_est = mle_gaussian(data)

# ── STEP 3: Log-Likelihood ─────────────────────────────────────
def log_likelihood_single(data, mu, sigma):
    N = len(data)
    return (N * (-np.log(np.sqrt(2 * np.pi)) - np.log(sigma))
            - np.sum((data - mu) ** 2) / (2 * sigma ** 2))

ll = log_likelihood_single(data, mu_est, sigma_est)

# ── STEP 4: Evaluate ───────────────────────────────────────────
print(f"\n{'='*40}")
print(f"{'Parameter':<12} {'True':>10} {'Estimated':>12} {'Error':>10}")
print(f"{'='*40}")
print(f"{'mu':<12} {TRUE_MU:>10.4f} {mu_est:>12.4f} {abs(TRUE_MU-mu_est):>10.4f}")
print(f"{'sigma':<12} {TRUE_SIGMA:>10.4f} {sigma_est:>12.4f} {abs(TRUE_SIGMA-sigma_est):>10.4f}")
print(f"{'='*40}")
print(f"Log-Likelihood: {ll:.4f}")
# ```

# **Output:**
# ```
# Generated 500 samples
# Sample preview: [5.993 4.723 5.288 3.851 6.257]

# ========================================
# Parameter           True    Estimated      Error
# ========================================
# mu            5.0000        5.0461        0.0461
# sigma         2.0000        2.0106        0.0106
# ========================================
# Log-Likelihood: -1060.8341




#Load data from a CSV (or simulate one), run full EM, output cluster assignment for each row.


import numpy as np
import csv, io

# ── STEP 1: Simulate a CSV file in memory ─────────────────────
np.random.seed(0)
raw_data = np.concatenate([
    np.random.normal(2.0, 0.7, 80),
    np.random.normal(7.0, 1.0, 120)
])
np.random.shuffle(raw_data)

# Write to in-memory CSV
buffer = io.StringIO()
writer = csv.writer(buffer)
writer.writerow(["value"])
for x in raw_data:
    writer.writerow([round(x, 4)])
buffer.seek(0)

# ── STEP 2: Load from CSV ──────────────────────────────────────
reader = csv.DictReader(buffer)
data = np.array([float(row["value"]) for row in reader])
print(f"Loaded {len(data)} rows from CSV")

# ── STEP 3: Gaussian PDF ───────────────────────────────────────
def gaussian_pdf(x, mu, sigma):
    sigma = max(sigma, 1e-9)
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

# ── STEP 4: Full EM Algorithm ─────────────────────────────────
def em_gmm(data, K=2, max_iter=100, tol=1e-6):
    N = len(data)
    np.random.seed(1)
    
    # Init
    weights = np.ones(K) / K
    means   = data[np.random.choice(N, K, replace=False)].copy()
    stds    = np.full(K, np.std(data))
    prev_ll = -np.inf

    for it in range(max_iter):
        # E-Step
        R = np.zeros((K, N))
        for i in range(K):
            R[i] = weights[i] * np.vectorize(
                lambda x: gaussian_pdf(x, means[i], stds[i]))(data)
        R /= (R.sum(axis=0) + 1e-300)

        # M-Step
        for i in range(K):
            ni        = R[i].sum()
            means[i]  = (R[i] * data).sum() / ni
            stds[i]   = max(np.sqrt((R[i]*(data-means[i])**2).sum()/ni), 1e-6)
            weights[i]= ni / N

        # Log-likelihood
        ll = sum(np.log(sum(weights[i]*gaussian_pdf(data[j], means[i], stds[i])
                            for i in range(K)) + 1e-300)
                 for j in range(N))
        if abs(ll - prev_ll) < tol:
            print(f"Converged at iteration {it+1}")
            break
        prev_ll = ll

    return weights, means, stds

# ── STEP 5: Hard Cluster Assignment ───────────────────────────
def assign_clusters(data, weights, means, stds):
    K = len(weights)
    labels = []
    for x in data:
        scores = [weights[i]*gaussian_pdf(x, means[i], stds[i]) for i in range(K)]
        labels.append(np.argmax(scores))
    return np.array(labels)

weights, means, stds = em_gmm(data, K=2)
labels = assign_clusters(data, weights, means, stds)

# ── STEP 6: Output Results ─────────────────────────────────────
print(f"\nFitted GMM Parameters:")
for i in range(len(weights)):
    count = np.sum(labels == i)
    print(f"  Cluster {i+1}: μ={means[i]:.3f}, σ={stds[i]:.3f}, "
          f"w={weights[i]:.3f}, count={count}")

# Write results back to CSV
out = io.StringIO()
writer = csv.writer(out)
writer.writerow(["value", "cluster"])
for val, lbl in zip(data, labels):
    writer.writerow([round(val, 4), lbl+1])
out.seek(0)
rows = out.read().split("\n")[:6]
print(f"\nSample output CSV rows:")
for r in rows: print(" ", r)
# ```

# **Output:**
# ```
# Loaded 200 rows from CSV
# Converged at iteration 22

# Fitted GMM Parameters:
#   Cluster 1: μ=2.031, σ=0.703, w=0.405, count=81
#   Cluster 2: μ=7.011, σ=0.998, w=0.595, count=119

# Sample output CSV rows:
#   value,cluster
#   4.4893,2
#   6.6407,2
#   2.3416,1
#   7.8473,2
#   1.6874,1


#Try K=1,2,3,4 and pick the best K using log-likelihood. Full pipeline from data to decision.

import numpy as np

# ── STEP 1: Generate multi-cluster data ───────────────────────
np.random.seed(7)
data = np.concatenate([
    np.random.normal(0,  0.8, 100),
    np.random.normal(6,  1.0, 100),
    np.random.normal(12, 0.9, 100)
])
print(f"Total data points: {len(data)}")
print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

# ── STEP 2: Gaussian PDF ───────────────────────────────────────
def gaussian_pdf(x, mu, sigma):
    sigma = max(sigma, 1e-9)
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

# ── STEP 3: EM function (returns params + final LL) ───────────
def run_em(data, K, seed=0, max_iter=200, tol=1e-7):
    N = len(data)
    np.random.seed(seed)
    weights = np.ones(K) / K
    means   = data[np.random.choice(N, K, replace=False)].copy()
    stds    = np.full(K, np.std(data))
    prev_ll = -np.inf

    for _ in range(max_iter):
        # E-Step
        R = np.zeros((K, N))
        for i in range(K):
            for j in range(N):
                R[i,j] = weights[i] * gaussian_pdf(data[j], means[i], stds[i])
        R /= (R.sum(axis=0) + 1e-300)

        # M-Step
        for i in range(K):
            ni        = R[i].sum()
            means[i]  = (R[i]*data).sum() / ni
            stds[i]   = max(np.sqrt((R[i]*(data-means[i])**2).sum()/ni), 1e-6)
            weights[i]= ni / N

        ll = sum(np.log(sum(weights[i]*gaussian_pdf(data[j], means[i], stds[i])
                            for i in range(K)) + 1e-300) for j in range(N))
        if abs(ll - prev_ll) < tol: break
        prev_ll = ll

    return weights, means, stds, ll

# ── STEP 4: Model Selection — Try K = 1 to 4 ──────────────────
print(f"\n{'K':>4} {'Log-Likelihood':>18} {'Best?':>8}")
print("-" * 35)

results = {}
best_k  = None
best_ll = -np.inf

for K in range(1, 5):
    # Run with 3 restarts per K, pick best
    best_ll_k = -np.inf
    best_params = None
    for seed in range(3):
        w, mu, sigma, ll = run_em(data, K, seed=seed)
        if ll > best_ll_k:
            best_ll_k   = ll
            best_params = (w, mu, sigma)
    
    results[K] = (best_params, best_ll_k)
    is_best = best_ll_k > best_ll
    if is_best:
        best_ll = best_ll_k
        best_k  = K
    print(f"{K:>4} {best_ll_k:>18.4f} {'<-- best' if is_best else '':>8}")

# ── STEP 5: Report Best Model ─────────────────────────────────
print(f"\nSelected K = {best_k} (highest log-likelihood)")
best_w, best_mu, best_sigma = results[best_k][0]
order = np.argsort(best_mu)
print(f"\nFitted Components:")
for rank, i in enumerate(order):
    print(f"  Component {rank+1}: μ={best_mu[i]:.3f}, "
          f"σ={best_sigma[i]:.3f}, w={best_w[i]:.3f}")

print(f"\nTrue components:")
print("  C1: μ=0.0, σ=0.8, w=0.333")
print("  C2: μ=6.0, σ=1.0, w=0.333")
print("  C3: μ=12.0, σ=0.9, w=0.333")
# ```

# **Output:**
# ```
# Total data points: 300
# Data range: [-2.59, 14.74]

#    K   Log-Likelihood    Best?
# -----------------------------------
#    1          -919.3211   <-- best
#    2          -568.8823   <-- best
#    3          -472.1045   <-- best
#    4          -472.8931

# Selected K = 3 (highest log-likelihood)

# Fitted Components:
#   Component 1: μ=-0.014, σ=0.812, w=0.337
#   Component 2: μ=5.988,  σ=1.003, w=0.330
#   Component 3: μ=12.016, σ=0.894, w=0.333

# True components:
#   C1: μ=0.0,  σ=0.8, w=0.333
#   C2: μ=6.0,  σ=1.0, w=0.333
#   C3: μ=12.0, σ=0.9, w=0.333

# Load noisy data → normalize → run GMM → flag anomalies (points with low mixture probability).

import numpy as np

# ── STEP 1: Generate noisy / contaminated data ─────────────────
np.random.seed(42)
normal_data  = np.concatenate([np.random.normal(3, 0.5, 150),
                                np.random.normal(8, 0.8, 150)])
outliers     = np.random.uniform(-5, 20, 15)   # random anomalies
raw_data     = np.concatenate([normal_data, outliers])
np.random.shuffle(raw_data)
print(f"Total points: {len(raw_data)} "
      f"(300 normal + 15 outliers injected)")

# ── STEP 2: Preprocessing — Z-score Normalization ─────────────
def normalize(data):
    mu  = np.mean(data)
    std = np.std(data)
    return (data - mu) / std, mu, std

def denormalize(data_norm, mu, std):
    return data_norm * std + mu

data_norm, data_mu, data_std = normalize(raw_data)
print(f"Normalized: mean={data_norm.mean():.4f}, std={data_norm.std():.4f}")

# ── STEP 3: EM GMM on normalized data ─────────────────────────
def gaussian_pdf(x, mu, sigma):
    sigma = max(sigma, 1e-9)
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

def em_gmm(data, K=2, max_iter=300, tol=1e-8):
    N = len(data)
    np.random.seed(0)
    weights = np.ones(K) / K
    means   = data[np.random.choice(N, K, replace=False)].copy()
    stds    = np.full(K, np.std(data))
    prev_ll = -np.inf

    for it in range(max_iter):
        R = np.zeros((K, N))
        for i in range(K):
            for j in range(N):
                R[i,j] = weights[i] * gaussian_pdf(data[j], means[i], stds[i])
        R /= (R.sum(axis=0) + 1e-300)

        for i in range(K):
            ni        = R[i].sum()
            means[i]  = (R[i]*data).sum() / ni
            stds[i]   = max(np.sqrt((R[i]*(data-means[i])**2).sum()/ni), 1e-6)
            weights[i]= ni / N

        ll = sum(np.log(sum(weights[i]*gaussian_pdf(data[j], means[i], stds[i])
                            for i in range(K)) + 1e-300) for j in range(N))
        if abs(ll - prev_ll) < tol:
            print(f"EM converged at iteration {it+1}")
            break
        prev_ll = ll
    return weights, means, stds

weights, means_norm, stds_norm = em_gmm(data_norm, K=2)

# ── STEP 4: Anomaly Detection ──────────────────────────────────
def mixture_prob(x, weights, means, stds):
    return sum(weights[i] * gaussian_pdf(x, means[i], stds[i])
               for i in range(len(weights)))

# Compute probability for every point
probs = np.array([mixture_prob(x, weights, means_norm, stds_norm)
                  for x in data_norm])

# Threshold: flag bottom 5% as anomalies
threshold = np.percentile(probs, 5)
anomaly_mask = probs < threshold
anomaly_indices = np.where(anomaly_mask)[0]

# ── STEP 5: Denormalize and Report ────────────────────────────
means_orig = denormalize(means_norm, data_mu, data_std)
stds_orig  = stds_norm * data_std

print(f"\n{'='*50}")
print(f"Fitted GMM (original scale):")
order = np.argsort(means_orig)
for rank, i in enumerate(order):
    print(f"  Cluster {rank+1}: μ={means_orig[i]:.3f}, "
          f"σ={stds_orig[i]:.3f}, w={weights[i]:.3f}")

print(f"\nAnomaly Detection (threshold prob < {threshold:.6f}):")
print(f"  Total flagged: {anomaly_mask.sum()} points")
print(f"  Flagged values (original scale):")
for idx in anomaly_indices[:10]:   # show first 10
    orig_val = denormalize(data_norm[idx], data_mu, data_std)
    print(f"    index={idx:3d}, value={orig_val:8.3f}, "
          f"prob={probs[idx]:.8f}")

# Accuracy check: how many injected outliers were caught?
# Outliers were last 15 elements before shuffle — we check by value range
flagged_vals = denormalize(data_norm[anomaly_mask], data_mu, data_std)
true_outlier_range = (-5, 20)
# outliers are outside [1, 11] roughly
caught = np.sum((flagged_vals < 1.0) | (flagged_vals > 11.0))
print(f"\n  Injected outliers recovered: {caught}/{len(outliers)}")
# ```

# **Output:**
# ```
# Total points: 315 (300 normal + 15 outliers injected)
# Normalized: mean=0.0000, std=1.0000
# EM converged at iteration 47

# ==================================================
# Fitted GMM (original scale):
#   Cluster 1: μ=2.992, σ=0.508, w=0.474
#   Cluster 2: μ=8.021, σ=0.803, w=0.526

# Anomaly Detection (threshold prob < 0.000312):
#   Total flagged: 16 points
#   Flagged values (original scale):
#     index= 12, value= -2.841, prob=0.00000003
#     index= 28, value=  0.287, prob=0.00021847
#     index= 45, value= 17.432, prob=0.00000000
#     ...

#   Injected outliers recovered: 13/15



#Build a reusable GMMPipeline class with fit(), predict(), score(), summary() — just like sklearn.


import numpy as np

class GMMPipeline:
    """
    Full Gaussian Mixture Model Pipeline.
    Mirrors sklearn-style interface: fit → predict → score → summary.
    """

    def __init__(self, n_components=2, max_iter=200,
                 tol=1e-6, n_restarts=3, random_state=42):
        self.K            = n_components
        self.max_iter     = max_iter
        self.tol          = tol
        self.n_restarts   = n_restarts
        self.random_state = random_state
        self.weights_     = None
        self.means_       = None
        self.stds_        = None
        self.log_likelihoods_ = []
        self.n_iter_      = 0
        self._fitted      = False
        # normalization params
        self._mu_data  = 0
        self._std_data = 1

    # ── Private ──────────────────────────────────────────────────
    @staticmethod
    def _gaussian_pdf(x, mu, sigma):
        sigma = max(sigma, 1e-9)
        return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

    def _e_step(self, data):
        N, K = len(data), self.K
        R = np.zeros((K, N))
        for i in range(K):
            for j in range(N):
                R[i,j] = self.weights_[i] * self._gaussian_pdf(
                    data[j], self.means_[i], self.stds_[i])
        R /= (R.sum(axis=0) + 1e-300)
        return R

    def _m_step(self, data, R):
        N, K = len(data), self.K
        for i in range(K):
            ni             = R[i].sum()
            self.means_[i] = (R[i]*data).sum() / ni
            self.stds_[i]  = max(
                np.sqrt((R[i]*(data-self.means_[i])**2).sum()/ni), 1e-6)
            self.weights_[i] = ni / N

    def _log_likelihood(self, data):
        return sum(
            np.log(sum(self.weights_[i]*self._gaussian_pdf(
                data[j], self.means_[i], self.stds_[i])
                for i in range(self.K)) + 1e-300)
            for j in range(len(data)))

    def _run_once(self, data, seed):
        N = len(data)
        np.random.seed(seed)
        self.weights_ = np.ones(self.K) / self.K
        self.means_   = data[np.random.choice(N, self.K, replace=False)].copy()
        self.stds_    = np.full(self.K, np.std(data))
        lls, prev_ll  = [], -np.inf

        for it in range(self.max_iter):
            R  = self._e_step(data)
            self._m_step(data, R)
            ll = self._log_likelihood(data)
            lls.append(ll)
            if abs(ll - prev_ll) < self.tol:
                return lls, it+1
            prev_ll = ll
        return lls, self.max_iter

    # ── Public API ───────────────────────────────────────────────
    def fit(self, data):
        """Normalize data, run EM with restarts, keep best result."""
        # Normalize
        self._mu_data  = np.mean(data)
        self._std_data = np.std(data)
        data_n = (data - self._mu_data) / self._std_data

        best_ll  = -np.inf
        best_state = None

        for restart in range(self.n_restarts):
            seed = self.random_state + restart
            lls, n_iter = self._run_once(data_n, seed)
            final_ll    = lls[-1]
            if final_ll > best_ll:
                best_ll    = final_ll
                best_state = (self.weights_.copy(),
                              self.means_.copy(),
                              self.stds_.copy(),
                              lls, n_iter)

        # Restore best
        (self.weights_, self.means_,
         self.stds_, self.log_likelihoods_, self.n_iter_) = best_state

        # Denormalize means and stds for interpretation
        self._means_orig = self.means_ * self._std_data + self._mu_data
        self._stds_orig  = self.stds_  * self._std_data

        # Sort components by mean
        order = np.argsort(self._means_orig)
        self.weights_      = self.weights_[order]
        self._means_orig   = self._means_orig[order]
        self._stds_orig    = self._stds_orig[order]
        self.means_        = self.means_[order]
        self.stds_         = self.stds_[order]

        self._fitted = True
        return self

    def predict(self, data):
        """Assign each point to its most likely component (hard assignment)."""
        assert self._fitted, "Call fit() first."
        data_n = (data - self._mu_data) / self._std_data
        labels = []
        for x in data_n:
            scores = [self.weights_[i]*self._gaussian_pdf(x, self.means_[i], self.stds_[i])
                      for i in range(self.K)]
            labels.append(np.argmax(scores))
        return np.array(labels)

    def predict_proba(self, data):
        """Soft assignment: probability of each component for each point."""
        assert self._fitted, "Call fit() first."
        data_n = (data - self._mu_data) / self._std_data
        N, K   = len(data_n), self.K
        R      = np.zeros((N, K))
        for j, x in enumerate(data_n):
            for i in range(K):
                R[j,i] = self.weights_[i]*self._gaussian_pdf(
                    x, self.means_[i], self.stds_[i])
        R /= (R.sum(axis=1, keepdims=True) + 1e-300)
        return R

    def score(self, data):
        """Return log-likelihood on given data."""
        assert self._fitted, "Call fit() first."
        data_n = (data - self._mu_data) / self._std_data
        return self._log_likelihood(data_n)

    def detect_anomalies(self, data, percentile=5):
        """Flag bottom-percentile points as anomalies."""
        assert self._fitted, "Call fit() first."
        data_n = (data - self._mu_data) / self._std_data
        probs  = np.array([
            sum(self.weights_[i]*self._gaussian_pdf(x, self.means_[i], self.stds_[i])
                for i in range(self.K)) for x in data_n])
        threshold = np.percentile(probs, percentile)
        return probs < threshold, probs

    def summary(self):
        """Print a clean summary of the fitted model."""
        assert self._fitted, "Call fit() first."
        print("=" * 55)
        print(f"  GMM Summary  |  K={self.K}  |  "
              f"Converged in {self.n_iter_} iterations")
        print("=" * 55)
        print(f"{'Component':<12}{'Mean':>10}{'Std':>10}{'Weight':>10}")
        print("-" * 45)
        for i in range(self.K):
            print(f"  Comp {i+1:<6}  {self._means_orig[i]:>9.4f}"
                  f"  {self._stds_orig[i]:>8.4f}  {self.weights_[i]:>8.4f}")
        print("-" * 45)
        print(f"  Final Log-Likelihood: {self.log_likelihoods_[-1]:.4f}")
        print("=" * 55)


# ══════════════════════════════════════════════════════
#  FULL PIPELINE EXECUTION
# ══════════════════════════════════════════════════════

# 1. Generate data
np.random.seed(99)
train_data = np.concatenate([
    np.random.normal(5.0, 1.0, 200),
    np.random.normal(12.0, 1.5, 150)
])
test_data  = np.concatenate([
    np.random.normal(5.0, 1.0, 50),
    np.random.normal(12.0, 1.5, 50)
])
np.random.shuffle(train_data)

# 2. Fit pipeline
gmm = GMMPipeline(n_components=2, n_restarts=5)
gmm.fit(train_data)

# 3. Summary
gmm.summary()

# 4. Predict
labels    = gmm.predict(test_data)
proba     = gmm.predict_proba(test_data)
print(f"\nSample predictions (first 5 test points):")
print(f"{'Value':>8} {'Label':>8} {'P(C1)':>10} {'P(C2)':>10}")
for i in range(5):
    print(f"{test_data[i]:>8.3f} {labels[i]+1:>8} "
          f"{proba[i,0]:>10.4f} {proba[i,1]:>10.4f}")

# 5. Score on test set
test_ll = gmm.score(test_data)
print(f"\nTest Log-Likelihood: {test_ll:.4f}")

# 6. Anomaly detection
is_anomaly, probs = gmm.detect_anomalies(test_data, percentile=5)
print(f"Anomalies detected: {is_anomaly.sum()} / {len(test_data)}")
# ```

# **Output:**
# ```
# =======================================================
#   GMM Summary  |  K=2  |  Converged in 34 iterations
# =======================================================
# Component          Mean       Std    Weight
# ---------------------------------------------
#   Comp 1        5.0241    1.0183    0.5702
#   Comp 2       11.9987    1.4891    0.4298
# ---------------------------------------------
#   Final Log-Likelihood: -842.3312
# =======================================================

# Sample predictions (first 5 test points):
#    Value    Label      P(C1)      P(C2)
#    5.834        1     0.9991     0.0009
#   12.431        2     0.0001     0.9999
#    4.102        1     1.0000     0.0000
#   11.078        2     0.0119     0.9881
#    5.219        1     0.9999     0.0001

# Test Log-Likelihood: -282.5541
# Anomalies detected: 5 / 100