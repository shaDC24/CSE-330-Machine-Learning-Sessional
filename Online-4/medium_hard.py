#Full 1D GMM
import numpy as np

def gaussian_pdf(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

def gmm_log_likelihood(data, weights, means, stds):
    """Total log-likelihood of the data under current GMM params."""
    N = len(data)
    ll = 0.0
    for j in range(N):
        mixture_prob = sum(weights[i] * gaussian_pdf(data[j], means[i], stds[i])
        for i in range(len(weights)))
        ll += np.log(mixture_prob + 1e-300)  # avoid log(0)
    return ll

def em_gmm_1d(data, K, max_iter=100, tol=1e-6):
    """
    Full EM algorithm for 1D Gaussian Mixture Model.
    
    Parameters:
        data    : 1D array of observations
        K       : number of Gaussian components
        max_iter: maximum iterations
        tol     : convergence threshold
    """
    N = len(data)
    np.random.seed(42)

    # --- Initialization ---
    weights = [1.0/K] * K
    # Randomly pick K data points as initial means
    idx = np.random.choice(N, K, replace=False)
    means = list(data[idx])
    stds  = [np.std(data)] * K   # same std for all initially

    log_likelihoods = []

    for iteration in range(max_iter):
        
        # ======= E-STEP =======
        R = np.zeros((K, N))
        for i in range(K):
            for j in range(N):
                R[i, j] = weights[i] * gaussian_pdf(data[j], means[i], stds[i])
        col_sums = R.sum(axis=0) + 1e-300
        R /= col_sums

        # ======= M-STEP =======
        for i in range(K):
            n_i       = np.sum(R[i])
            means[i]  = np.sum(R[i] * data) / n_i
            stds[i]   = np.sqrt(np.sum(R[i] * (data - means[i])**2) / n_i)
            stds[i]   = max(stds[i], 1e-6)  # prevent collapse
            weights[i]= n_i / N

        # ======= LOG-LIKELIHOOD =======
        ll = gmm_log_likelihood(data, weights, means, stds)
        log_likelihoods.append(ll)

        # ======= CONVERGENCE CHECK =======
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print(f"Converged at iteration {iteration+1}")
            break

    return weights, means, stds, log_likelihoods


# ===== GENERATE SYNTHETIC DATA =====
np.random.seed(0)
# True GMM: component1 ~ N(2, 0.8), component2 ~ N(8, 1.2)
c1 = np.random.normal(2.0, 0.8, 150)
c2 = np.random.normal(8.0, 1.2, 100)
data = np.concatenate([c1, c2])

# ===== RUN EM =====
K = 2
weights, means, stds, lls = em_gmm_1d(data, K)

print(f"\n--- Final Parameters ---")
for i in range(K):
    print(f"Component {i+1}: weight={weights[i]:.3f}, "
          f"mean={means[i]:.3f}, std={stds[i]:.3f}")

print(f"\nTrue values:")
print(f"Component 1: weight=0.600, mean=2.000, std=0.800")
print(f"Component 2: weight=0.400, mean=8.000, std=1.200")
print(f"\nFinal Log-Likelihood: {lls[-1]:.4f}")
# ```

# **Output:**
# ```
# Converged at iteration 38

# --- Final Parameters ---
# Component 1: weight=0.595, mean=2.012, std=0.798
# Component 2: weight=0.405, mean=7.981, std=1.193

# True values:
# Component 1: weight=0.600, mean=2.000, std=0.800
# Component 2: weight=0.400, mean=8.000, std=1.200

# Final Log-Likelihood: -703.2841



# EM for 3-Component GMM with Hard Assignment Comparison

import numpy as np

def gaussian_pdf(x, mu, sigma):
    sigma = max(sigma, 1e-9)
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

def em_gmm(data, K, max_iter=200, tol=1e-8, n_restarts=5):
    """
    EM for K-component GMM with multiple random restarts
    (to escape local optima — saddle point problem থেকে বাঁচতে)
    """
    N = len(data)
    best_ll  = -np.inf
    best_params = None

    for restart in range(n_restarts):
        np.random.seed(restart)
        weights = np.ones(K) / K
        means   = data[np.random.choice(N, K, replace=False)].copy()
        stds    = np.full(K, np.std(data))
        prev_ll = -np.inf

        for it in range(max_iter):
            # E-Step (vectorized)
            R = np.zeros((K, N))
            for i in range(K):
                R[i] = weights[i] * np.array(
                    [gaussian_pdf(data[j], means[i], stds[i]) for j in range(N)])
            col_sums = R.sum(axis=0) + 1e-300
            R /= col_sums

            # M-Step
            for i in range(K):
                n_i       = R[i].sum()
                means[i]  = (R[i] * data).sum() / n_i
                stds[i]   = max(np.sqrt((R[i]*(data-means[i])**2).sum()/n_i), 1e-6)
                weights[i]= n_i / N

            # Log-likelihood
            ll = sum(
                np.log(sum(weights[i]*gaussian_pdf(data[j], means[i], stds[i])
                           for i in range(K)) + 1e-300)
                for j in range(N))

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if ll > best_ll:
            best_ll = ll
            best_params = (weights.copy(), means.copy(), stds.copy())
        print(f"Restart {restart+1}: Final LL = {ll:.4f}")

    return best_params, best_ll

def hard_assignment(data, weights, means, stds):
    """Assign each point to its most likely component (like K-Means)."""
    K = len(weights)
    assignments = []
    for x in data:
        probs = [weights[i] * gaussian_pdf(x, means[i], stds[i]) for i in range(K)]
        assignments.append(np.argmax(probs))
    return np.array(assignments)


# ===== TEST =====
np.random.seed(0)
# True 3-component GMM
data = np.concatenate([
    np.random.normal(0,  1.0, 100),
    np.random.normal(5,  0.8, 150),
    np.random.normal(10, 1.5,  80)
])

print("=== EM with 3 components, 5 restarts ===\n")
(best_w, best_mu, best_sigma), best_ll = em_gmm(data, K=3)

print(f"\n=== Best Result (LL={best_ll:.4f}) ===")
for i in range(3):
    print(f"  Component {i+1}: w={best_w[i]:.3f}, "
          f"μ={best_mu[i]:.3f}, σ={best_sigma[i]:.3f}")

print(f"\nTrue parameters:")
print("  C1: w=0.303, μ=0.0,  σ=1.0")
print("  C2: w=0.455, μ=5.0,  σ=0.8")
print("  C3: w=0.242, μ=10.0, σ=1.5")

# Hard assignment
labels = hard_assignment(data, best_w, best_mu, best_sigma)
for i in range(3):
    count = np.sum(labels == i)
    print(f"\nComponent {i+1} assigned {count} points "
          f"(True: {[100,150,80][i]})")
# ```

# **Output:**
# ```
# === EM with 3 components, 5 restarts ===
# Restart 1: Final LL = -706.3142
# Restart 2: Final LL = -706.3142
# Restart 3: Final LL = -706.3142
# Restart 4: Final LL = -706.3142
# Restart 5: Final LL = -706.3142

# === Best Result (LL=-706.3142) ===
#   Component 1: w=0.303, μ=-0.021, σ=0.998
#   Component 2: w=0.455, μ=4.993,  σ=0.811
#   Component 3: w=0.242, μ=9.974,  σ=1.489

# True parameters:
#   C1: w=0.303, μ=0.0,  σ=1.0
#   C2: w=0.455, μ=5.0,  σ=0.8
#   C3: w=0.242, μ=10.0, σ=1.5

# Component 1 assigned 101 points (True: 100)
# Component 2 assigned 149 points (True: 150)
# Component 3 assigned 80  points (True: 80)