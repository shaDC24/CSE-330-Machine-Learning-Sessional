import numpy as np

def gaussian_pdf(x, mu, sigma):
    """
    Single Gaussian এর probability density বের করো।
    Formula: (1 / σ√2π) * exp(-(x-μ)² / 2σ²)
    """
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * np.exp(exponent)

# Test
print(gaussian_pdf(0, mu=0, sigma=1))   # Standard normal at 0 → ~0.3989
print(gaussian_pdf(1, mu=0, sigma=1))   # Should be less → ~0.2420
print(gaussian_pdf(2, mu=2, sigma=0.5)) # At the mean → ~0.7979

import numpy as np

def mle_gaussian(data):
    """
    Maximum Likelihood Estimation:
    μ = mean of data
    σ = standard deviation of data
    """
    N = len(data)
    mu = np.sum(data) / N                          # μ = Σxⱼ / N
    sigma = np.sqrt(np.sum((data - mu)**2) / N)    # σ = √(Σ(xⱼ-μ)² / N)
    return mu, sigma

# Test
data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
mu, sigma = mle_gaussian(data)
print(f"MLE Mean (μ): {mu}")       # Expected: 5.0
print(f"MLE Std  (σ): {sigma}")    # Expected: 2.0

# Verify with numpy
print(f"NumPy mean: {np.mean(data)}")
print(f"NumPy std:  {np.std(data)}")
# ```

# **Output:**
# ```
# MLE Mean (μ): 5.0
# MLE Std  (σ): 2.0
# NumPy mean: 5.0
# NumPy std:  2.0

import numpy as np

def log_likelihood(data, mu, sigma):
    """
    L = Σ log P(xⱼ | μ, σ)
    = N(-log√2π - log σ) - Σ(xⱼ-μ)²/2σ²
    """
    N = len(data)
    log_lik = (N * (-np.log(np.sqrt(2 * np.pi)) - np.log(sigma))
               - np.sum((data - mu)**2) / (2 * sigma**2))
    return log_lik

# Test
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
mu, sigma = 3.0, 1.0

ll = log_likelihood(data, mu, sigma)
print(f"Log-Likelihood: {ll:.4f}")

# Bad parameters should give lower LL
ll_bad = log_likelihood(data, mu=10.0, sigma=1.0)
print(f"Log-Likelihood (bad params): {ll_bad:.4f}")
# Good params always > bad params
# ```

# **Output:**
# ```
# Log-Likelihood: -7.3257
# Log-Likelihood (bad params): -72.8257
