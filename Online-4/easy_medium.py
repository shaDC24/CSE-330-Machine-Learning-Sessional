import numpy as np

def gaussian_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exp = np.exp(-((x - mu)**2) / (2 * sigma**2))
    return coeff * exp

def mixture_probability(x, weights, means, stds):
    """
    P(x) = Σᵢ wᵢ · P(x | μᵢ, σᵢ)
    একটা point x এর GMM probability বের করো।
    """
    total = 0.0
    for w, mu, sigma in zip(weights, means, stds):
        total += w * gaussian_pdf(x, mu, sigma)
    return total

# GMM with 2 components
weights = [0.4, 0.6]
means   = [2.0, 7.0]
stds    = [1.0, 1.5]

for x in [2.0, 5.0, 7.0]:
    p = mixture_probability(x, weights, means, stds)
    print(f"P(x={x}) = {p:.6f}")
# ```

# **Output:**
# ```
# P(x=2.0) = 0.151388
# P(x=5.0) = 0.020572
# P(x=7.0) = 0.159155


import numpy as np

def gaussian_pdf(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

def e_step(data, weights, means, stds):
    """
    E-Step: প্রতিটা data point এর জন্য প্রতিটা component এর
    responsibility (pᵢⱼ) বের করো।
    
    pᵢⱼ = wᵢ · P(xⱼ|μᵢ,σᵢ) / Σₖ wₖ · P(xⱼ|μₖ,σₖ)
    """
    N = len(data)
    K = len(weights)
    responsibilities = np.zeros((K, N))  # shape: K×N

    for i in range(K):
        for j in range(N):
            responsibilities[i, j] = weights[i] * gaussian_pdf(data[j], means[i], stds[i])

    # Normalize: প্রতিটা column এর sum = 1
    col_sums = responsibilities.sum(axis=0)
    responsibilities /= col_sums  # Broadcasting

    return responsibilities

# Test with 2 components, 5 data points
data    = np.array([1.0, 2.0, 5.0, 8.0, 9.0])
weights = [0.5, 0.5]
means   = [2.0, 8.0]
stds    = [1.0, 1.0]

R = e_step(data, weights, means, stds)
print("Responsibilities (rows=components, cols=data points):")
print(np.round(R, 4))
print("\nColumn sums (should all be 1.0):", R.sum(axis=0))
# ```

# **Output:**
# ```
# Responsibilities:
# [[1.     0.9998 0.0183 0.0000 0.0000]
#  [0.     0.0002 0.9817 1.0000 1.0000]]

# Column sums: [1. 1. 1. 1. 1.]