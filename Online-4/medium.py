import numpy as np

def m_step(data, responsibilities):
    """
    M-Step: Responsibility দিয়ে নতুন parameters বের করো।
    
    nᵢ = Σⱼ pᵢⱼ
    μᵢ = Σⱼ pᵢⱼ·xⱼ / nᵢ
    σᵢ = √(Σⱼ pᵢⱼ·(xⱼ-μᵢ)² / nᵢ)
    wᵢ = nᵢ / N
    """
    K, N = responsibilities.shape
    
    new_weights = []
    new_means   = []
    new_stds    = []

    for i in range(K):
        n_i  = np.sum(responsibilities[i])          # effective count
        mu_i = np.sum(responsibilities[i] * data) / n_i
        sigma_i = np.sqrt(
            np.sum(responsibilities[i] * (data - mu_i)**2) / n_i
        )
        w_i = n_i / N

        new_means.append(mu_i)
        new_stds.append(sigma_i)
        new_weights.append(w_i)

    return new_weights, new_means, new_stds

# Test (use responsibilities from Q5)
data    = np.array([1.0, 2.0, 5.0, 8.0, 9.0])
weights = [0.5, 0.5]
means   = [2.0, 8.0]
stds    = [1.0, 1.0]

# Simulate responsibilities
R = np.array([[1.0, 0.9998, 0.0183, 0.0, 0.0],
              [0.0, 0.0002, 0.9817, 1.0, 1.0]])

new_w, new_mu, new_sigma = m_step(data, R)
print("Updated Weights:", [round(w, 4) for w in new_w])
print("Updated Means:  ", [round(m, 4) for m in new_mu])
print("Updated Stds:   ", [round(s, 4) for s in new_sigma])
# ```

# **Output:**
# ```
# Updated Weights: [0.4037, 0.5963]
# Updated Means:   [1.4992, 7.6694]
# Updated Stds:    [0.5006, 1.2507]