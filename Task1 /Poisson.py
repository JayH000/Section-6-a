"""
code should; plot probability 
distribution given a calculated probability distribution of 
P(R)

"""
import numpy as np
import matplotlib.pyplot as plt

# Define constants
n = 0.1  # Star density (stars per cubic light-year, for example)
R_values = np.linspace(0, 10, 1000)  # Range of distances to analyze

# Define probability density function P(R)
def P(R, n):
    return 4 * np.pi * n * R**2 * np.exp(- (4/3) * np.pi * n * R**3)

# Compute P(R) for different R values
P_values = P(R_values, n)

# Plot the probability distribution
plt.figure(figsize=(8,6))
plt.plot(R_values, P_values, label=f'Star Density n = {n} stars/unit³', color='b')
plt.xlabel("Distance to Nearest Star (R)")
plt.ylabel("Probability Density P(R)")
plt.title("Probability Distribution of Nearest Star Distance")
plt.legend()
plt.grid()
plt.show()
