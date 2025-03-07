import numpy as np

# Define constants
J = 1  # Exchange interaction strength (can be adjusted)
k_B = 1  # Boltzmann constant (can be adjusted)
T = 1  # Temperature (can be adjusted)

# Number of magnon states (N = 3 for this case)
N = 3

# Energy function for magnon states (Ek = 2J * sin^2(πk / N))
def energy(k, J, N):
    return 2 * J * np.sin(np.pi * k / N)**2

# Function to calculate transition probabilities
def transition_probabilities(E, T, k_B):
    """Calculate the Boltzmann-type transition probability matrix."""
    P = np.zeros((N, N))
    
    for k in range(N):
        for k_prime in range(N):
            delta_E = E[k] - E[k_prime]
            P[k, k_prime] = np.exp(-delta_E / (k_B * T))
    
    # Normalize the rows so that the sum of each row is 1 (probabilities should sum to 1)
    for k in range(N):
        P[k, :] /= np.sum(P[k, :])
    
    return P

# Energy values for each magnon state (k = 0, 1, 2)
E = np.array([energy(k, J, N) for k in range(N)])

# Calculate the transition matrix using the Boltzmann factor
P = transition_probabilities(E, T, k_B)

# Output the results
print("Energy values for magnon states (k = 0, 1, 2):")
print(E)

print("\nTransition Matrix (Magnon Basis):")
print(P)

"""
printed result;
Energy values for magnon states (k = 0, 1, 2):
[0.  1.5 1.5]

Transition Matrix (Magnon Basis):
[[0.10036756 0.44981622 0.44981622]
 [0.10036756 0.44981622 0.44981622]
"""