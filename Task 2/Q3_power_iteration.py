import numpy as np

def power_iteration(P, pi_0, max_iter=1000, tolerance=1e-6):
    """
    Perform power iteration to find the stationary distribution.
    
    Args:
        P (numpy.ndarray): Transition matrix of the Markov chain.
        pi_0 (numpy.ndarray): Initial guess for the stationary distribution.
        max_iter (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
        
    Returns:
        numpy.ndarray: The stationary distribution vector pi.
    """
    pi = pi_0
    for i in range(max_iter):
        # Update the distribution
        pi_next = np.dot(pi, P)
        
        # Normalize the result
        pi_next = pi_next / np.sum(pi_next)
        
        # Check for convergence (if the change is smaller than tolerance)
        if np.linalg.norm(pi_next - pi) < tolerance:
            print(f"Converged after {i+1} iterations.")
            return pi_next
        
        pi = pi_next
    
    # Return the last computed distribution if max_iter is reached
    return pi

# Example transition matrix for a 3-state system
P = np.array([[0.5, 0.3, 0.2],  
              [0.4, 0.4, 0.2],
              [0.3, 0.2, 0.5]])

# 1) Initial guess: Pr(|↑↑↑⟩) = 1
pi_1 = np.array([1, 0, 0])

# 2) Initial guess: Pr(|↑↑↑⟩) = 1/2, Pr(|↓↑↓⟩) = 1/2
pi_2 = np.array([0.5, 0, 0.5])

# 3) Initial guess: Uniform distribution
pi_3 = np.array([1/3, 1/3, 1/3])

# Run power iteration for all initial guesses
pi_1_stationary = power_iteration(P, pi_1)
pi_2_stationary = power_iteration(P, pi_2)
pi_3_stationary = power_iteration(P, pi_3)

# Print the results
print("Stationary Distribution for Initial Guess 1 (Pr(|↑↑↑⟩) = 1):")
print(pi_1_stationary)

print("\nStationary Distribution for Initial Guess 2 (Pr(|↑↑↑⟩) = 1/2, Pr(|↓↑↓⟩) = 1/2):")
print(pi_2_stationary)

print("\nStationary Distribution for Initial Guess 3 (Uniform Distribution):")
print(pi_3_stationary)
"""
printed result;
Converged after 12 iterations.
Converged after 12 iterations.
Converged after 10 iterations.
Stationary Distribution for Initial Guess 1 (Pr(|↑↑↑⟩) = 1):
[0.41269849 0.30158738 0.28571413]

Stationary Distribution for Initial Guess 2 (Pr(|↑↑↑⟩) = 1/2, Pr(|↓↑↓⟩) = 1/2):
[0.41269836 0.30158724 0.2857144 ]

Stationary Distribution for Initial Guess 3 (Uniform Distribution):
[0.41269827 0.30158716 0.28571457]
"""