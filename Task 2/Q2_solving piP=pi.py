import numpy as np

def find_stationary_distribution(P):
    """
    Finds the stationary distribution pi of a Markov chain.
    
    Args:
        P (numpy.ndarray): Transition matrix of the Markov chain (size n x n).
    
    Returns:
        numpy.ndarray: The stationary distribution vector pi (size n).
    """
    # Number of states in the Markov chain
    n = P.shape[0]
    
    # Create the matrix (P - I), where I is the identity matrix
    I = np.eye(n)
    A = P - I
    
   
    
    # Append the normalization condition
    A = np.vstack([A, np.ones(n)])  # Add a row of ones at the bottom
    b = np.zeros(n)  # The right-hand side of the equation (including the normalization)
    b = np.append(b, 1)  # Add a 1 at the end
    
    # Solve the system of linear equations A * pi = b
    pi = np.linalg.lstsq(A, b, rcond=None)[0]  # Use lstsq to solve the least-squares problem
    
    return pi

# Example usage
P = np.array([[0.5, 0.5],  # Example transition matrix
              [0.2, 0.8]])

pi = find_stationary_distribution(P)

print("Stationary Distribution:")
print(pi)

"""
printed result 
Stationary Distribution:
[0.5 0.5]

"""