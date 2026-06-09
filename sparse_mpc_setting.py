import numpy as np
from scipy.signal import cont2discrete

from core.optimal_control_problem import OptimalControlProblem
from core.contractivity_LMI_checker import LMI_checker

# simulation name (used as ID)
simulation_name = "sparse_mpc"

# Dimensions
n = 5      # State dimension
m = 8      # Input dimension
Hor = 10   # Horizon


# Contractivity rate
eta = 0.99

# Time interval
epsilon = 0.2

# Diffusion coefficient
alpha = 0.1

# Cost matrices
Q = np.eye(n) + np.ones((n, n))
r = 2.0
R = r * np.eye(m)

# Consensus network system
A = np.array([
    [1 - 2 * epsilon*alpha,   epsilon*alpha,           0,                     0,                       epsilon*alpha],
    [epsilon*alpha,           1 - 3 * epsilon*alpha,   epsilon*alpha,         0,                       epsilon*alpha],
    [-epsilon*alpha,          0,                       1 - 4 * epsilon*alpha, 0,                      -epsilon*alpha],
    [-epsilon*alpha,          -epsilon*alpha,          0,                     1 - 3 * epsilon*alpha,  -epsilon*alpha],
    [0,                       0,                       -epsilon*alpha,        -epsilon*alpha,  1 - 4 * epsilon*alpha]
])

B = epsilon * np.array([
    [ 0,  0,  0, 0, 0, 0,  1,  1],
    [ 1,  0,  1, 0, 0, 0, -1,  0],
    [-1,  1,  0, 1, 0, 0,  0,  0],
    [ 0, -1,  0, 0, 0, 1,  0,  0],
    [ 0,  0, -1, 0, 1, 0,  0, -1]
])


print("A =\n", A)
print("B =\n", B)

if __name__ == "__main__":
    ocp = OptimalControlProblem(A, B)
    ocp.set_nominal_ocp(Q, R, Hor)
    checker = LMI_checker(ocp, regularizer_type="ccp", simulation_name=simulation_name)
    checker.check_contractivity(Hor, eta, eps_P=1e-1, eps_lmi=1e-3)
    ocp.baseline_contraction_rate()
