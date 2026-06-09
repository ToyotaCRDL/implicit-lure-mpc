import numpy as np
from scipy.signal import cont2discrete

from core.optimal_control_problem import OptimalControlProblem
from core.contractivity_LMI_checker import LMI_checker

# simulation name (used as ID)
simulation_name = "input_constrained_tracking_mpc"

# Dimensions
n = 4      # State dimension
m = 2      # Input dimension
Hor = 15   # Horizon

m1 = 9.0
m2 = 8.0
k1 = 7.0
k2 = 6.0
c1 = 5.0
c2 = 4.0
Ts = 1.0  # Sampling period


# Cost matrices
Q = 10 * np.diag([1,1,1,1])
R = 1 * np.eye(m)

# Contractivity
# eta = 0.95  # contractivity factor
eta = 0.95  # contractivity factor


# State vector: [x1, v1, x2, v2]
A_c = np.array([
    [0, 1, 0, 0],
    [-(k1 + k2)/m1, -(c1 + c2)/m1, k2/m1, c2/m1],
    [0, 0, 0, 1],
    [k2/m2, c2/m2, -k2/m2, -c2/m2]
])

B_c = np.array([
    [0, 0],
    [1/m1, 0],
    [0, 0],
    [0, 1/m2]
])

C = np.eye(n)
D = np.zeros((n, m))
sys_d = cont2discrete((A_c, B_c, C, D), Ts, method='zoh')
A, B, _, _, _ = sys_d

print("A =\n", A)
print("B =\n", B)

if __name__ == "__main__":
    ocp = OptimalControlProblem(A, B)
    ocp.set_nominal_ocp(Q, R, Hor)
    checker = LMI_checker(ocp, regularizer_type="ccp", simulation_name=simulation_name)
    checker.check_contractivity(Hor, eta, eps_P=1e-1, eps_lmi=1e-2)
    ocp.baseline_contraction_rate()
