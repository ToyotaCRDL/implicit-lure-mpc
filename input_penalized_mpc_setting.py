import numpy as np
from scipy.signal import cont2discrete

from core.optimal_control_problem import OptimalControlProblem
from core.contractivity_LMI_checker import LMI_checker

# simulation name (used as ID)
simulation_name = "input_penalized_mpc"

# Dimensions
n = 4      # state dimension
m = 2      # input dimension
Hor = 5    # horizon

m1 = 9.0
m2 = 10.0
k1 = 5.0
k2 = 6.0
c1 = 0.0
c2 = 0.0
Ts = 1.0  # Sampling period


# Cost matrices
Q = 5 * np.diag([1,1,1,1])
R = 1 * np.eye(m)

# Contractivity
eta = 0.99  # contractivity factor

Lip = 10.0
Theta = np.eye(m) * Lip  # L-smoothness


# ======================
# State variable: [q1, v1, q2, v2]
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

C_c = np.eye(n)
D_c = np.zeros((n, m))
sys_d = cont2discrete((A_c, B_c, C_c, D_c), Ts, method='zoh')
A, B, C, D, _ = sys_d

if __name__ == "__main__":
    ocp = OptimalControlProblem(A, B)
    ocp.set_nominal_ocp(Q, R, Hor)
    checker = LMI_checker(ocp, regularizer_type="convex_smooth", simulation_name=simulation_name)
    checker.check_contractivity(Hor, eta, eps_P=1e-1, eps_lmi=5e-2, Theta=Theta)
    ocp.baseline_contraction_rate()
