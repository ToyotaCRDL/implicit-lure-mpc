import numpy as np

np.random.seed(1)

from input_penalized_mpc_setting import A, B, n, Hor, Q, R
from input_penalized_mpc import run_simulations, SimStep, input_bound
from core.optimal_control_problem import OptimalControlProblem


n_ic = 50  # number of initial conditions

x0_list = [np.random.uniform(-3, 3, size=n) for _ in range(n_ic)]  # random ICs in [-3, 3]

# Initial states
initial_states = {
    "Init1": np.array([-1.0,-0.8, 1.0, 0.5]),
    "Init2": np.array([1.0, -1.0, 1.0, 1.0])
}

input_bound = [-1.0, 0.5]


# ========== Numerical Examples of Contracting MPC ============#

def sum_input_penalty(trajectory):
    sum = 0.0
    for t in range(SimStep):
        u_t = trajectory["u"][t]
        sum += absolute_sum_violation(u_t)
    return sum


def absolute_sum_violation(u_t):
    # calculate sum of absolute values of violations of input bound for each input dimension
    sum_violation = 0.0
    for i in range(len(u_t)):
        if u_t[i] < input_bound[0]:
            sum_violation += abs(u_t[i] - input_bound[0])
        elif u_t[i] > input_bound[1]:
            sum_violation += abs(u_t[i] - input_bound[1])
    return sum_violation


def check_violation(ocp, x0_list):
    penalties_reg = []
    penalties_nom = []
    for x0 in x0_list:
        results, _ = run_simulations(ocp, {"x0": x0})
        trajectories = {key: results[key] for key in ["Nominal MPC", "MPC with Soft Barrier"]}
        trajectory_reg = trajectories["MPC with Soft Barrier"]["x0"]
        penalty_reg = sum_input_penalty(trajectory_reg)
        penalties_reg.append(penalty_reg)

        trajectory_nom = trajectories["Nominal MPC"]["x0"]
        penalty_nom = sum_input_penalty(trajectory_nom)
        penalties_nom.append(penalty_nom)

        print("penalty_reg =", penalty_reg, "and penalty_nom =", penalty_nom)
        if penalty_reg > penalty_nom:
            print(">>> Unexpected <<< The penalty for regularized MPC is larger than for nominal MPC.")
            return False

    print("As expected, penalty for regularized MPC is smaller than nominal MPC.")
    penalty_reg_mean = np.mean(penalties_reg)
    penalty_nom_mean = np.mean(penalties_nom)
    rate = penalty_reg_mean / penalty_nom_mean
    print("Average penalty for regularized MPC:", penalty_reg_mean)
    print("Average penalty for nominal MPC:", penalty_nom_mean)
    print("Rate of regularized to nominal penalty:", rate)
    return True


if __name__ == "__main__":
    ocp = OptimalControlProblem(A, B)
    ocp.set_nominal_ocp(Q, R, Hor)
    has_smaller_violation = check_violation(ocp, x0_list)
