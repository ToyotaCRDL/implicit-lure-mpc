import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

from sparse_mpc_setting import A, B, Hor, Q, R, simulation_name
from core.optimal_control_problem import OptimalControlProblem
import core.matplotlib_style as pltstyle


SimStep = 40

# Initial states
initial_states = {
    "Init1": np.array([-3.0,-3, 2.0, 1.0, -5]),
    "Init2": np.array([1.0, -1.0, 1.0, 1.0, 3.0])
}

coeff_sparse = 1.5


def run_simulations(initial_states):
    # --- Nonlinear cost functions ---
    def f0_zero(x_k, u_k, t, xd_k, ud_k):
        return 0.0

    # --- Cost functions ---
    nonlinear_costs = {
        "Open Loop": None,
        "Nominal MPC": f0_zero,
        "Sparse MPC": None
    }

    # --- Run simulation ---
    results = {}
    for cost_name in ["Nominal MPC"]:
        cost_fn = nonlinear_costs[cost_name]
        results[cost_name] = {}
        for xname, x0 in initial_states.items():
            x_res, u_res = ocp.run_simulation(x0, SimStep, stage_regularizer=cost_fn)
            results[cost_name][xname] = {"x": x_res, "u": u_res}

    # Sparsification by L1 regularizer
    results["Sparse MPC"] = {}
    for xname, x0 in initial_states.items():
        x_res, u_res = ocp.run_simulation_sparse(x0, T=SimStep, lambda_stage=coeff_sparse)
        results["Sparse MPC"][xname] = {"x": x_res, "u": u_res}

    # --- Run simulation without input ---
    results["Open Loop"] = {}
    for xname, x0 in initial_states.items():
        x_res, u_res = ocp.run_simulation_without_control(x0, SimStep)
        results["Open Loop"][xname] = {"x": x_res, "u": u_res}

    return results, nonlinear_costs


if __name__ == "__main__":
    ocp = OptimalControlProblem(A, B)
    ocp.set_nominal_ocp(Q, R, Hor)
    ocp.set_references(SimStep)
    results, nonlinear_costs = run_simulations(initial_states)

    linestyles = {
        "Open Loop": ":",
        "Nominal MPC": "-.",
        "Sparse MPC": "-"
    }
    colors = {
        "Open Loop": pltstyle.blues[6],
        "Nominal MPC": pltstyle.blues[4],
        "Sparse MPC": pltstyle.blues[2]
    }

    eta = np.loadtxt(f'./output/{simulation_name}/LMI_solutions/eta.txt')   # make sure if path is correct
    P = np.loadtxt(f'./output/{simulation_name}/LMI_solutions/P.txt')   # make sure if path is correct
    norms = pltstyle.calculate_norm(results, P, nonlinear_costs)
    trajectories = {key: results[key] for key in ["Nominal MPC", "Sparse MPC"]}
    pltstyle.plot_norm_and_input_for_paper(
        trajectories,
        norms,
        nonlinear_costs,
        eta,
        x_lim=20,
        linestyles=linestyles,
        colors=colors,
        filename=f'./output/{simulation_name}/figures/trajectories_for_paper.pdf',
        ylim_norm=[1e-1, 1e1],
        input_idx=[1,3]
    )
