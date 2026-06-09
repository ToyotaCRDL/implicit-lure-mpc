import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

from input_penalized_mpc_setting import A, B, n, m, Hor, Q, R, Lip, simulation_name
from core.optimal_control_problem import OptimalControlProblem
import core.matplotlib_style as pltstyle


SimStep = 40

# Initial states
initial_states = {
    "Init1": np.array([-1.0,-0.8, 1.0, 0.5]),
    "Init2": np.array([1.0, -1.0, 1.0, 1.0])
}

input_bound = [-1.0, 0.5]


# ========== Numerical Examples of Contracting MPC ============#

def run_simulations(ocp, initial_states):
    # --- Nonlinear cost functions ---
    def f0_zero(x_k, u_k, t, xd_k, ud_k):
        return 0.0

    def flat_quadratic(a, t, th_flat):
        if a > th_flat:
            return 0.5 * (a - th_flat)**2
        if a < -th_flat:
            return 0.5 * (a + th_flat)**2
        return 0.0

    def f2_flat_quadratic(x_k, u_k, t, xd_k, ud_k):
        offset_u = (input_bound[0] + input_bound[1]) / 2
        th_u = (input_bound[1] - input_bound[0]) / 2
        tilde_u_k = u_k - offset_u
        regfunc_u = np.sum([flat_quadratic(a, t, th_u) for a in tilde_u_k])
        return Lip * regfunc_u

    # --- Cost functions ---
    nonlinear_costs = {
        "Open Loop": None,
        "Nominal MPC": f0_zero,
        "MPC with Soft Barrier": f2_flat_quadratic
    }

    # --- Run simulations ---
    ocp.set_references(SimStep)
    results = {}
    for cost_name in ["Nominal MPC", "MPC with Soft Barrier"]:
        cost_fn = nonlinear_costs[cost_name]
        results[cost_name] = {}
        for xname, x0 in initial_states.items():
            x_res, u_res = ocp.run_simulation(x0, SimStep, stage_regularizer=cost_fn)
            results[cost_name][xname] = {"x": x_res, "u": u_res}

    results["Open Loop"] = {}
    for xname, x0 in initial_states.items():
        x_res, u_res = ocp.run_simulation_without_control(x0, SimStep)
        results["Open Loop"][xname] = {"x": x_res, "u": u_res}

    return results, nonlinear_costs


if __name__ == "__main__":
    ocp = OptimalControlProblem(A, B)
    ocp.set_nominal_ocp(Q, R, Hor)

    results, nonlinear_costs = run_simulations(ocp, initial_states)
    linestyles = {
        "Open Loop": ":",
        "Nominal MPC": "-.",
        "MPC with Soft Barrier": "-"
        }
    colors = {
        "Open Loop": pltstyle.blues[6],
        "Nominal MPC": pltstyle.blues[4],
        "MPC with Soft Barrier": pltstyle.blues[2]
        }

    eta = np.loadtxt(f'./output/{simulation_name}/LMI_solutions/eta.txt')
    P = np.loadtxt(f'./output/{simulation_name}/LMI_solutions/P.txt')
    norms = pltstyle.calculate_norm(results, P, nonlinear_costs)
    trajectories = {key: results[key] for key in ["Nominal MPC", "MPC with Soft Barrier"]}
    pltstyle.plot_norm_and_input_for_paper(
        trajectories,
        norms,
        nonlinear_costs,
        eta,
        x_lim=20,
        linestyles=linestyles,
        colors=colors,
        filename=f'./output/{simulation_name}/figures/trajectories_for_paper.pdf',
        input_bound=input_bound,
        ylim_norm = [4e-0, 1e2],
        ylim_input = [ [-2.5, 2.5], [-1.5, 2.5] ]
    )
