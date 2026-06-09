import numpy as np

np.random.seed(1)

from state_penalized_mpc_setting import A, B, n, m, Hor, Q, R, simulation_name
from core.optimal_control_problem import OptimalControlProblem
import core.matplotlib_style as pltstyle


SimStep = 30

ocp = OptimalControlProblem(A, B)
ocp.set_nominal_ocp(Q, R, Hor)
ocp.set_references(SimStep)

# Initial states
initial_states = {
    "Init1": np.array([-0.5,  3.5, -2.5,  1.5]),
    "Init2": np.array([-2.0, -2.0, -2.0, -2.0]),
}

Lip = 0.6  # execute contractivity_check and get the maximum Lipschitz constant

state_bound = [-0.6, 0.4]  # state bounds for plotting


# ========== Numerical Examples of Contracting MPC ============#

norm_BbarT = np.linalg.norm(ocp.Bbar.T, 2)  # induced 2-norm (maximum singular value)
print("induced 2-norm of BbarT:", norm_BbarT)

Lip_X = Lip / np.sqrt(2) / norm_BbarT
Lip_U = Lip / np.sqrt(2)
print("Admitted Lipschitz constant of nabla_Vreg wrt. X:", Lip_X)
print("Admitted Lipschitz constant of nabla_Vreg wrt. U:", Lip_U)


def run_simulations(initial_states):
    # --- Regularizers ---
    def flat_quadratic(a, t, th_flat):
        if a > th_flat:
            return 0.5 * (a - th_flat)**2
        if a < -th_flat:
            return 0.5 * (a + th_flat)**2
        return 0.0

    def f1_flat_quadratic(x_k, u_k, t, xd_k, ud_k):
        offset_x = (state_bound[0] + state_bound[1]) / 2
        th_x = (state_bound[1] - state_bound[0]) / 2
        tilde_x_k = x_k - offset_x
        regfunc_x = np.sum([flat_quadratic(a, t, th_x) for a in tilde_x_k])
        offset_u = 0
        tilde_u_k = u_k - offset_u
        regfunc_u = - 0.5*np.linalg.norm(tilde_u_k)**2  # relaxing the quadratic penalty on input
        return Lip_X * regfunc_x + Lip_U * regfunc_u

    def f1_terminal_flat_quadratic(x_k, t, xd_k):
        offset_x = (state_bound[0] + state_bound[1]) / 2
        th_x = (state_bound[1] - state_bound[0]) / 2
        tilde_x_k = x_k - offset_x
        regfunc_x = np.sum([flat_quadratic(a, t, th_x) for a in tilde_x_k])
        return Lip_X * regfunc_x

    # --- Regularizing stage cost functions ---
    regularizing_costs = {
        "Open Loop": None,
        "Nominal MPC": None,
        "MPC with Soft Barrier": f1_flat_quadratic
    }

    # --- Regularizing terminal cost functions ---
    regularizing_terminal_costs = {
        "Open Loop": None,
        "Nominal MPC": None,
        "MPC with Soft Barrier": f1_terminal_flat_quadratic
    }

    # --- Run simulation ---
    results = {}
    for cost_name in ["Nominal MPC", "MPC with Soft Barrier"]:
        cost_fn = regularizing_costs[cost_name]
        terminal_cost_fn = regularizing_terminal_costs[cost_name]
        results[cost_name] = {}
        for xname, x0 in initial_states.items():
            x_res, u_res = ocp.run_simulation(x0, SimStep, cost_fn, terminal_cost_fn)
            results[cost_name][xname] = {"x": x_res, "u": u_res}

    # --- Run simulation without input ---
    regularizing_costs["Open Loop"] = None
    regularizing_terminal_costs["Open Loop"] = None
    results["Open Loop"] = {}
    for xname, x0 in initial_states.items():
        x_res, u_res = ocp.run_simulation_without_control(x0, SimStep)
        results["Open Loop"][xname] = {"x": x_res, "u": u_res}

    return results, regularizing_costs


if __name__ == "__main__":
    results, regularizing_costs = run_simulations(initial_states)

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

    eta = np.loadtxt(f'./output/{simulation_name}/LMI_solutions/eta.txt')   # make sure if path is correct
    P = np.loadtxt(f'./output/{simulation_name}/LMI_solutions/P.txt')   # make sure if path is correct
    norms = pltstyle.calculate_norm(results, P, regularizing_costs)
    pltstyle.plot_norm_and_state_for_paper(
        results,
        norms,
        regularizing_costs,
        eta,
        x_lim=20,
        linestyles=linestyles,
        colors=colors,
        filename=f'./output/{simulation_name}/figures/trajectories_for_paper.pdf',
        state_bound=state_bound,
        ylim_norm = [4e-1, 1e2],
        ylim_state = [[-2.5, 2.5], [-2.5, 2.5]]
    )
    print("Trajectories of the state:")
    print(results["MPC with Soft Barrier"]["Init1"]["x"])
