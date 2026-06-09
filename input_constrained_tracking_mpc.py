import numpy as np
from scipy.optimize import LinearConstraint

np.random.seed(1)

from input_constrained_tracking_mpc_setting import A, B, n, m, Hor, Q, R, simulation_name
from core.optimal_control_problem import OptimalControlProblem
import core.matplotlib_style as pltstyle


SimStep = 30
ocp = OptimalControlProblem(A, B)
ocp.set_nominal_ocp(Q, R, Hor)

# Initial states
initial_states = {
    "Init1": np.array([ -2.0, 2.0, -5.0, 2.0]),
    "Init2": np.array([ -9.0, 10.0, -2.0, 1.0])
}

input_bound = [-6, 6]

# === Desired Trajectory ===

period = 8
position_d =  np.concatenate([np.array([2 * np.sin(t*2*np.pi/period), 4 * np.sin(t*2*np.pi/period)]) for t in range(period)])
ocp.calculate_periodic_desired_trajectory(position_d, period)
Xref, Uref = ocp.get_reference_from_periodic(SimStep)
ocp.set_references(SimStep, Xref, Uref)


# ========== Numerical Examples of Contracting MPC ============#

def run_simulations(initial_states):
    # --- Nonlinear cost functions ---
    def f0_zero(x_k, u_k, t, xd_k, ud_k):
        return 0.0

    # Define box constraints: 0 <= u[0], u[1] <= 1
    Coef = np.eye(ocp.horizon*ocp.m)  # Coefficients
    lb = input_bound[0] * np.ones(ocp.horizon*ocp.m)  # Lower bound
    ub = input_bound[1] * np.ones(ocp.horizon*ocp.m)  # Upper bound
    linear_constraint = LinearConstraint(Coef, lb, ub)

    # --- Cost functions ---
    nonlinear_costs = {
        "Open Loop": None,
        "Nominal MPC": (f0_zero, None),
        "MPC with Hard Constraint": (f0_zero, linear_constraint)
    }

    # --- Run simulation ---
    results = {}
    for cost_name in ["Nominal MPC", "MPC with Hard Constraint"]:
        cost_fn, constraints = nonlinear_costs[cost_name]
        results[cost_name] = {}
        for xname, x0 in initial_states.items():
            x_res, u_res = ocp.run_simulation(x0, SimStep, stage_regularizer=cost_fn, constraints=constraints)
            results[cost_name][xname] = {"x": x_res, "u": u_res}

    results["Open Loop"] = {}
    for xname, x0 in initial_states.items():
        x_res, u_res = ocp.run_simulation_without_control(x0, SimStep)
        results["Open Loop"][xname] = {"x": x_res, "u": u_res}

    return results, nonlinear_costs


if __name__ == "__main__":
    results, nonlinear_costs = run_simulations(initial_states)

    linestyles = {"Open Loop": ":", "Nominal MPC": "-.", "MPC with Hard Constraint": "-"}
    colors = {"Open Loop": pltstyle.blues[6], "Nominal MPC": pltstyle.blues[4], "MPC with Hard Constraint": pltstyle.blues[2]}
    eta = np.loadtxt(f'./output/{simulation_name}/LMI_solutions/eta.txt')
    P = np.loadtxt(f'./output/{simulation_name}/LMI_solutions/P.txt')
    norms = pltstyle.calculate_norm(results, P, nonlinear_costs)
    trajectories = {key: results[key] for key in ["Nominal MPC", "MPC with Hard Constraint"]}
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
        ylim_norm = [1e-2, 5e1]
    )
