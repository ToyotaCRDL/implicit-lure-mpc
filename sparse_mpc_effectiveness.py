import numpy as np

from core.matplotlib_style import plot_statistics
from core.optimal_control_problem import OptimalControlProblem
from sparse_mpc_setting import A, B, n, m, Hor, Q, R, simulation_name


n_ic = 50  # number of initial conditions
simstep = 40
coeff_sparse_list = [0.0, 0.5, 1.0, 1.5, 5.0]


def calculate_statistics(ocp, n_ic, simstep, coeff_sparse_list):
    x0_list = [np.random.uniform(-3, 3, size=n) for _ in range(n_ic)]  # random ICs in [-3, 3]
    zero_counts_list = []

    for i, coeff in enumerate(coeff_sparse_list):
        zero_counts_all = []  # to store zero counts for all ICs
        print(coeff)
        for x0 in x0_list:
            x_hist, u_hist = ocp.run_simulation_sparse(x0, T=simstep, lambda_stage=coeff)
            # Count effectively zero inputs at each timestep
            zero_count = np.sum(np.abs(u_hist) < 1e-6, axis=1)
            zero_counts_all.append(zero_count)

        zero_counts_np = np.array(zero_counts_all)
        zero_counts_list.append(zero_counts_np)
    return zero_counts_list


if __name__ == "__main__":
    ocp = OptimalControlProblem(A, B)
    ocp.set_nominal_ocp(Q, R, Hor)
    ocp.set_references(simstep)
    zero_counts_list = calculate_statistics(ocp, n_ic, simstep, coeff_sparse_list)
    plot_statistics(
        coeff_sparse_list,
        zero_counts_list,
        n_ic,
        filename=f'./output/{simulation_name}/figures/sparsity_statistics_{n_ic}_inits.pdf'
    )
