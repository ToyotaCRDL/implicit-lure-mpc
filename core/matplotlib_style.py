import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


np.random.seed(1)


nominal_fontsize = 20
legend_fontsize = 17

plt.rcParams.update({'font.size': nominal_fontsize})

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"] })

## if you will need AMS fonts, like \mathbb, say: ax.text(1.25*R_circle, 1.0*R_circle, r"$\mathbb{C}$", fontsize=fs_C, color=col)
plt.rcParams.update({
   "text.usetex": True,
   "font.family": "serif",
   "font.serif": ["Computer Modern Roman"],
   "text.latex.preamble": r"\usepackage{amsmath,amssymb,amsfonts}"
})

# Define the requested colors from darker to lighter shades
blues = ['#002447', '#003c76', '#0055A4', '#006CD4', '#0085ff', '#239cff', '#58b1ff']
oranges = ['#471b00', '#752d00', '#a43e00', '#d35000', '#ff6100', '#ff7f1a', '#ff9b56']

grays = [
   '#1a1a1a',  # Very dark gray (almost black)
   '#333333',  # Dark gray
   '#4d4d4d',  # Medium-dark gray
   '#666666',  # Medium gray
   '#999999',  # Light-medium gray
   '#cccccc',  # Light gray
   '#e6e6e6',  # Very light gray
   '#f5f5f5',  # Whitesmoke (ultra light gray)
]


def _save_figure(filename):
    # Create parent directory automatically so plot scripts work on a fresh clone.
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf')


def plot_results(results, nonlinear_costs, initial_states, n, m, colors, linestyles, filename):

    plt.figure(figsize=(10, 20))

    # --- State ---
    for i in range(n):
        plt.subplot(n + m, 1, i + 1)
        for cost_name in nonlinear_costs:
            for xname in initial_states:
                label = f"{cost_name}, {xname}" if i == 0 else None
                plt.plot(results[cost_name][xname]["x"][:, i],
                         label=label,
                         color=colors[xname],
                         linestyle=linestyles[cost_name])
        plt.ylabel(fr"$x_{i+1}$")
        plt.xticks(np.arange(0, results[cost_name][xname]["x"].shape[0], step=10))
        plt.grid(True, axis='both', which='major', linestyle='--')
        if i == 0:
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(linestyles), fontsize=10, frameon=True)

    # --- Input ---
    for j in range(m):
        plt.subplot(n + m, 1, n + j + 1)
        for cost_name in nonlinear_costs:
            for xname in initial_states:
                label = f"{cost_name}, {xname}" if j == 0 else None
                plt.plot(results[cost_name][xname]["u"][:, j],
                         label=label,
                         color=colors[xname],
                         linestyle=linestyles[cost_name])
        plt.ylabel(fr"$u_{j+1}$")
        plt.xlabel("Time Step")
        plt.xticks(np.arange(0, results[cost_name][xname]["u"].shape[0], step=10))
        plt.grid(True, axis='both', which='major', linestyle='--')
    plt.tight_layout()
    _save_figure(filename)


def plot_norm_and_input_for_paper(trajectories, norms, nonlinear_costs, eta, x_lim, linestyles, colors, filename, input_bound=None, ylim_norm=None, ylim_input=None, input_idx=[0,1]):

    plt.figure(figsize=(10, 8))

    # --- norm ---
    plt.subplot(3, 1, 1)
    max_initial_norm = 0.0
    for cost_name in nonlinear_costs:
        initial_norm = norms[cost_name][0]
        if initial_norm > max_initial_norm:
            max_initial_norm = initial_norm
    norm_baseline = np.array([max_initial_norm * (eta**k) for k in range(x_lim+1)])
    plt.semilogy(norm_baseline,
                color=grays[4],
                linestyle="-")
    for cost_name in nonlinear_costs:
        plt.semilogy(norms[cost_name],
                     label=f"{cost_name}",
                     color=colors[cost_name],
                     linestyle=linestyles[cost_name])
    plt.ylabel(fr"$\|x^{{(1)}}(k)-x^{{(2)}}(k)\|_P$")
    plt.xlim(0, x_lim)
    if ylim_norm is not None:
        plt.ylim(ylim_norm[0], ylim_norm[1])
    plt.xticks(np.arange(0, x_lim+1, step=5))
    plt.grid(True, axis='both', which='major', linestyle='--')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(linestyles), fontsize=legend_fontsize, frameon=True)

    # --- u1 ---
    plt.subplot(3, 1, 2)
    if input_bound is not None:
        plt.fill_between(
            np.arange(x_lim + 1),
            input_bound[0],
            input_bound[1],
            color=grays[4],
            alpha=0.3
        )
    for cost_name in trajectories.keys():
        plt.plot(trajectories[cost_name]["Init1"]["u"][:, input_idx[0]],
                    color=colors[cost_name],
                    linestyle=linestyles[cost_name],
                    linewidth=1.0,
                    marker="o",
                    markersize=3,
                    markerfacecolor=colors[cost_name],
                    markeredgewidth=0
                )
    plt.ylabel(fr"$u_{input_idx[0]+1}$")
    plt.xlim(0, x_lim)
    if ylim_input is not None:
        plt.ylim(ylim_input[0][0], ylim_input[0][1])
    plt.xticks(np.arange(0, x_lim+1, step=5))
    plt.grid(True, axis='both', which='major', linestyle='--')

    # --- u2 ---
    plt.subplot(3, 1, 3)
    if input_bound is not None:
        plt.fill_between(
            np.arange(x_lim + 1),
            input_bound[0],
            input_bound[1],
            color=grays[4],
            alpha=0.3
        )
    for cost_name in trajectories.keys():
        plt.plot(trajectories[cost_name]["Init1"]["u"][:, input_idx[1]],
                    color=colors[cost_name],
                    linestyle=linestyles[cost_name],
                    linewidth=1.0,
                    marker="o",
                    markersize=3,
                    markerfacecolor=colors[cost_name],
                    markeredgewidth=0
                )
    plt.ylabel(fr"$u_{input_idx[1]+1}$")
    plt.xlim(0, x_lim)
    plt.xlabel("Time Step")
    if ylim_input is not None:
        plt.ylim(ylim_input[1][0], ylim_input[1][1])
    plt.xticks(np.arange(0, x_lim+1, step=5))
    plt.grid(True, axis='both', which='major', linestyle='--')

    plt.tight_layout()
    _save_figure(filename)


def plot_norm_and_state_for_paper(results, norms, nonlinear_costs, eta, x_lim, linestyles, colors, filename, state_bound=None, ylim_norm=None, ylim_state=None):

    plt.figure(figsize=(10, 8))

    # --- norm ---
    plt.subplot(3, 1, 1)
    max_initial_norm = 0.0
    for cost_name in nonlinear_costs:
        initial_norm = norms[cost_name][0]
        if initial_norm > max_initial_norm:
            max_initial_norm = initial_norm
    norm_baseline = np.array([max_initial_norm * (eta**k) for k in range(x_lim+1)])
    plt.semilogy(norm_baseline,
                color=grays[4],
                linestyle="-")
    for cost_name in nonlinear_costs:
        plt.semilogy(norms[cost_name],
                     label=f"{cost_name}",
                     color=colors[cost_name],
                     linestyle=linestyles[cost_name])
    plt.ylabel(fr"$\|x^{{(1)}}(k)-x^{{(2)}}(k)\|_P$")
    plt.xlim(0, x_lim)
    if ylim_norm is not None:
        plt.ylim(ylim_norm[0], ylim_norm[1])
    plt.xticks(np.arange(0, x_lim+1, step=2))
    plt.grid(True, axis='both', which='major', linestyle='--')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(linestyles), fontsize=legend_fontsize, frameon=True)

    # --- x1 ---
    plt.subplot(3, 1, 2)
    if state_bound is not None:
        plt.fill_between(
            np.arange(x_lim + 1),
            state_bound[0],
            state_bound[1],
            color=grays[4],
            alpha=0.3
        )
    for cost_name in nonlinear_costs:
        label = f"{cost_name}"
        plt.plot(
            results[cost_name]["Init1"]["x"][:, 0],
            label=label,
            color=colors[cost_name],
            linestyle=linestyles[cost_name],
            linewidth=1.0,
            marker="o",
            markersize=3,
            markerfacecolor=colors[cost_name],
            markeredgewidth=0
        )
    plt.ylabel(fr"$x_1$")
    plt.xlim(0, x_lim)
    if ylim_state is not None:
        plt.ylim(ylim_state[0][0], ylim_state[0][1])
    plt.xticks(np.arange(0, x_lim+1, step=2))
    plt.grid(True, axis='both', which='major', linestyle='--')

    # --- x3 ---
    plt.subplot(3, 1, 3)
    if state_bound is not None:
        plt.fill_between(
            np.arange(x_lim + 1),
            state_bound[0],
            state_bound[1],
            color=grays[4],
            alpha=0.3
        )
    for cost_name in nonlinear_costs:
        label = f"{cost_name}"
        plt.plot(
            results[cost_name]["Init1"]["x"][:, 2],
            label=label,
            color=colors[cost_name],
            linestyle=linestyles[cost_name],
            linewidth=1.0,
            marker="o",
            markersize=3,
            markerfacecolor=colors[cost_name],
            markeredgewidth=0
        )
    plt.ylabel(fr"$x_3$")
    plt.xlim(0, x_lim)
    if ylim_state is not None:
        plt.ylim(ylim_state[1][0], ylim_state[1][1])
    plt.xlabel("Time Step")
    plt.xticks(np.arange(0, x_lim+1, step=2))
    plt.grid(True, axis='both', which='major', linestyle='--')

    plt.tight_layout()
    _save_figure(filename)


def calculate_norm(results, P, nonlinear_costs):
    # calculate square root of (x[init1] - x[init2])^\top P (x[init1] - x[init2]) for each cost function
    norms = {}
    for cost_name in nonlinear_costs:
        diff = results[cost_name]["Init1"]["x"] - results[cost_name]["Init2"]["x"]
        norm = np.array([np.sqrt(diff[k] @ P @ diff[k]) for k in range(diff.shape[0])])
        norms[cost_name] = norm
    return norms


def plot_statistics(coeff_sparse_list, zero_counts_list, n_ic, filename):
    colors = plt.cm.tab10(np.linspace(0, 1, len(coeff_sparse_list)))

    plt.figure(figsize=(10, 5))

    for i, coeff in enumerate(coeff_sparse_list):
        zero_counts_all = zero_counts_list[i]
        zero_counts_mean = zero_counts_all.mean(axis=0)
        zero_counts_std = zero_counts_all.std(axis=0)

        # Plot mean and shaded std
        plt.plot(zero_counts_mean,
                color=colors[i],
                linewidth=1.0,
                label=rf"$\lambda={coeff}$",
                marker="o",
                markersize=3,
                markerfacecolor=colors[i],
                markeredgewidth=0   
        )
        plt.fill_between(
            np.arange(zero_counts_mean.size),
            zero_counts_mean - zero_counts_std,
            zero_counts_mean + zero_counts_std,
            color=colors[i],
            alpha=0.2
        )

    plt.xlabel("Time step")
    plt.ylabel(r"\#\{$i: |u_i|<10^{-6}$\}")
    plt.title(f"Average sparsity over {n_ic} initial conditions", fontsize=nominal_fontsize)
    plt.grid(True)
    plt.legend(fontsize=nominal_fontsize)
    plt.tight_layout()
    _save_figure(filename)
