# Implicit Lur'e MPC

Numerical case studies of the contractivity of regularized MPC via implicit Lur'e analysis.
The paper is posted on [ArXiv](https://arxiv.org/abs/2607.00383v1).

## Development Environment
Python 3.13.13

## Required Libraries

- scipy
- cvxpy
- matplotlib


# Codes

## MPC of Series Mass-Spring-Damper System with Soft Penalty on Inputs

This code simulates a series mass-spring-damper system controlled by an MPC with input penalty.
Run [input_penalized_mpc_setting.py](input_penalized_mpc_setting.py) to solve an SDP that certifies contractivity of the MPC closed loop.
The solution is saved in [output/input_penalized_mpc/LMI_solutions](output/input_penalized_mpc/LMI_solutions).

To simulate the closed-loop system, run [input_penalized_mpc.py](input_penalized_mpc.py) after solving the SDP.
This script generates [output/input_penalized_mpc/figures/trajectories_for_paper.pdf](output/input_penalized_mpc/figures/trajectories_for_paper.pdf), which corresponds to Fig. 2 in the paper.

To quantify the effect of the input penalty, [input_penalized_mpc_effectiveness.py](input_penalized_mpc_effectiveness.py) measures the cumulative constraint violation:

$$
\sum_{t=0}^{39} \sum_{i=1}^{2} \max \lbrace 0, u_i(t)-\bar{u}, \underline{u}-u_i(t) \rbrace
$$

For each of 50 initial states sampled uniformly from $[-3.0, 3.0]^4$, the code reports the cumulative violations of the regularized MPC and the nominal MPC.
If the cumulative violation of regularized MPC is smaller than that of nominal MPC for all sampled initial states, it reports:

`As expected, penalty for regularized MPC is smaller than nominal MPC.`

Otherwise, it reports:

`>>> Unexpected <<< The penalty for regularized MPC is larger than for nominal MPC.`

and stops checking further initial states.


## Tracking MPC for Series Mass-Spring-Damper System with Hard Constraints on Inputs

This code simulates tracking control of a series mass-spring-damper system by an tracking MPC with input constraints.
Run [input_constrained_tracking_mpc_setting.py](input_constrained_tracking_mpc_setting.py) to solve an SDP that certifies contractivity of the MPC closed loop.
The solution is saved in [output/input_constrained_tracking_mpc/LMI_solutions](output/input_constrained_tracking_mpc/LMI_solutions).

To simulate the closed-loop system, run [input_constrained_tracking_mpc.py](input_constrained_tracking_mpc.py) after solving the SDP.
This script generates [output/input_constrained_tracking_mpc/figures/trajectories_for_paper.pdf](output/input_constrained_tracking_mpc/figures/trajectories_for_paper.pdf), which corresponds to Fig. 3 in the paper.


## Sparse Control of Consensus Networks

This code simulates sparse control of consensus network system by an $\ell_1$-regularized MPC.
Run [sparse_mpc_setting.py](sparse_mpc_setting.py) to solve an SDP that certifies contractivity of the MPC closed loop.
The solution is saved in [output/sparse_mpc/LMI_solutions](output/sparse_mpc/LMI_solutions).

To simulate the closed-loop system, run [sparse_mpc.py](sparse_mpc.py) after solving the SDP.
This script generates [output/sparse_mpc/figures/trajectories_for_paper.pdf](output/sparse_mpc/figures/trajectories_for_paper.pdf), which corresponds to Fig. 5 in the paper.

To see the effect of the sparsity-enhancing regularizer, [sparse_mpc_effectiveness.py](sparse_mpc_effectiveness.py) plots [output/sparse_mpc/figures/sparsity_statistics_50_inits.pdf](output/sparse_mpc/figures/sparsity_statistics_50_inits.pdf), which corresponds to Fig. 6 in the paper.


## MPC of Series Mass-Spring-Damper System with Soft Penalty on States

This code simulates a series mass-spring-damper system controlled by an MPC with state penalty.
Run [state_penalized_mpc_setting.py](state_penalized_mpc_setting.py) to solve an SDP that certifies contractivity of the MPC closed loop.
The solution is saved in [output/state_penalized_mpc/LMI_solutions](output/state_penalized_mpc/LMI_solutions).

To simulate the closed-loop system, run [state_penalized_mpc.py](state_penalized_mpc.py) after solving the SDP.
This script generates [output/state_penalized_mpc/figures/trajectories_for_paper.pdf](output/state_penalized_mpc/figures/trajectories_for_paper.pdf), which corresponds to Fig. 7 in the paper.
