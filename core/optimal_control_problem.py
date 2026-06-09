import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize


class OptimalControlProblem:
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.n, self.m = self.B.shape

    def set_nominal_ocp(self, Q, R, horizon):
        self.Q = Q
        self.R = R
        self.Qf = solve_discrete_are(self.A, self.B, Q, R)

        self.horizon = horizon
        self.make_selector()
        self.calc_bar_matrices()

    def make_selector(self):
        """Return list of selector matrix Pi_U2ui that extracts the i-th input block."""
        n = self.n
        m = self.m
        self.selector = []
        for i in range(self.horizon):
            Pi = np.zeros((m, self.horizon * m))
            Pi[:, i*m:(i+1)*m] = np.eye(m)
            self.selector.append(Pi)

    def calc_bar_matrices(self):
        A_powers = [np.linalg.matrix_power(self.A, i) for i in range(1, self.horizon + 1)]
        self.Abar = np.vstack(A_powers)
    
        self.Bbar = np.zeros((self.horizon * self.n, self.horizon * self.m))
        for row in range(self.horizon):
            for col in range(row + 1):
                A_power = np.linalg.matrix_power(self.A, row - col)
                block = A_power @ self.B
                self.Bbar[row*self.n : (row+1)*self.n, col*self.m : (col+1)*self.m] = block

        self.Qbar = np.kron(np.eye(self.horizon), self.Q)
        self.Qbar[-self.n:, -self.n:] = self.Qf
        self.Rbar = np.kron(np.eye(self.horizon), self.R)
        self.C_mat = self.Bbar.T @ self.Qbar @ self.Abar
        self.D_mat = self.Rbar + self.Bbar.T @ self.Qbar @ self.Bbar

    def baseline_contraction_rate(self):
        Pi1 = self.selector[0]
        self.K = - Pi1 @ np.linalg.inv(self.D_mat) @ (self.C_mat)
        eig_Acl, eigvec_Acl = np.linalg.eig(self.A + self.B @ self.K)
        eig_A, eigvec_A = np.linalg.eig(self.A)
        print('Spectral Radius of Open Loop:\n  ', np.max(np.abs(eig_A)))
        print('  eigs:', eig_A)
        print('Spectral Radius of Closed Loop by Nominal MPC:\n  ', np.max(np.abs(eig_Acl)))
        print('  eigs:', eig_Acl)

    def next_state(self, x, u):
        x_next = self.A @ x + self.B @ u
        return x_next

    # Cost functions
    def calculate_stage_cost(self, x, u, x_ref, u_ref):
        cost = 0.5 * (x - x_ref) @ self.Q @ (x - x_ref) + 0.5 * (u - u_ref) @ self.R @ (u - u_ref)
        return cost

    def calculate_terminal_cost(self, x, x_ref):
        cost = 0.5 * (x - x_ref) @ self.Qf @ (x - x_ref)
        return cost


    # ========= Model Predictive Control =============

    def cost_calculator_vectorized(self, U, x0, Xd, Ud, stage_regularizer = lambda x,u,k,xd,ud:0.0, terminal_regularizer = lambda x,k,xd:0.0):
        if stage_regularizer is None:
            stage_regularizer = lambda x,u,k,xd,ud:0.0
        if terminal_regularizer is None:
            terminal_regularizer = lambda x,k,xd:0.0

        horizon = self.horizon
        X = self.Abar @ x0 + self.Bbar @ U
        Ud_1d = Ud.transpose().reshape(-1)

        cost = 0.5 * (U - Ud_1d) @ self.D_mat @ (U - Ud_1d) + (U - Ud_1d) @ self.C_mat @ (x0 - Xd[:,0])
        for h in range(horizon):
            u_h = U[h*self.m:(h+1)*self.m]
            x_h = X[h*self.n:(h+1)*self.n]
            cost += stage_regularizer(x_h, u_h, h, Xd[:,h], Ud[:,h])
        x_H = X[-self.n:]
        cost += terminal_regularizer(x_H, horizon, Xd[:,horizon])
        return cost


    def run_simulation(self, x0, T, stage_regularizer = lambda x,u,k,xd,ud:0.0, terminal_regularizer = lambda x,k,xd:0.0, constraints=None):
        H = self.horizon
        x_hist = [x0.copy()]
        u_hist = []
        x_curr = x0.copy()
        for t in range(T):
            u0 = np.zeros(H * self.m)
            xd_mat = self.X_ref[:, t:t+H+1]
            ud_mat = self.U_ref[:, t:t+H]
            if constraints is None:
                res = minimize(self.cost_calculator_vectorized,
                            u0,
                            args = (x_curr, xd_mat, ud_mat, stage_regularizer, terminal_regularizer),
                            method='SLSQP',
                            )
            else:
                res = minimize(self.cost_calculator_vectorized,
                            u0,
                            args = (x_curr, xd_mat, ud_mat, stage_regularizer, terminal_regularizer),
                            method='SLSQP',
                            constraints=constraints
                            )

            u_seq = res.x.reshape(H, self.m)
            u_apply = u_seq[0]

            x_next = self.next_state(x_curr, u_apply)
            x_curr = x_next

            x_hist.append(x_curr.copy())
            u_hist.append(u_apply)

        return np.array(x_hist), np.array(u_hist)


    # ========= Sparse Model Predictive Control =============

    def solve_ocp_l1(self, x0, lambda_stage, Xd=None, Ud=None):
        """
        Solve optimal control problem with L1 stage regularization using CVXPY.
        Only supports sparsity/L1 penalties on the control input.
        """
        H = self.horizon
        n, m = self.n, self.m

        # Decision variable
        U = cp.Variable(H * m)

        # Predicted trajectory
        X = self.Abar @ x0 + self.Bbar @ U

        # Reference trajectory
        if Ud is None:
            Ud = np.zeros((self.m, H))
        if Xd is None:
            Xd = np.zeros((self.n, H+1))

        # Quadratic cost (same as your existing D_mat and C_mat)
        Ud_1d = Ud.transpose().reshape(-1)
        cost = 0.5 * cp.quad_form(U - Ud_1d, self.D_mat) + (U - Ud_1d) @ self.C_mat @ (x0 - Xd[:, 0])

        # L1 stage regularization
        for h in range(H):
            u_h = U[h*m:(h+1)*m]
            cost += lambda_stage * cp.norm1(u_h)

        # Build problem and solve
        ocp = cp.Problem(cp.Minimize(cost))
        ocp.solve(solver=cp.OSQP, warm_start=True)

        return U.value.reshape(H, m)


    def run_simulation_sparse(self, x0, T, lambda_stage=0.0):
        H = self.horizon
        x_hist = [x0.copy()]
        u_hist = []
        x_curr = x0.copy()

        for t in range(T):
            xd_mat = self.X_ref[:, t:t+H+1]
            ud_mat = self.U_ref[:, t:t+H]
            u_seq = self.solve_ocp_l1(x_curr, lambda_stage, xd_mat, ud_mat)
            u_apply = u_seq[0]

            x_curr = self.next_state(x_curr, u_apply)

            x_hist.append(x_curr.copy())
            u_hist.append(u_apply)

        return np.array(x_hist), np.array(u_hist)


    def run_simulation_without_control(self, x0, T, u_ff=None):
        if u_ff is None:
            u_ff = [np.zeros(self.m)]*T
        x_hist = [x0.copy()]
        u_hist = []
        x_curr = x0.copy()

        for t in range(T):
            u_apply = u_ff[t]

            x_curr = self.A @ x_curr + self.B @ u_apply

            x_hist.append(x_curr.copy())
            u_hist.append(u_apply)

        return np.array(x_hist), np.array(u_hist)


    def calculate_periodic_desired_trajectory(self, pos_des_periodic, prd):
        # pos_des_periodic: a flat vector of size 2*H
        # prd: period
        # solve linear equation system of Cx=d

        m = self.m
        n = self.n

        select_pos = np.array([[1,0,0,0],[0,0,1,0]])

        C = np.zeros([n*prd+m*prd, n*prd+m*prd])
        for h in range(prd-1):
            C[n*h:n*(h+1), n*h:n*(h+1)] = self.A
            C[n*h:n*(h+1), n*(h+1):n*(h+2)] = - np.eye(n)
            C[n*h:n*(h+1), n*prd+m*h:n*prd+m*(h+1)] = self.B
        C[n*(prd-1):n*prd, n*(prd-1):n*prd] = self.A
        C[n*(prd-1):n*prd, :n] = - np.eye(n)
        C[n*(prd-1):n*prd, n*prd+m*(prd-1):] = self.B

        for h in range(prd):
            C[n*prd+m*h:n*prd+m*(h+1), n*h:n*(h+1)] = select_pos

        d = np.concatenate([np.zeros(n*prd), pos_des_periodic])

        solution = np.linalg.solve(C, d)
        X_periodic = solution[:n*prd].reshape(prd, n).transpose()  # [x^1, x^2, ..., x^prd]
        U_periodic = solution[n*prd:].reshape(prd, m).transpose()  # [u^1, u^2, ..., u^prd]
        self.X_periodic = X_periodic
        self.U_periodic = U_periodic
        self.period = prd


    def set_references(self, simstep, X_ref=None, U_ref=None):
        # Reference trajectory
        if X_ref is None:
            self.X_ref = np.zeros((self.n, simstep + self.horizon + 1))
        else:
            self.X_ref = X_ref
        if U_ref is None:
            self.U_ref = np.zeros((self.m, simstep + self.horizon))
        else:
            self.U_ref = U_ref


    def get_reference_from_periodic(self, simstep):
        # X_ref: [x^ref_1, x^ref_2, ..., x^ref_(H+1)]
        # U_ref: [u^ref_1, u^ref_2, ..., u^ref_H]
        # Note: (x_ref, u_ref) is required to satisfy state space equation (realizable trajectory)

        max_step = simstep + self.horizon
        X_ref = np.zeros((self.n, max_step + 1))
        U_ref = np.zeros((self.m, max_step))

        for k in range(max_step):
            h = k % self.period
            X_ref[:,k] = self.X_periodic[:,h]
            U_ref[:,k] = self.U_periodic[:,h]
        h = max_step % self.period
        X_ref[:,max_step] = self.X_periodic[:,h]
        return X_ref, U_ref
