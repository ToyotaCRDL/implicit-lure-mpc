import numpy as np
import cvxpy as cp
import os

from core.optimal_control_problem import OptimalControlProblem

class LMI_checker():
    def __init__(self, ocp: OptimalControlProblem, regularizer_type, simulation_name=None):
        self.ocp = ocp
        self.regularizer_type = regularizer_type
        if simulation_name is not None:
            self.simulation_name = simulation_name

    def convex_smooth_multiplier(self, horizon, n, m, P, Theta):
        # Case 1: convex smooth regularizers presented in Theorem 1

        lmbd = cp.Variable(horizon, nonneg=True)
        M_uigrad = np.block([
            [np.zeros((m, m)), np.eye(m)],
            [np.eye(m), -2*np.linalg.inv(Theta)]
        ])

        Coeff_xU2UGrad = np.block([
            [np.zeros((horizon*m, n)), np.eye(horizon*m)],
            [-self.ocp.C_mat, -self.ocp.D_mat]
        ])

        M_Uast = 0.0
        for i in range(horizon):
            Pi = self.ocp.selector[i]
            Coeff_UGrad2uigrad = np.block([[Pi, np.zeros_like(Pi)], [np.zeros_like(Pi), Pi]])

            M_Uast += lmbd[i] * Coeff_xU2UGrad.T @ Coeff_UGrad2uigrad.T @ M_uigrad @ Coeff_UGrad2uigrad @ Coeff_xU2UGrad

        return M_Uast, cp.trace(P)+cp.sum(lmbd) , lmbd

    def ccp_multiplier(self, horizon, n, m, P):
        # Case 2: convex closed proper regularizers presented in Theorem 2

        lmbd = cp.Variable(horizon, nonneg=True)
        M_uigrad = np.block([
                    [np.zeros((m, m)), np.eye(m)],
                    [np.eye(m), np.zeros((m, m))]
                ])

        Coeff_xU2UGrad = np.block([
                    [np.zeros((horizon*m, n)), np.eye(horizon*m)],
                    [-self.ocp.C_mat, -self.ocp.D_mat]
                ])

        M_Uast = 0.0
        for i in range(horizon):
            Pi = self.ocp.selector[i]
            Coeff_UGrad2uigrad = np.block([[Pi, np.zeros_like(Pi)], [np.zeros_like(Pi), Pi]])

            M_Uast += lmbd[i] * Coeff_xU2UGrad.T @ Coeff_UGrad2uigrad.T @ M_uigrad @ Coeff_UGrad2uigrad @ Coeff_xU2UGrad

        return M_Uast, cp.trace(P) + cp.sum(lmbd), lmbd

    def lipschitz_gradient_multiplier(self, horizon, n, m):
        # Case 3: Lipschitz-gradient regularizers presented in Theorem 3

        c = cp.Variable(nonneg=True)
        M_Lip = cp.bmat([
                    [np.eye(horizon*(n+m)), np.zeros((horizon*(n+m), horizon*m))],
                    [np.zeros((horizon*m, horizon*(n+m))), - c * np.eye(horizon*m)]
                ])

        Coeff_xU2XUGrad = np.block([
                    [self.ocp.Abar, self.ocp.Bbar],
                    [np.zeros((horizon*m, n)), np.eye(horizon*m)],
                    [-self.ocp.C_mat, -self.ocp.D_mat]
                ])

        return Coeff_xU2XUGrad.T @ M_Lip @ Coeff_xU2XUGrad, c, c

    def check_contractivity(self, horizon, rate, eps_P=1e-0, eps_lmi=1e-0, Theta = None):
        # Check the contractivity LMI using CVXPY

        n = self.ocp.n
        m = self.ocp.m
        if not hasattr(self.ocp, "Abar"):
            self.ocp.calc_bar_matrices()
        Pi1 = self.ocp.selector[0]

        A = self.ocp.A
        B = self.ocp.B

        P = cp.Variable((n, n), symmetric = True)

        nominal_term = cp.bmat([
            [A.T @ P @ A - rate**2 * P, A.T @ P @ B @ Pi1],
            [Pi1.T @ B.T @ P @ A,       Pi1.T @ B.T @ P @ B @ Pi1]
        ])

        multiplier =  0
        cost = 0

        if(self.regularizer_type == "convex_smooth"):
            if Theta is None:
                raise ValueError("Theta must be provided for convex smooth regularizer type")
            multiplier, cost, lmbd = self.convex_smooth_multiplier(horizon, n, m, P, Theta)
        elif(self.regularizer_type == "ccp"):
            multiplier, cost, lmbd = self.ccp_multiplier(horizon, n, m, P)
        elif(self.regularizer_type == "lipschitz_gradient"):
            multiplier, cost, c = self.lipschitz_gradient_multiplier(horizon, n, m)
        else:
            raise ValueError("Unknown regularizer type. Define multiplier in core/contractivity_LMI_checker.py")

        LHS = nominal_term + multiplier
        constraints = [
            P >> eps_P * np.eye(P.shape[0]),
            - LHS >> eps_lmi * np.eye(LHS.shape[0])
            ]

        # Solve SDP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(
            solver=cp.SCS,
            max_iters=20000000,   # maximum iterations
            eps=1e-4,             # tolerance
            acceleration_lookback=50,  # example of acceleration ON
            use_indirect=False,   # direct method (=False) / indirect method (=True)
            verbose=True
        )

        if problem.status == cp.OPTIMAL:
            print("Contractivity LMI is feasible")
            print("P: ", P.value)
            eigvals_P = np.linalg.eigvals(P.value)
            print("Eigenvalues of P:", eigvals_P)
            print("LHS: ", LHS.value)
            eigvals_lhs = np.linalg.eigvals(LHS.value)
            print("Eigenvalues of LHS:", eigvals_lhs)
            self.contractive = True

            if self.regularizer_type == "ccp" or self.regularizer_type == "convex_smooth":
                print("lambda: ", lmbd.value)
            if self.regularizer_type == "lipschitz_gradient":
                Lip = 1/np.sqrt(c.value)
                print("Maximum Lipschitz constant:", Lip)
            self.save_contractivity(P.value, rate, f"./output/{self.simulation_name}/LMI_solutions")

        else:
            print("Contractivity LMI is NOT feasible")
            self.contractive = False


    def save_contractivity(self, P, eta, path_dir):
        # create directory according to regularizer type
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        # save P matrix and eta value to txt file
        np.savetxt(path_dir + "/P.txt", P)
        with open(path_dir + "/eta.txt", "w") as f:
            f.write(str(eta))

        # save A, B, Q, R matrices to txt file
        np.savetxt(path_dir + "/A.txt", self.ocp.A)
        np.savetxt(path_dir + "/B.txt", self.ocp.B)
        np.savetxt(path_dir + "/Q.txt", self.ocp.Q)
        np.savetxt(path_dir + "/R.txt", self.ocp.R)
