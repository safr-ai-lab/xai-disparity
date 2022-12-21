import pdb

import numpy as np

class ConstrainedSolver:
    """
    Class defining constraints on the optimization problem
    :param expFunc: class that contains expressivities for each data point/feature
    :param alpha_s: minimum size desired for subgroup
    :param alpha_L: maximum size desired for subgroup
    :param B: weighted bound of the constraint penalty
    """
    def __init__(self, expFunc, alpha_s, alpha_L, B, nu):
        self.expFunc = expFunc
        self.alpha_s = alpha_s
        self.alpha_L = alpha_L
        self.B = B
        self.nu = nu

        # initialized variables
        self.v_t = 1000000
        self.thetas = [[0 ,0]]
        self.lambda_history = []
        self.g_history = []
        self.pred_history = []
        self.exp_history = []

    # return 1 if minimum size constraint is broken
    def phi_s(self, assigns):
        if self.alpha_s - (sum(assigns)/len(assigns)) <= 0:
            return 0
        else:
            return 1

    # return 1 if maximum size constraint is broken
    def phi_L(self, assigns):
        if (sum(assigns)/len(assigns)) - self.alpha_L <= 0:
            return 0
        else:
            return 1

    # using current theta values, update lambdas using exponential ratio
    def update_lambdas(self):
        theta = self.thetas[-1]
        lam0 = self.B * np.exp(theta[0]) / (1 + np.exp(theta[1]))
        lam1 = self.B * np.exp(theta[1]) / (1 + np.exp(theta[0]))
        self.lambda_history.append(np.array([lam0 ,lam1]))

    # using most recent classifier assignments, update thetas based on constraint violations
    def update_thetas(self, assigns):
        theta = self.thetas[-1]
        s_violation = self.nu*(self.alpha_s - np.mean(assigns))
        L_violation = self.nu*(np.mean(assigns) - self.alpha_L)
        self.thetas.append([theta[0]+s_violation, theta[1]+L_violation])

    # Solves the best lambda response of Auditor given mixture of classifiers
    def best_lambda(self, assigns):
        lam = [0 ,0]
        if self.phi_s(assigns) == 1:
            lam[0] = self.B
        if self.phi_L(assigns) == 1:
            lam[1] = self.B
        return lam

    # Solves best classifier response of Learner given avg lambdas
    def best_g(self, learner, feature_num, lams, minimize=True):
        costs0 = [0 for _ in range(len(learner.X))]
        if minimize:
            costs1 = [self.expFunc.exps[i][feature_num]-lams[0]+lams[1] for i in range(len(learner.X))]
        else:
            costs1 = [-self.expFunc.exps[i][feature_num]-lams[0]+lams[1] for i in range(len(learner.X))]

        l_response = learner.best_response(costs0, costs1)
        return l_response

    # Returns value of the Lagrangian
    def lagrangian(self, assigns, lams, feature_num, minimize):
        sign = 1
        if not minimize:
            sign = -1
        lambda_s = lams[0]
        lambda_L = lams[1]
        L = 0
        for i in range(len(assigns)):
            L += assigns[i] * sign * self.expFunc.get_exp(row=i, feature=feature_num)
        constraint_terms = self.phi_s(assigns) * lambda_s + self.phi_L(assigns) * lambda_L
        return L + constraint_terms

    def get_valid_model_i(self):
        valids = []
        for i in range(len(self.pred_history)):
            assigns = self.pred_history[i]
            if self.phi_s(assigns) + self.phi_L(assigns) == 0:
                valids.append(i)
        if len(valids) == 0:
            print('NOTHING VALID HERE!!!')
            valids.append(len(self.pred_history)-1)
        return valids

    @staticmethod
    def minimize_to_sign(minimize):
        if minimize:
            return 1
        else:
            return -1

    def get_best_valid_model(self, minimize):
        valids = self.get_valid_model_i()
        sign = self.minimize_to_sign(minimize)
        best_i = valids[0]
        best_exp = self.exp_history[best_i]
        for i in valids:
            if sign*self.exp_history[i] <= best_exp:
                best_i = i
        return self.g_history[best_i], self.pred_history[best_i], self.exp_history[best_i]
