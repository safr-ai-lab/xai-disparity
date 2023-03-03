import numpy as np

class ConstrainedSolver:
    """
    Class defining constraints on the optimization problem
    :param expFunc: class that contains expressivities for each data point/feature
    :param alpha_L: minimum size desired for subgroup
    :param alpha_U: maximum size desired for subgroup
    :param B: weighted bound of the constraint penalty
    """
    def __init__(self, expFunc, alpha_L, alpha_U, B, nu):
        self.expFunc = expFunc
        self.alpha_L = alpha_L
        self.alpha_U = alpha_U
        self.B = B
        self.nu = nu

        # initialized variables
        self.v_t = 1000000
        self.thetas = [[0 ,0]]
        self.lambda_history = []
        self.g_history = []
        self.pred_history = []
        self.exp_history = []
        # temporary
        self.avg_pred_size = []
        self.avg_lambda = []
        self.besth_avg_lambda = []
        self.L_ceilings = []
        self.L_floors = []
        self.Ls = []
        self.size_history = []
        self.vt_history = []
        self.iters = []

    # returns lower size violation. Want this value <= 0
    def phi_L(self, assigns):
        diff = self.alpha_L - np.mean(assigns)
        if diff <= 0:
            return 0
        else:
            return diff
        #return self.alpha_L - np.mean(assigns)
        # if self.alpha_L - np.mean(assigns) <= 0:
        #     return 0
        # else:
        #     return 1

    # returns upper size violation. Want this value <= 0
    def phi_U(self, assigns):
        diff = np.mean(assigns) - self.alpha_U
        if diff <= 0:
            return 0
        else:
            return diff
        #return np.mean(assigns) - self.alpha_U
        # if np.mean(assigns) - self.alpha_U <= 0:
        #     return 0
        # else:
        #     return 1

    # using current theta values, update lambdas using exponential ratio
    def update_lambdas(self):
        theta = self.thetas[-1]
        lam0 = self.B * np.exp(theta[0]) / (1 + np.exp(theta[1]))
        lam1 = self.B * np.exp(theta[1]) / (1 + np.exp(theta[0]))
        self.lambda_history.append([lam0 ,lam1])

    # using most recent classifier assignments, update thetas based on constraint violations
    def update_thetas(self, assigns):
        theta = self.thetas[-1]
        L_violation = self.nu * self.phi_L(assigns)
        U_violation = self.nu * self.phi_U(assigns)
        # L_violation = self.nu*(self.alpha_L - np.mean(assigns))
        # U_violation = self.nu*(np.mean(assigns) - self.alpha_U)
        self.thetas.append([theta[0]+L_violation, theta[1]+U_violation])

    # Solves the best lambda response of Auditor given mixture of classifiers
    def best_lambda(self, assigns):
        lam = [0, 0]
        if self.phi_L(assigns) > 0:
            lam[0] = self.B
        if self.phi_U(assigns) > 0:
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
        lambda_L = lams[0]
        lambda_U = lams[1]
        L = 0
        for i in range(len(assigns)):
            L += assigns[i] * sign * self.expFunc.get_exp(row=i, feature=feature_num)
        constraint_terms = self.phi_L(assigns) * lambda_L + self.phi_U(assigns) * lambda_U
        return L + constraint_terms

    def update_vt(self, learner, x_s, feature_num, minimize):
        # compute avg(Q), Best_lam, and L ceiling
        avg_pred = [np.mean(k) for k in zip(*self.pred_history)]
        self.avg_pred_size.append(np.mean(avg_pred))
        best_lam = self.best_lambda(avg_pred)
        L_ceiling = self.lagrangian(avg_pred, best_lam, feature_num, minimize)
        self.L_ceilings.append(L_ceiling)

        # compute avg(lam),
        avg_lam = [np.mean(k) for k in zip(*self.lambda_history)]
        self.avg_lambda.append(avg_lam)
        costs0 = [0 for _ in range(len(x_s))]
        if minimize:
            costs1 = [self.expFunc.exps[i][feature_num] - avg_lam[0] + avg_lam[1] for i in range(len(x_s))]
        else:
            costs1 = [-self.expFunc.exps[i][feature_num] - avg_lam[0] + avg_lam[1] for i in range(len(x_s))]
        best_h = learner.best_response(costs0, costs1)
        best_h_preds = best_h.predict(x_s)[0]
        self.besth_avg_lambda.append(np.mean(best_h_preds))
        L_floor = self.lagrangian(best_h_preds, avg_lam, feature_num, minimize)
        self.L_floors.append(L_floor)

        L = self.lagrangian(avg_pred, avg_lam, feature_num, minimize)
        self.Ls.append(L)
        self.v_t = max(abs(L - L_floor), abs(L_ceiling - L))
        self.vt_history.append(self.v_t)

    def get_valid_model_i(self):
        valids = []
        for i in range(len(self.pred_history)):
            assigns = self.pred_history[i]
            if (self.phi_L(assigns) <= 0) and (self.phi_U(assigns) <= 0):
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
