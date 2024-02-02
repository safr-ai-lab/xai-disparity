from helpers.constrained_solver import ConstrainedSolver
from helpers.learner import Learner
from sklearn import linear_model
import pandas as pd
import numpy as np
import time

class SeparableSolver:
    def __init__(self, df, trains, tests, imp_func_train, imp_func_test, f_sensitive, alphas, seed):
        self.df = df
        self.x_train, self.y_train = trains[0], trains[1]
        self.x_test, self.y_test = tests[0], tests[1]
        self.imp_func_train = imp_func_train
        self.imp_func_test = imp_func_test
        self.f_sensitive = f_sensitive
        self.alphas = alphas
        np.random.seed(seed)

    # Executes iterative CSC solve for a given feature
    def argmin_g(self, feature_num, nu, c_B, c_v, minimize):
        imp_order = np.mean([abs(self.imp_func_train.imps[i][feature_num]) for i in range(len(self.x_train))])+.001
        size_reg = len(self.x_train)*np.mean(self.alphas)
        solver = ConstrainedSolver(self.imp_func_train, alpha_L=self.alphas[0], alpha_U=self.alphas[1],
                                   B=c_B*imp_order, nu=nu)
        v = c_v*imp_order*size_reg
        print('imp order:', imp_order, v)

        x_sensitive = self.x_train[:, self.f_sensitive]
        costs0 = [0 for _ in range(len(self.x_train))] # costs0 is always zeros
        learner = Learner(x_sensitive, self.y_train, linear_model.LinearRegression())

        _ = 1
        start2 = time.time()
        while solver.v_t > v:
            solver.update_lambdas()

            # CSC solver, returns regoracle fit using costs0/costs1
            # h_t <- Best_h(lam_t)
            current_lam = solver.lambda_history[-1]
            if minimize:
                costs1 = [self.imp_func_train.imps[i][feature_num]-current_lam[0]+current_lam[1] for i in range(len(self.x_train))]
            else:
                costs1 = [-self.imp_func_train.imps[i][feature_num]-current_lam[0]+current_lam[1] for i in range(len(self.x_train))]
            l_response = learner.best_response(costs0, costs1)
            solver.g_history.append(l_response)

            assigns, cost = l_response.predict(x_sensitive)
            importance = self.imp_func_train.get_total_imp(assigns, feature_num)
            solver.pred_history.append(assigns)
            solver.size_history.append(np.mean(assigns))
            solver.imp_history.append(importance)

            if _%10 == 0:
                solver.update_vt(learner, x_sensitive, feature_num, minimize)

            if _%500==0:
                print("ITERATION NUMBER ", _, "time:", time.time()-start2)
                print(np.mean(assigns))

            if _%100 == 0:
                print('Max iterations reached')
                solver.v_t = 0
            solver.update_thetas(assigns)
            _ += 1
        solver.iters = _

        print('num iterations: ', _-1)
        return solver


    # Given distribution of models, compute predictions on x and return average
    def get_avg_prediction(self, mix_models, x):
        predictions = [m.predict(x)[0] for m in mix_models]
        avg_pred = [np.mean(k) for k in zip(*predictions)]
        return avg_pred

    def full_dataset_importance(self, imp_func, feature_num):
        total = 0
        for row in imp_func.imps:
            total += row[feature_num]
        return total

    # Iterate through all features, find min/max subgroups, choose largest FID subgroup
    def extremize_imps_dataset(self, nu, c_B, c_v):
        out_df = pd.DataFrame()
        for feature_num in range(len(self.x_train[0])):
            print('*****************')
            print(self.df.columns[feature_num])
            total_imp_train = self.full_dataset_importance(self.imp_func_train, feature_num)
            print('total imp: ', total_imp_train)

            min_solver = self.argmin_g(feature_num, nu, c_B, c_v, minimize=True)
            min_model, min_assigns, min_imp = min_solver.get_best_valid_model(minimize=True)
            print('min imp', min_imp, '| size', np.mean(min_assigns))

            max_solver = self.argmin_g(feature_num, nu, c_B, c_v, minimize=False)
            max_model, max_assigns, max_imp = max_solver.get_best_valid_model(minimize=False)
            print('max imp', max_imp, '| size', np.mean(max_assigns))

            # Choose max difference
            if abs(max_imp-total_imp_train) > abs(min_imp-total_imp_train):
                furthest_imp_train = max_imp
                assigns_train = max_assigns
                best_model = max_model
                best_solver = max_solver
                direction = 'maximize'

                assign_history = max_solver.pred_history
            else:
                furthest_imp_train = min_imp
                assigns_train = min_assigns
                best_model = min_model
                best_solver = min_solver
                direction = 'minimize'

                assign_history = min_solver.pred_history
            subgroup_size_train = np.mean(assigns_train)

            # compute test values
            total_imp_test = self.full_dataset_importance(self.imp_func_test, feature_num)
            #assigns_test = get_avg_prediction(best_model, x_test) # mix model method
            assigns_test = best_model.predict(self.x_test[:, self.f_sensitive])[0] # sensitive features only method
            #assigns_test = best_model.predict(x_test)[0]
            subgroup_size_test = np.mean(assigns_test)
            furthest_imp_test = 0
            for i in range(len(assigns_test)):
                furthest_imp_test += assigns_test[i]*self.imp_func_test.imps[i][feature_num]

            # TODO: temporary
            assign_history.append(assigns_test)

            # # from mix models, pick model with largest imp diff that is valid
            params = best_model.b1.coef_
            params_with_labels = {self.df.columns[i]: float(param) for (i, param) in zip(self.f_sensitive, params)}
            params_with_labels['Intercept'] = best_model.b1.intercept_
            print(params_with_labels)
            out_df = pd.concat([out_df,
                                pd.DataFrame.from_records([{'Feature': self.df.columns[feature_num],
                                                            'Alpha': self.alphas,
                                                            'F(D)': total_imp_test,
                                                            'max(F(S))': furthest_imp_test,
                                                            'Difference': abs(furthest_imp_test - total_imp_test),
                                                            'Percent Change': 100*abs(furthest_imp_test - total_imp_test)/
                                                                              (abs(total_imp_test)+.000001),
                                                            'avg(F(D))': total_imp_test/len(assigns_test),
                                                            'avg(F(S))': furthest_imp_test/(sum(assigns_test)+.000001),
                                                            'avg diff': abs(total_imp_test/len(assigns_test) -
                                                                            furthest_imp_test/(sum(assigns_test)+.000001)),
                                                            'Subgroup Coefficients': params_with_labels,
                                                            'Subgroup Size': subgroup_size_test,
                                                            'Direction': direction,
                                                            'F(D)_train': total_imp_train,
                                                            'max(F(S))_train': furthest_imp_train,
                                                            'Difference_train': abs(furthest_imp_train-total_imp_train),
                                                            'Percent Change_train': 100*abs(furthest_imp_train-total_imp_train)/
                                                                                    (abs(total_imp_train) + .000001),
                                                            'Subgroup Size_train': subgroup_size_train,
                                                            'assigns history': assign_history,
                                                            'size history': best_solver.size_history,
                                                            'lambda history': best_solver.lambda_history,
                                                            'imp history': best_solver.imp_history,
                                                            'avg pred size': best_solver.avg_pred_size,
                                                            'avg lambda': best_solver.avg_lambda,
                                                            'besth_avg_lambda': best_solver.besth_avg_lambda,
                                                            'L': best_solver.Ls,
                                                            'L_ceiling': best_solver.L_ceilings,
                                                            'L_floor': best_solver.L_floors,
                                                            'vt_history': best_solver.vt_history,
                                                            'iters': best_solver.iters
                                                            }])])
        return out_df

