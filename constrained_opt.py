from helpers.constrained_solver import ConstrainedSolver
from helpers.learner import Learner
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import json

class SeparableSolver:
    def __init__(self, df, target, sensitive_features, imp_method, df_name, t_split):
        self.df = df
        self.target = target
        self.sensitive_features = sensitive_features
        self.imp_method = imp_method
        self.df_name = df_name
        self.t_split = t_split

    def argmin_g(self, x, y, feature_num, f_sensitive, imp_func, minimize, alphas):
        imp_order = np.mean([abs(imp_func.imps[i][feature_num]) for i in range(len(x))])+.001
        size_reg = len(x)*np.mean(alphas)
        solver = ConstrainedSolver(imp_func, alpha_L=alphas[0], alpha_U=alphas[1], B=10000*imp_order, nu=.00001)
        v = .05*imp_order*size_reg
        print('imp order:',imp_order, v)

        x_sensitive = x[:,f_sensitive]
        costs0 = [0 for _ in range(len(x))] # costs0 is always zeros
        learner = Learner(x_sensitive, y, linear_model.LinearRegression())

        _ = 1
        start2 = time.time()
        while solver.v_t > v:
            solver.update_lambdas()

            # CSC solver, returns regoracle fit using costs0/costs1
            # h_t <- Best_h(lam_t)
            current_lam = solver.lambda_history[-1]
            if minimize:
                costs1 = [imp_func.imps[i][feature_num]-current_lam[0]+current_lam[1] for i in range(len(x))]
            else:
                costs1 = [-imp_func.imps[i][feature_num]-current_lam[0]+current_lam[1] for i in range(len(x))]
            l_response = learner.best_response(costs0, costs1)
            solver.g_history.append(l_response)

            assigns, cost = l_response.predict(x_sensitive)
            importance = imp_func.get_total_imp(assigns, feature_num)
            solver.pred_history.append(np.array(assigns))
            solver.size_history.append(np.mean(assigns))
            solver.imp_history.append(importance)

            if _%10 == 0:
                solver.update_vt(learner, x_sensitive, feature_num, minimize)

            if _%500==0:
                print("ITERATION NUMBER ", _, "time:", time.time()-start2)
                print(np.mean(assigns))

            if _%5000 == 0:
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

    def split_out_dataset(self, dataset, target_column):
        x = dataset.drop(target_column, axis=1).to_numpy()
        y = dataset[target_column].to_numpy()
        #sensitive_ds = dataset[f_sensitive].to_numpy()
        return x, y


    def extremize_imps_dataset(self, dataset, imp_func, target_column, f_sensitive, alphas, seed=0, t_split=.5):
        np.random.seed(seed)
        """
        :param dataset: pandas dataframe
        :param imp_func: class for impressivities
        :param target_column: string, column name in dataset
        :param f_sensitive: list of column names that are sensitive features
        :param seed: int, random seed
        :return: total importance over these rows
        """
        train_df, test_df = train_test_split(dataset, test_size=t_split, random_state=seed)
        x_train, y_train = self.split_out_dataset(train_df, target_column)
        x_test, y_test = self.split_out_dataset(test_df, target_column)
        classifier = RandomForestClassifier(random_state=seed)
        classifier.fit(x_train, y_train)
        out_df = pd.DataFrame()

        imp_func_train = imp_func(classifier, x_train, y_train, seed)
        #print("Populating train importance values")
        #imp_func_train.populate_imps()

        imp_func_test = imp_func(classifier, x_test, y_test, seed)
        #print("Populating test importance values")
        #imp_func_test.populate_imps()

        # Populating with pre-computed values
        with open(f'input/imps/{self.df_name}_train_{self.imp_method}_seed0', 'r') as f:
            train_temp = list(map(json.loads, f))[0]
        for e_list in train_temp:
            imp_func_train.imps.append({int(k):v for k,v in e_list.items()})
        with open(f'input/imps/{self.df_name}_test_{self.imp_method}_seed0', 'r') as f:
            test_temp = list(map(json.loads, f))[0]
        for e_list in test_temp:
            imp_func_test.imps.append({int(k):v for k,v in e_list.items()})

        #for feature_num in range(len(x_train[0])):
        for feature_num in range(1):
            print('*****************')
            print(train_df.columns[feature_num])
            total_imp_train = self.full_dataset_importance(imp_func_train, feature_num)
            print('total imp: ', total_imp_train)
            min_solver = self.argmin_g(x_train, y_train, feature_num, f_sensitive, imp_func_train,
                                                       minimize=True, alphas=alphas)
            min_model, min_assigns, min_imp = min_solver.get_best_valid_model(minimize=True)
            print('min imp', min_imp, '| size', np.mean(min_assigns))

            max_solver = self.argmin_g(x_train, y_train, feature_num, f_sensitive, imp_func_train,
                                                       minimize=False, alphas=alphas)
            max_model, max_assigns, max_imp = max_solver.get_best_valid_model(minimize=False)
            print('max imp', max_imp, '| size', np.mean(max_assigns))

            # Choose max difference
            if abs(max_imp-total_imp_train) > abs(min_imp-total_imp_train):
                furthest_imp_train = max_imp
                assigns_train = max_assigns
                best_model = max_model
                best_solver = max_solver
                direction = 'maximize'
            else:
                furthest_imp_train = min_imp
                assigns_train = min_assigns
                best_model = min_model
                best_solver = min_solver
                direction = 'minimize'
            subgroup_size_train = np.mean(assigns_train)

            # compute test values
            total_imp_test = self.full_dataset_importance(imp_func_test, feature_num)
            #assigns_test = get_avg_prediction(best_model, x_test) # mix model method
            assigns_test = best_model.predict(x_test[:,f_sensitive])[0] # sensitive features only method
            #assigns_test = best_model.predict(x_test)[0]
            subgroup_size_test = np.mean(assigns_test)
            furthest_imp_test = 0
            for i in range(len(assigns_test)):
                furthest_imp_test += assigns_test[i]*imp_func_test.imps[i][feature_num]

            # # from mix models, pick model with largest imp diff that is valid
            params = best_model.b1.coef_
            params_with_labels = {dataset.columns[i]: float(param) for (i, param) in zip(f_sensitive, params)}
            params_with_labels['Intercept'] = best_model.b1.intercept_
            print(params_with_labels)
            out_df = pd.concat([out_df,
                                pd.DataFrame.from_records([{'Feature': dataset.columns[feature_num],
                                                            'Alpha': alphas,
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

