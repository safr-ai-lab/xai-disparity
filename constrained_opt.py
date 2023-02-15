from lime_exp_func import LimeExpFunc
from shap_exp_func import ShapExpFunc
from grad_exp_func import GradExpFunc
from constrained_solver import ConstrainedSolver
from learner import Learner
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from aif360.datasets import CompasDataset, BankDataset
import time
import argparse
from datetime import datetime
import json
import sys


parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('exp_method', type=str)
parser.add_argument('--dummy', action='store_true')
args = parser.parse_args()
exp_method = args.exp_method
dummy = args.dummy

if exp_method == 'lime':
    expFunc = LimeExpFunc
elif exp_method == 'shap':
    expFunc = ShapExpFunc
elif exp_method == 'grad':
    expFunc = GradExpFunc
else:
    sys.exit('Exp method not recognized')


def argmin_g(x, y, feature_num, f_sensitive, exp_func, minimize, alphas):
    exp_order = np.mean([abs(exp_func.exps[i][feature_num]) for i in range(len(x))])
    solver = ConstrainedSolver(exp_func, alpha_L=alphas[0], alpha_U=alphas[1], B=10000*exp_order, nu=.0001)
    v = .05*exp_order*len(x)
    print('exp order:',exp_order, v)

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
            costs1 = [exp_func.exps[i][feature_num]-current_lam[0]+current_lam[1] for i in range(len(x))]
        else:
            costs1 = [-exp_func.exps[i][feature_num]-current_lam[0]+current_lam[1] for i in range(len(x))]
        l_response = learner.best_response(costs0, costs1)
        solver.g_history.append(l_response)

        assigns, cost = l_response.predict(x_sensitive)
        expressivity = exp_func.get_total_exp(assigns, feature_num)
        solver.pred_history.append(np.array(assigns))
        solver.size_history.append(np.mean(assigns))
        solver.exp_history.append(expressivity)

        if _%10 == 0:
            solver.update_vt(learner, x_sensitive, feature_num, minimize)

        if _%500==0:
            print("ITERATION NUMBER ", _, "time:", time.time()-start2)
            print(np.mean(assigns))

        # if (solver.phi_L(assigns) <= 0) and (solver.phi_U(assigns) <= 0):
        #     solver.v_t = 0
        if _%1000 == 0:
            print('Max iterations reached')
            solver.v_t = 0
        solver.update_thetas(assigns)
        _ += 1
    solver.iters = _
    ### method 2: Returning best valid model
    print('num iterations: ', _-1)
    return solver
    #best_model, best_assigns, best_exp = solver.get_best_valid_model(minimize)
    #return best_model, best_assigns, best_exp

# Given distribution of models, compute predictions on x and return average
def get_avg_prediction(mix_models, x):
    predictions = [m.predict(x)[0] for m in mix_models]
    avg_pred = [np.mean(k) for k in zip(*predictions)]
    return avg_pred

def full_dataset_expressivity(exp_func, feature_num):
    total = 0
    for row in exp_func.exps:
        total += row[feature_num]
    return total

def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    #sensitive_ds = dataset[f_sensitive].to_numpy()
    return x, y


def extremize_exps_dataset(dataset, exp_func, target_column, f_sensitive, alphas, seed=0, t_split=.5):
    np.random.seed(seed)
    """
    :param dataset: pandas dataframe
    :param exp_func: class for expressivities
    :param target_column: string, column name in dataset
    :param f_sensitive: list of column names that are sensitive features
    :param seed: int, random seed
    :return: total expressivity over these rows
    """
    train_df, test_df = train_test_split(dataset, test_size=t_split, random_state=seed)
    x_train, y_train = split_out_dataset(train_df, target_column)
    x_test, y_test = split_out_dataset(test_df, target_column)
    classifier = RandomForestClassifier(random_state=seed)
    classifier.fit(x_train, y_train)
    out_df = pd.DataFrame()

    exp_func_train = exp_func(classifier, x_train, seed)
    #print("Populating train expressivity values")
    #exp_func_train.populate_exps()

    exp_func_test = exp_func(classifier, x_test, seed)
    #print("Populating test expressivity values")
    #exp_func_test.populate_exps()

    # Temporary exp populating
    with open(f'data/exps_8020/{df_name}_train_{exp_method}_seed0', 'r') as f:
        train_temp = list(map(json.loads, f))[0]
    for e_list in train_temp:
        exp_func_train.exps.append({int(k):v for k,v in e_list.items()})
    with open(f'data/exps_8020/{df_name}_test_{exp_method}_seed0', 'r') as f:
        test_temp = list(map(json.loads, f))[0]
    for e_list in test_temp:
        exp_func_test.exps.append({int(k):v for k,v in e_list.items()})
    # ^Temporary code, delete later^

    for feature_num in range(len(x_train[0])):
        print('*****************')
        print(train_df.columns[feature_num])
        total_exp_train = full_dataset_expressivity(exp_func_train, feature_num)
        print('total exp: ', total_exp_train)
        min_solver = argmin_g(x_train, y_train, feature_num, f_sensitive, exp_func_train,
                                                   minimize=True, alphas=alphas)
        min_model, min_assigns, min_exp = min_solver.get_best_valid_model(minimize=True)
        print('min exp', min_exp, '| size', np.mean(min_assigns))

        max_solver = argmin_g(x_train, y_train, feature_num, f_sensitive, exp_func_train,
                                                   minimize=False, alphas=alphas)
        max_model, max_assigns, max_exp = max_solver.get_best_valid_model(minimize=False)
        print('max exp', max_exp, '| size', np.mean(max_assigns))

        # Choose max difference
        if abs(max_exp-total_exp_train) > abs(min_exp-total_exp_train):
            furthest_exp_train = max_exp
            assigns_train = max_assigns
            best_model = max_model
            best_solver = max_solver
            direction = 'maximize'
        else:
            furthest_exp_train = min_exp
            assigns_train = min_assigns
            best_model = min_model
            best_solver = min_solver
            direction = 'minimize'
        subgroup_size_train = np.mean(assigns_train)

        # compute test values
        total_exp_test = full_dataset_expressivity(exp_func_test, feature_num)
        #assigns_test = get_avg_prediction(best_model, x_test) # mix model method
        assigns_test = best_model.predict(x_test[:,f_sensitive])[0] # sensitive features only method
        #assigns_test = best_model.predict(x_test)[0]
        subgroup_size_test = np.mean(assigns_test)
        furthest_exp_test = 0
        for i in range(len(assigns_test)):
            furthest_exp_test += assigns_test[i]*exp_func_test.exps[i][feature_num]

        # # from mix models, pick model with largest exp diff that is valid
        params = best_model.b1.coef_
        params_with_labels = {dataset.columns[i]: float(param) for (i, param) in zip(f_sensitive, params)}
        params_with_labels['Intercept'] = best_model.b1.intercept_
        print(params_with_labels)
        out_df = pd.concat([out_df,
                            pd.DataFrame.from_records([{'Feature': dataset.columns[feature_num],
                                                        'Alpha': alphas,
                                                        'F(D)': total_exp_test,
                                                        'max(F(S))': furthest_exp_test,
                                                        'Difference': abs(furthest_exp_test - total_exp_test),
                                                        'Percent Change': 100*abs(furthest_exp_test - total_exp_test)/
                                                                          (abs(total_exp_test)+.000001),
                                                        'avg(F(D))': total_exp_test/len(assigns_test),
                                                        'avg(F(S))': furthest_exp_test/(sum(assigns_test)+.000001),
                                                        'avg diff': abs(total_exp_test/len(assigns_test) -
                                                                        furthest_exp_test/(sum(assigns_test)+.000001)),
                                                        'Subgroup Coefficients': params_with_labels,
                                                        'Subgroup Size': subgroup_size_test,
                                                        'Direction': direction,
                                                        'F(D)_train': total_exp_train,
                                                        'max(F(S))_train': furthest_exp_train,
                                                        'Difference_train': abs(furthest_exp_train-total_exp_train),
                                                        'Percent Change_train': 100*abs(furthest_exp_train-total_exp_train)/
                                                                                (abs(total_exp_train) + .000001),
                                                        'Subgroup Size_train': subgroup_size_train,
                                                        'size history': best_solver.size_history,
                                                        'lambda history': best_solver.lambda_history,
                                                        'exp history': best_solver.exp_history,
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


def run_system(df, target, sensitive_features, df_name, dummy=False, t_split=.5):
    if dummy:
        df[target] = df[target].sample(frac=1).values
        df_name = 'dummy_' + df_name

    # Sort columns so target is at end
    new_cols = [col for col in df.columns if col != target] + [target]
    df = df[new_cols]

    f_sensitive = list(df.columns.get_indexer(sensitive_features))

    # final_df = pd.DataFrame()
    # alpha_ranges = [[.01,.05],[.05,.1],[.1,.15],[.15,.2],[.2,.25]]
    # for a in alpha_ranges:
    #     print("Running", df_name, ", Alphas =", a)
    #     start = time.time()
    #     out = extremize_exps_dataset(dataset=df, exp_func=LimeExpFunc, target_column=target,
    #                                  f_sensitive=f_sensitive, alphas=a, t_split=t_split)
    #     final_df = pd.concat([final_df, out])
    #     print("Runtime:", '%.2f'%((time.time()-start)/3600), "Hours")
    # date = datetime.today().strftime('%m_%d')
    # final_df.to_csv(f'output/{df_name}_{exp_method}_output_{date}.csv')

    a = [.01,.05]
    print("Running", df_name, ", Alphas =", a)
    start = time.time()
    final_df = extremize_exps_dataset(dataset=df, exp_func=expFunc, target_column=target,
                                      f_sensitive=f_sensitive, alphas=a, t_split=t_split)
    print("Runtime:", '%.2f' % ((time.time() - start) / 3600), "Hours")
    date = datetime.today().strftime('%m_%d')
    final_df.to_csv(f'output/{df_name}_{exp_method}_output_{date}_alpha{a}.csv')

    return 1


df = pd.read_csv('data/student/student_cleaned.csv')
target = 'G3'
t_split = .5
sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
df_name = 'student'
run_system(df, target, sensitive_features, df_name, dummy, t_split)

# df = pd.read_csv('data/compas/compas_recid.csv')
# target = 'two_year_recid'
# t_split = .2
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
# df_name = 'compas_recid'
# run_system(df, target, sensitive_features, df_name, dummy, t_split)

# df = pd.read_csv('data/compas/compas_decile.csv')
# target = 'decile_score'
# t_split = .2
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
# df_name = 'compas_decile'
# run_system(df, target, sensitive_features, df_name, dummy, t_split)

# df = BankDataset().convert_to_dataframe()[0]
# target = 'y'
# t_split = .2
# sensitive_features = ['age', 'marital=married', 'marital=single', 'marital=divorced']
# df_name = 'bank'
# run_system(df, target, sensitive_features, df_name, dummy, t_split)

# df = pd.read_csv('data/folktables/ACSIncome_MI_2018_new.csv')
# target = 'PINCP'
# t_split = .2
# sensitive_features = ['AGEP', 'SEX', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0', 'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0',
#                       'RAC1P_3.0', 'RAC1P_4.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0', 'RAC1P_9.0']
# df_name = 'folktables'
# run_system(df, target, sensitive_features, df_name, dummy, t_split)
