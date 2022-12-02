from lime_exp_func import LimeExpFunc
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


parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('--dummy', action='store_true')
args = parser.parse_args()
dummy = args.dummy


def argmin_g(x, y, feature_num, exp_func, minimize=True):
    exp_order = np.mean([abs(exp_func.exps[i][feature_num]) for i in range(len(x))])
    solver = ConstrainedSolver(exp_func, alpha_s=.05, alpha_L=.5, B=10*exp_order)
    v = exp_order

    costs0 = [0 for _ in range(len(x))] # costs0 is always zeros
    learner = Learner(x, y, linear_model.LinearRegression())
    #for _ in range(5):
    _ = 0
    while solver.v_t > v:
        _ += 1
        print("ITERATION NUMBER ", _)
        solver.update_lambdas()
        #print('lambdas: ',solver.lambda_history[-1])
        avg_lam = [np.mean(k) for k in zip(*solver.lambda_history)]
        print('avg lambdas: ', avg_lam)
        costs1 = [exp_func.exps[i][feature_num]-avg_lam[0]+avg_lam[1] for i in range(len(x))]

        # CSC solver, returns regoracle fit using costs0/costs1
        l_response = learner.best_response(costs0, costs1, minimize)
        solver.g_history.append(l_response)

        assigns, expressivity = l_response.predict(x)
        solver.pred_history.append(np.array(assigns))
        solver.exp_history.append(expressivity)
        #print('assigns', assigns)
        print('assign size', np.mean(assigns))
        #print('exps', expressivity)

        avg_pred = [np.mean(k) for k in zip(*solver.pred_history)]
        best_lam = solver.best_lambda(avg_pred)
        L_ceiling = solver.lagrangian(avg_pred, best_lam, feature_num)
        #print(avg_pred, np.mean(avg_pred))
        #print('best lam: ',best_lam)
        #print('L ceiling', L_ceiling)

        avg_lam = [np.mean(k) for k in zip(*solver.lambda_history)]
        best_g = solver.best_g(learner, feature_num, avg_lam, minimize)
        best_g_assigns, best_g_exps = best_g.predict(x)
        L_floor = solver.lagrangian(best_g_assigns, avg_lam, feature_num)
        #print('avg_lam', avg_lam)
        #print('best g assigns', best_g_assigns)
        #print('best g exps', best_g_exps)
        #print('L floor', L_floor)

        L = solver.lagrangian(avg_pred, avg_lam, feature_num)
        #print('L',L)

        solver.v_t = max(L-L_floor, L_ceiling-L)
        print('end v_t:', solver.v_t)

        solver.update_thetas(assigns)
        print('new thetas:', solver.thetas[-1])

    avg_pred = [np.mean(k) for k in zip(*solver.pred_history)]
    avg_lam = [np.mean(k) for k in zip(*solver.lambda_history)]
    final_expressivity = 0
    for i in range(len(avg_pred)):
        final_expressivity += avg_pred[i]*exp_func.exps[i][feature_num]
    return solver.g_history, avg_pred, avg_lam, final_expressivity

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

def split_out_dataset(dataset, target_column, f_sensitive):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    sensitive_ds = dataset[f_sensitive].to_numpy()
    return x, y, sensitive_ds


def extremize_exps_dataset(dataset, exp_func, target_column, f_sensitive, seed):
    """
    :param dataset: pandas dataframe
    :param exp_func: class for expressivities
    :param target_column: string, column name in dataset
    :param f_sensitive: list of column names that are sensitive features
    :param seed: int, random seed
    :return: total expressivity over these rows
    """
    train_df, test_df = train_test_split(dataset, test_size=0.5, random_state=seed)
    x_train, y_train, sensitive_train = split_out_dataset(train_df, target_column, f_sensitive)
    x_test, y_test, sensitive_test = split_out_dataset(test_df, target, f_sensitive)
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
    with open('data/exps/student_train_seed0', 'r') as f:
        train_temp = list(map(json.loads, f))[0]
    for e_list in train_temp:
        exp_func_train.exps.append({int(k):v for k,v in e_list.items()})
    with open('data/exps/student_test_seed0', 'r') as f:
        test_temp = list(map(json.loads, f))[0]
    for e_list in test_temp:
        exp_func_test.exps.append({int(k):v for k,v in e_list.items()})
    # ^Temporary code, delete later^

    # for feature_num in range(len(train_x[0])):
    for feature_num in range(1):
        total_exp_train = full_dataset_expressivity(exp_func_train, feature_num)
        min_models, min_assigns, _, min_exp = argmin_g(x_train, y_train, feature_num, exp_func_train, minimize=True)
        max_models, max_assigns, _, max_exp = argmin_g(x_train, y_train, feature_num, exp_func_train, minimize=False)
        print('total exp: ', total_exp_train)
        print('min exp', min_exp, '| size', sum(min_assigns)/len(min_assigns))
        print('max exp', max_exp, '| size', sum(max_assigns)/len(max_assigns))

        # Choose max difference
        if abs(max_exp-total_exp_train) > abs(min_exp-total_exp_train):
            furthest_exp_train = max_exp
            assigns_train = max_assigns
            mix_model = max_models
            direction = 'maximize'
        else:
            furthest_exp_train = min_exp
            assigns_train = min_assigns
            mix_model = min_models
            direction = 'minimize'
        subgroup_size_train = sum(assigns_train)/len(assigns_train)

        total_exp_test = full_dataset_expressivity(exp_func_test, feature_num)
        assigns_test = get_avg_prediction(mix_model, x_test)
        subgroup_size_test = sum(assigns_test)/len(assigns_test)
        furthest_exp_test = 0
        for i in range(len(assigns_test)):
            furthest_exp_test += assigns_test[i]*exp_func_test.exps[i][feature_num]

        ### What do we use to define the group? Logistic regression on the assigned points? ###
        subgroup_model = linear_model.LogisticRegression(solver='lbfgs', max_iter=200, random_state=seed).fit(sensitive_test,
                                                                                                 assigns_test)
        params = subgroup_model.coef_[0]
        print(params)
        params_with_labels = {dataset[f_sensitive].columns[i]: float(param) for (i, param) in enumerate(params)}

        out_df = pd.concat([out_df, pd.DataFrame.from_records([{'Feature': dataset.columns[feature_num],
                                                                'F(D)': total_exp_test,
                                                                'max(F(S))': furthest_exp_test,
                                                                'Difference': abs(furthest_exp_test - total_exp_test),
                                                                'Subgroup Coefficients': params_with_labels,
                                                                'Subgroup Size': subgroup_size_test,
                                                                'Direction': direction,
                                                                'F(D)_train': total_exp_train,
                                                                'max(F(S))_train': furthest_exp_train,
                                                                'Difference_train': abs(furthest_exp_train-total_exp_train),
                                                                'Subgroup Size_train': subgroup_size_train}])])


    return out_df


def run_system(df, target, sensitive_features, df_name, dummy=False):
    if dummy:
        df[target] = df[target].sample(frac=1).values
        df_name = 'dummy_' + df_name

    # Sort columns so target is at end
    new_cols = [col for col in df.columns if col != target] + [target]
    df = df[new_cols]

    seeds = [0]
    for s in seeds:
        np.random.seed(s)
        print("Running", df_name, ", Seed =", s)
        start = time.time()

        out = extremize_exps_dataset(dataset=df, exp_func=LimeExpFunc, target_column=target,
                                     f_sensitive=sensitive_features, seed=s)

        date = datetime.today().strftime('%m_%d')
        #out.to_csv(f'output/sep/{df_name}_LIME_output_seed{s}_{date}.csv')
        print("Runtime:", '%.2f'%((time.time()-start)/3600), "Hours")
    return 1



df = pd.read_csv('data/student/student_cleaned.csv')
target = 'G3'
sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
df_name = 'student'
run_system(df, target, sensitive_features, df_name, dummy)