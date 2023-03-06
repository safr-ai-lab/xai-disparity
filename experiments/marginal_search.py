from ..notions import grad_imp_func, lime_imp_func, shap_imp_func
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from aif360.datasets import BankDataset
import pandas as pd
import numpy as np
import time
import argparse
from datetime import datetime
import json
import sys

parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('exp_method', type=str)
args = parser.parse_args()
exp_method = args.exp_method

if exp_method == 'lime':
    expFunc = lime_imp_func.LimeImpFunc
elif exp_method == 'shap':
    expFunc = shap_imp_func.ShapImpFunc
elif exp_method == 'grad':
    expFunc = grad_imp_func.GradImpFunc
else:
    sys.exit('Importance method not recognized')


def marginal_g(df, feature_num, f_sensitive, expfunc, total_exp):
    max_fid, max_feature, max_threshold, max_direction, max_size = 0, None, 0, None, 0

    # Search each sensitive attribute
    for f_i in f_sensitive:
        feature = df.columns[f_i]
        f_vals = df[feature]

        # Search each threshold within that attribute
        for threshold in set(f_vals):
            assigns = list(df[feature] < threshold)
            direction = '<'
            # Look at the minority group only
            if np.mean(assigns) > .5:
                assigns = list(df[feature] >= threshold)
                direction = '>='
            good_size = True
            if (np.mean(assigns) < .01) or (np.mean(assigns) > .99):
                good_size = False

            if good_size:
                importance = 0
                for i in range(len(assigns)):
                    importance += assigns[i] * expfunc.get_exp(row=i, feature=feature_num)
                fid = abs((total_exp/df.shape[0]) - (importance/sum(assigns)))
                if fid >= max_fid:
                    max_fid, max_feature, max_threshold, max_direction = fid, feature, threshold, direction
                    max_size = np.mean(assigns)
    return max_fid, max_feature, max_direction, max_threshold, max_size


def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    #sensitive_ds = dataset[f_sensitive].to_numpy()
    return x, y

def full_dataset_expressivity(exp_func, feature_num):
    total = 0
    for row in exp_func.exps:
        total += row[feature_num]
    return total

def partial_dataset_expressivity(exp_func, indices, feature_num):
    total = 0
    for i in indices:
        row = exp_func.exps[i]
        total += row[feature_num]
    return total


def max_subgroup(dataset, exp_func, target_column, f_sensitive, seed, t_split):
    train_df, test_df = train_test_split(dataset, test_size=t_split, random_state=seed)
    x_train, y_train = split_out_dataset(train_df, target_column)
    x_test, y_test = split_out_dataset(test_df, target_column)
    classifier = RandomForestClassifier(random_state=seed)
    classifier.fit(x_train, y_train)

    out_df = pd.DataFrame()

    exp_func_train = exp_func(classifier, x_train, seed)
    # print("Populating train expressivity values")
    # exp_func_train.populate_exps()

    exp_func_test = exp_func(classifier, x_test, seed)
    # print("Populating test expressivity values")
    # exp_func_test.populate_exps()

    # Temporary exp populating
    with open(f'data/exps_8020/{df_name}_train_{exp_method}LR_seed0', 'r') as f:
        train_temp = list(map(json.loads, f))[0]
    for e_list in train_temp:
        exp_func_train.exps.append({int(k): v for k, v in e_list.items()})
    with open(f'data/exps_8020/{df_name}_test_{exp_method}LR_seed0', 'r') as f:
        test_temp = list(map(json.loads, f))[0]
    for e_list in test_temp:
        exp_func_test.exps.append({int(k): v for k, v in e_list.items()})
    # ^Temporary code, delete later^

    for feature_num in range(len(x_train[0])):
        print('*****************')
        print(train_df.columns[feature_num])
        total_exp_train = full_dataset_expressivity(exp_func_train, feature_num)
        print('total exp: ', total_exp_train)

        # compute maximum g
        fid_train, feature, direction, threshold, size_train = marginal_g(train_df, feature_num, f_sensitive,
                                                                          exp_func_train, total_exp_train)

        # compute values on test
        total_exp_test = full_dataset_expressivity(exp_func_test, feature_num)
        # using feature, direction, threshold, compute assigns
        if direction == '<':
            assigns = list(test_df[feature] < threshold)
        else:
            assigns = list(test_df[feature] >= threshold)
        importance = 0
        for i in range(len(assigns)):
            importance += assigns[i] * exp_func_test.get_exp(row=i, feature=feature_num)
        fid_test = abs((total_exp_test / len(assigns)) - (importance / (sum(assigns)+.000001)))
        size_test = np.mean(assigns)

        # append to out_df
        out_df = pd.concat([out_df,
                            pd.DataFrame.from_records([{'Feature': dataset.columns[feature_num],
                                                        'F(D)': total_exp_test,
                                                        'max(F(S))': importance,
                                                        'Difference': abs(importance - total_exp_test),
                                                        'Percent Change': 100 * abs(
                                                            importance - total_exp_test) /
                                                                          (abs(total_exp_test) + .000001),
                                                        'avg(F(D))': total_exp_test / len(assigns),
                                                        'avg(F(S))': importance / (sum(assigns) + .000001),
                                                        'avg diff': fid_test,
                                                        'Subgroup Definition': feature + direction + str(threshold),
                                                        'Subgroup Size': size_test,
                                                        'F(D)_train': total_exp_train,
                                                        'avg diff_train': fid_train,
                                                        'Subgroup Size_train': size_train,
                                                        }])])

    return out_df


def run_system(df, target, sensitive_features, df_name, t_split=.5):
    # Sort columns so target is at end
    new_cols = [col for col in df.columns if col != target] + [target]
    df = df[new_cols]

    f_sensitive = list(df.columns.get_indexer(sensitive_features))

    print("Running", df_name)
    start = time.time()
    final_df = max_subgroup(dataset=df, exp_func=expFunc, target_column=target, f_sensitive=f_sensitive,
                            seed=0, t_split=t_split)
    print("Runtime:", '%.2f' % ((time.time() - start) / 3600), "Hours")
    date = datetime.today().strftime('%m_%d')
    final_df.to_csv(f'output/{df_name}_marginal_{exp_method}_output_{date}.csv', index=False)

    return 1



# df = pd.read_csv('data/student/student_cleaned.csv')
# target = 'G3'
# t_split = .5
# sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
# df_name = 'student'
# run_system(df, target, sensitive_features, df_name, t_split)
#
# df = pd.read_csv('data/compas/compas_recid.csv')
# target = 'two_year_recid'
# t_split = .2
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
# df_name = 'compas_recid'
# run_system(df, target, sensitive_features, df_name, t_split)
#
# df = pd.read_csv('data/compas/compas_decile.csv')
# target = 'decile_score'
# t_split = .2
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
# df_name = 'compas_decile'
# run_system(df, target, sensitive_features, df_name, t_split)
#
# df = BankDataset().convert_to_dataframe()[0]
# target = 'y'
# t_split = .2
# sensitive_features = ['age', 'marital=married', 'marital=single', 'marital=divorced']
# df_name = 'bank'
# run_system(df, target, sensitive_features, df_name, t_split)

# df = pd.read_csv('data/folktables/ACSIncome_MI_2018_new.csv')
# target = 'PINCP'
# t_split = .2
# sensitive_features = ['AGEP', 'SEX', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0', 'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0',
#                       'RAC1P_3.0', 'RAC1P_4.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0', 'RAC1P_9.0']
# df_name = 'folktables'
# run_system(df, target, sensitive_features, df_name, t_split)
