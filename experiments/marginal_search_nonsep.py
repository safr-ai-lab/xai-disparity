import torch
from torch.special import expit as sigmoid
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from aif360.datasets import BankDataset
import pandas as pd
import numpy as np
import time
import argparse
from datetime import datetime
import json
import sys
import pdb


parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('flatval', type=float)
parser.add_argument('--dummy', action='store_true')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
flatval = args.flatval
dummy = args.dummy
useCUDA = args.cuda

# Enable GPU if desired. Sometimes returns false values
if useCUDA:
    torch.cuda.set_device('cuda:0')
else:
    torch.device('cuda:0')

def marginal_g(df, x_train, y_train, feature_num, f_sensitive, total_exp):
    max_fid, max_feature, max_threshold, max_direction, max_size = 0, None, 0, None, 0

    # Search each sensitive attribute (-1 to ignore intercept)
    for f_i in f_sensitive[:-1]:
        feature = df.columns[f_i]
        f_vals = df[feature]

        # Search each threshold within that attribute
        for threshold in set(f_vals):
            assigns = np.array(df[feature] < threshold)
            direction = '<'
            # Look at the minority group only
            if np.mean(assigns) > .5:
                assigns = np.array(df[feature] >= threshold)
                direction = '>='
            good_size = True
            if (np.mean(assigns) < .01) or (np.mean(assigns) > .99):
                good_size = False
            assigns = assigns.astype(float)

            if good_size:
                t_assigns = torch.tensor(assigns, requires_grad=True).float()
                if useCUDA:
                    t_assigns = torch.tensor(assigns, requires_grad=True).cuda()
                importance = lr_value(x_train, y_train, t_assigns, feature_num)
                fid = abs(total_exp - importance)
                if fid >= max_fid:
                    max_fid, max_feature, max_threshold, max_direction = fid, feature, threshold, direction
                    max_size = np.mean(assigns)
    return max_fid, max_feature, max_direction, max_threshold, max_size


def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    #sensitive_ds = dataset[f_sensitive].to_numpy()
    return x, y

def remove_intercept_column(x):
    mask = torch.arange(0, x.shape[1] - 1)
    x_cpu = x.cpu()
    out = torch.index_select(x_cpu, 1, mask)
    return out

def lr_value(x_0: torch.Tensor, y: torch.Tensor, assigns: torch.Tensor, feature_num: int):
    """
    Given a defined subgroup function, returns the expressivity over the test data set
    :param x_0: the test data tensor
    :param y: the test target tensor
    :param assigns: tensor with subgroup membership
    :param feature_num: the feature to test
    :return: the float value of expressivity over the dataset and subgroup assignments
    """
    x = remove_intercept_column(x_0)
    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    flat_list = [flatval for _ in range(x.shape[1])]

    if useCUDA:
        basis = torch.tensor(basis_list, requires_grad=True).cuda()
        flat = torch.tensor(flat_list, requires_grad=True).cuda()
        assigns = torch.tensor(assigns, requires_grad=True, device="cuda")
        x_t = torch.t(x).cuda()
    else:
        basis = torch.tensor(basis_list, requires_grad=True)
        flat = torch.tensor(flat_list, requires_grad=True)
        x_t = torch.t(x)
    #one_d = sigmoid(x_0 @ params)
    diag = torch.diag(assigns)
    denom = torch.inverse((x_t @ diag @ x) + torch.diag(flat))
    return (basis @ (denom @ (x_t @ diag @ y))).cpu().detach().numpy()[0]



def max_subgroup(dataset, target_column, f_sensitive, seed, t_split):
    out_df = pd.DataFrame()

    train_df, test_df = train_test_split(dataset, test_size=t_split, random_state=seed)

    if useCUDA:
        y_train = torch.tensor(train_df[target_column].values).float().cuda()
        x_train = torch.tensor(train_df.drop(target_column, axis=1).values.astype('float16')).float().cuda()
        y_test = torch.tensor(test_df[target_column].values).float().cuda()
        x_test = torch.tensor(test_df.drop(target_column, axis=1).values.astype('float16')).float().cuda()
    else:
        y_train = torch.tensor(train_df[target_column].values).float()
        x_train = torch.tensor(train_df.drop(target_column, axis=1).values.astype('float16')).float()
        y_test = torch.tensor(test_df[target_column].values).float()
        x_test = torch.tensor(test_df.drop(target_column, axis=1).values.astype('float16')).float()

    for feature_num in range(x_train.shape[1]-1):
        print('*****************')
        print(train_df.columns[feature_num])
        full_assigns = torch.tensor([1. for _ in range(x_train.shape[0])], requires_grad=True)
        if useCUDA:
            full_assigns = torch.tensor([1. for _ in range(x_train.shape[0])], requires_grad=True).cuda()
        total_exp_train = lr_value(x_train, y_train, full_assigns, feature_num)
        print('total exp: ', total_exp_train)

        # compute maximum g
        fid_train, feature, direction, threshold, size_train = marginal_g(train_df, x_train, y_train,
                                                                          feature_num, f_sensitive, total_exp_train)

        # compute values on test
        full_assigns = torch.tensor([1. for _ in range(x_test.shape[0])], requires_grad=True)
        if useCUDA:
            full_assigns = torch.tensor([1. for _ in range(x_train.shape[0])], requires_grad=True).cuda()
        total_exp_test = lr_value(x_test, y_test, full_assigns, feature_num)
        # using feature, direction, threshold, compute assigns
        if direction == '<':
            assigns = np.array(test_df[feature] < threshold)
        else:
            assigns = np.array(test_df[feature] >= threshold)
        assigns = assigns.astype(float)

        t_assigns = torch.tensor(assigns, requires_grad=True).float()
        if useCUDA:
            t_assigns = torch.tensor(t_assigns, requires_grad=True).cuda()
        importance = lr_value(x_test, y_test, t_assigns, feature_num)
        fid_test = abs(total_exp_test - importance)
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
                                                        'Subgroup Definition': feature + direction + str(threshold),
                                                        'Subgroup Size': size_test,
                                                        'F(D)_train': total_exp_train,
                                                        'avg diff_train': fid_train,
                                                        'Subgroup Size_train': size_train,
                                                        }])])

    return out_df


def run_system(df, target, sensitive_features, df_name, t_split=.5):
    # Add intercept column at the end
    df['Intercept'] = np.ones(df.shape[0])

    # Sort columns so target is at end
    new_cols = [col for col in df.columns if col != target] + [target]
    df = df[new_cols]

    f_sensitive = list(df.columns.get_indexer(sensitive_features))
    f_sensitive.append(df.shape[1] - 2)

    print("Running", df_name)
    start = time.time()
    final_df = max_subgroup(dataset=df, target_column=target, f_sensitive=f_sensitive,
                            seed=0, t_split=t_split)
    print("Runtime:", '%.2f' % ((time.time() - start) / 3600), "Hours")
    date = datetime.today().strftime('%m_%d')
    final_df.to_csv(f'output/{df_name}_marginal_linear_output_{date}.csv', index=False)

    return 1



# df = pd.read_csv('data/student/student_cleaned.csv')
# target = 'G3'
# t_split = .5
# sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
# df_name = 'student'
# run_system(df, target, sensitive_features, df_name, t_split)

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
#
# df = pd.read_csv('data/folktables/ACSIncome_MI_2018_new.csv')
# target = 'PINCP'
# t_split = .2
# sensitive_features = ['AGEP', 'SEX', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0', 'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0',
#                       'RAC1P_3.0', 'RAC1P_4.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0', 'RAC1P_9.0']
# df_name = 'folktables'
# run_system(df, target, sensitive_features, df_name, t_split)
