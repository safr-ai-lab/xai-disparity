import torch
import pandas as pd
import numpy as np
from torch.special import expit as sigmoid
from torch.optim import Adam
import time
from datetime import datetime
from aif360.datasets import CompasDataset, BankDataset
from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('--dummy', action='store_true')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
dummy = args.dummy
useCUDA = args.cuda

# Enable GPU if desired. Sometimes returns false values
if useCUDA:
    torch.cuda.set_device('cuda:0')
else:
    torch.device('cuda:0')


def loss_fn_generator(x_0: torch.Tensor, y: torch.Tensor, initial_val: float, feature_num: int, sensitives: torch.Tensor):
    """
    Factory for the loss function that pytorch runs will be optimizing in WLS
    :param x_0: the data tensor with intercept column
    :param y: the target tensor
    :param feature_num: Which feature in the data do we care about
    :param minimize: Boolean -- are we minimizing or maximizing
    :return: a loss function for our particular WLS problem.
    """
    x = remove_intercept_column(x_0)

    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    flat_list = [0.001 for _ in range(x.shape[1])]

    if useCUDA:
        basis = torch.tensor(basis_list, requires_grad=True).cuda()
        flat = torch.tensor(flat_list, requires_grad=True).cuda()
    else:
        basis = torch.tensor(basis_list, requires_grad=True)
        flat = torch.tensor(flat_list, requires_grad=True)

    # Look into derivation of gradient by hand and implementing it here instead.
    def loss_fn(params):
        # only train using sensitive features
        one_d = sigmoid(x_0 @ (sensitives * params))
        diag = torch.diag(one_d)
        x_t = torch.t(x)
        # Try ridge reg by weight of params and not by flat value?
        denom = torch.inverse((x_t @ diag @ x) + torch.diag(flat))
        difference_penalty = torch.abs(basis @ (denom @ (x_t @ diag @ y)) - initial_val)

        size_penalty = 0
        subgroup_size = torch.sum(one_d)/x.shape[0]
        if subgroup_size < .05 or subgroup_size > .8:
            size_penalty = 100*torch.abs((torch.sum(one_d)/x.shape[0])-.5)

        # we want to maximize difference penalty but minimize size penalty
        return size_penalty - difference_penalty

    return loss_fn


def train_and_return(x: torch.Tensor, y: torch.Tensor, feature_num: int, initial_val: float, f_sensitive: list, seed: int):
    """
    Given an x, y, feature num, and the expressivity over the whole dataset,
    returns the differential expressivity and maximal subset for that feature
    :param x: The data tensor
    :param y: the target tensor
    :param feature_num: which feature to optimize, int.
    :param initial_val: What the expressivity over the whole dataset for the feature is.
    :param f_sensitive: indices of sensitive features
    :param seed: initializing seed for reproducibility
    :return: the differential expressivity and maximal subset weights.
    """
    niters = 1000
    # Set seed to const value for reproducibility
    torch.manual_seed(seed)
    s_list = [0. for _ in range(x.shape[1])]
    for f in f_sensitive:
        s_list[f] = 1.
    if useCUDA:
        sensitives = torch.tensor(s_list, requires_grad=True).cuda()
        params_max = torch.randn(x.shape[1], requires_grad=True, device="cuda")
    else:
        sensitives = torch.tensor(s_list, requires_grad=True)
        params_max = torch.randn(x.shape[1], requires_grad=True)

    optim = Adam(params=[params_max], lr=0.05)
    iters = 0
    curr_error = 10000
    loss_max = loss_fn_generator(x, y, initial_val, feature_num, sensitives)
    while iters < niters:
        optim.zero_grad()
        loss_res = loss_max(params_max)
        loss_res.backward()
        optim.step()
        curr_error = loss_res.item()
        iters += 1
    params_max = sensitives * params_max
    max_error = curr_error * -1
    assigns = (sigmoid(x @ params_max)).cpu().detach().numpy()
    print(torch.sum(sigmoid(x @ params_max))/x.shape[0])
    #print(max_error, initial_val, assigns[assigns >= 0.02])
    return max_error, assigns, params_max.cpu().detach().numpy()


def initial_value(x: torch.Tensor, y: torch.Tensor, feature_num: int) -> float:
    """
    Given a dataset, target, and feature number, returns the expressivity of that feature over the dataset.
    :param x: the data tensor
    :param y: the target tensor
    :param feature_num: the feature to test
    :return: the float value of expressivity over the dataset
    """
    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    flat_list = [0.001 for _ in range(x.shape[1])]

    if useCUDA:
        basis = torch.tensor(basis_list, requires_grad=True).cuda()
        flat = torch.tensor(flat_list, requires_grad=True).cuda()
        x_t = torch.t(x).cuda()
        denom = torch.inverse((x_t @ x) + torch.diag(flat)).cuda()
    else:
        basis = torch.tensor(basis_list, requires_grad=True)
        flat = torch.tensor(flat_list, requires_grad=True)
        x_t = torch.t(x)
        denom = torch.inverse((x_t @ x) + torch.diag(flat))
    return (basis @ (denom @ (x_t @ y))).item()

def final_value(x_0: torch.Tensor, y: torch.Tensor, params: torch.Tensor, feature_num: int):
    """
    Given a defined subgroup function, returns the expressivity over the test data set
    :param x_0: the test data tensor
    :param y: the test target tensor
    :param params: tensor with coefficients defining the subgroup
    :param feature_num: the feature to test
    :return: the float value of expressivity over the dataset and subgroup assignments
    """
    x = remove_intercept_column(x_0)
    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    flat_list = [0.001 for _ in range(x.shape[1])]

    if useCUDA:
        basis = torch.tensor(basis_list, requires_grad=True).cuda()
        flat = torch.tensor(flat_list, requires_grad=True).cuda()
        params = torch.tensor(params, requires_grad=True, device="cuda")
        x_t = torch.t(x).cuda()
    else:
        basis = torch.tensor(basis_list, requires_grad=True)
        flat = torch.tensor(flat_list, requires_grad=True)
        x_t = torch.t(x)

    one_d = sigmoid(x_0 @ params)
    diag = torch.diag(one_d)
    denom = torch.inverse((x_t @ diag @ x) + torch.diag(flat))
    return (basis @ (denom @ (x_t @ diag @ y))).detach().numpy()[0], one_d.cpu().detach().numpy()

def find_extreme_subgroups(dataset: pd.DataFrame, seed: int, target_column: str, f_sensitive: list):
    """
    Given a dataset, finds the differential expressivity and maximal subset over all features.
    Saves that subset to a file.
    :param dataset: the pandas dataframe to use
    :param seed: pseudorandom seed for reproducibility
    :param target_column:  Which column in that dataframe is the target.
    :param f_sensitive: Which features are sensitive characteristics
    :return:  N/A.  Logs results.
    """
    out_df = pd.DataFrame()

    train_df, test_df = train_test_split(dataset, test_size=.2, random_state=seed)

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
    errors_and_weights = []
    for feature_num in range(x_train.shape[1]-1):
        print("Feature", feature_num, "of", x_train.shape[1]-1)
        x_train_ni = remove_intercept_column(x_train)
        total_exp_train = initial_value(x_train_ni, y_train, feature_num)
        try:
            _, assigns_train, params = train_and_return(x_train, y_train, feature_num, total_exp_train, f_sensitive, seed)
            furthest_exp_train, _ = final_value(x_train, y_train, params, feature_num)
            print(furthest_exp_train)
            subgroup_size_train = [round(a) for a in assigns_train].count(1)/len(assigns_train)
            if not (np.isnan(furthest_exp_train)):
                x_test_ni = remove_intercept_column(x_test)
                total_exp = initial_value(x_test_ni, y_test, feature_num)
                furthest_exp, assigns = final_value(x_test, y_test, params, feature_num)
                subgroup_size = [round(a) for a in assigns].count(1) / len(assigns)
                errors_and_weights.append((furthest_exp, feature_num))
                print(furthest_exp, feature_num)
                params_with_labels = {dataset.columns[i]: float(param) for (i, param) in enumerate(params)}
                out_df = pd.concat([out_df, pd.DataFrame.from_records([{'Feature': dataset.columns[feature_num],
                                                                        'F(D)': total_exp,
                                                                        'max(F(S))': furthest_exp,
                                                                        'Difference': abs(furthest_exp - total_exp),
                                                                        'Subgroup Coefficients': params_with_labels,
                                                                        'Subgroup Size': subgroup_size,
                                                                        'F(D)_train': total_exp_train,
                                                                        'max(F(S))_train': furthest_exp_train,
                                                                        'Difference_train': abs(furthest_exp_train - total_exp_train),
                                                                        'Subgroup Size_train':subgroup_size_train}])])
        except RuntimeError as e:
            print(e)
            continue
    errors_sorted = sorted(errors_and_weights, key=lambda elem: abs(elem[0]), reverse=True)
    print(errors_sorted[0])
    #i_value = initial_value(x, y, errors_sorted[0][1])
    #error, assigns, params = train_and_return(x, y, errors_sorted[0][1], i_value, f_sensitive, seed)
    #print(error, assigns[(assigns >= 0.002) & (assigns <= 1.0)])
    # params_with_labels = np.array(
    #     sorted([[dataset.columns[i], float(param)] for i, param in enumerate(params)], key=lambda row: abs(row[1]),
    #            reverse=True))
    # print(params_with_labels)
    print(dataset.columns[errors_sorted[0][1]])

    return out_df

def remove_intercept_column(x):
    mask = torch.arange(0, x.shape[1] - 1)
    x_cpu = x.cpu()
    out = torch.index_select(x_cpu, 1, mask)
    if useCUDA:
        out = out.cuda()
    return out

def run_system(df, target, sensitive_features, df_name, dummy=False):
    if dummy:
        df[target] = df[target].sample(frac=1).values
        df_name = 'dummy_' + df_name

    # Add intercept column at the end
    df['Intercept'] = np.ones(df.shape[0])

    # Sort columns so target is at end
    new_cols = [col for col in df.columns if col != target] + [target]
    df = df[new_cols]

    # Get indices of sensitive features. Append a 1 for intercept
    f_sensitive = list(df.columns.get_indexer(sensitive_features))
    f_sensitive.append(df.shape[1]-2)

    print(df.shape[1])
    print(f_sensitive)

    seeds = [0]
    for s in seeds:
        print("Running", df_name, ", Seed =", s)
        start = time.time()
        out = find_extreme_subgroups(df, seed=s, target_column=target, f_sensitive=f_sensitive)
        # use date as naming convention
        date = datetime.today().strftime('%m_%d')
        fname = f'output/nonsep/t{df_name}_output_{date}.csv'
        out.to_csv(fname)
        print("Runtime:", '%.2f'%((time.time()-start)/3600), "Hours")
    return 1


df = pd.read_csv('data/student/student_cleaned.csv')
target = 'G3'
sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
df_name = 'student'
run_system(df, target, sensitive_features, df_name, dummy)

# df = CompasDataset().convert_to_dataframe()[0]
# df = pd.read_csv('data/compas/compas_cleaned.csv')  #compas_cleaned_decile.csv
# target = 'two_year_recid'  #'decile_score'
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
# df_name = 'compas'
# run_system(df, target, sensitive_features, df_name, dummy)

# df = BankDataset().convert_to_dataframe()[0]
# target = 'y'
# sensitive_features = ['age', 'marital=married', 'marital=single', 'marital=divorced']
# df_name = 'bank'
# run_system(df, target, sensitive_features, df_name, dummy)

# df = pd.read_csv('data/folktables/ACSIncome_MI_2018_sampled.csv')
# target = 'PINCP'
# sensitive_features = ['AGEP', 'SEX', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0', 'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0',
#                       'RAC1P_3.0', 'RAC1P_4.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0', 'RAC1P_9.0']
# df_name = 'folktables'
# run_system(df, target, sensitive_features, df_name, dummy)

