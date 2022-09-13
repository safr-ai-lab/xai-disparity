import torch
import pandas as pd
import numpy as np
from torch.special import expit as sigmoid
from torch.optim import Adam
import time
from aif360.datasets import CompasDataset

# Enable GPU if desired
useCUDA = True
if useCUDA:
    torch.cuda.set_device('cuda:0')
else:
    torch.device('cuda:0')

# Initialize the dataset from CSV
compas_df = CompasDataset().convert_to_dataframe()[0]


def loss_fn_generator(x: torch.Tensor, y: torch.Tensor, initial_val: float, flat: torch.Tensor, feature_num: int):
    """
    Factory for the loss function that pytorch runs will be optimizing in WLS
    :param x: the data tensor
    :param y: the target tensor
    :param flat: The flat weight to add to each row -- prevents div by 0
    :param feature_num: Which feature in the data do we care about
    :param minimize: Boolean -- are we minimizing or maximizing
    :return: a loss function for our particular WLS problem.
    """
    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    if useCUDA:
        basis = torch.tensor(basis_list, requires_grad=True).cuda()
    else:
        basis = torch.tensor(basis_list, requires_grad=True)

    def loss_fn(params):
        one_d = sigmoid(x @ params)
        # We add a flat value to prevent div by 0, then normalize by the trace
        diag = torch.diag(one_d + flat)
        x_t = torch.t(x)
        denom = torch.inverse(x_t @ diag @ x)
        # we want to maximize this, so multiply by -1
        return -1 * (torch.abs(basis @ (denom @ (x_t @ diag @ y)) - initial_val) + 15 * torch.sum(one_d + flat) /
                     x.shape[0])

    return loss_fn


def train_and_return(x: torch.Tensor, y: torch.Tensor, feature_num: int, initial_val: float, seed: int):
    """
    Given an x, y, feature num, and the expressivity over the whole dataset,
    returns the differential expressivity and maximal subset for that feature
    :param x: The data tensor
    :param y: the target tensor
    :param feature_num: which feature to optimize, int.
    :param initial_val: What the expressivity over the whole dataset for the feature is.
    :return: the differential expressivity and maximal subset weights.
    """
    niters = 1000
    # Set seed to const value for reproducibility
    torch.manual_seed(seed)
    flat_list = [0.001 for _ in range(x.shape[0])]
    if useCUDA:
        flat = torch.tensor(flat_list, requires_grad=True).cuda()
        params_max = torch.randn(x.shape[1], requires_grad=True, device="cuda")
    else:
        flat = torch.tensor(flat_list, requires_grad=True)
        params_max = torch.randn(x.shape[1], requires_grad=True)
    optim = Adam(params=[params_max], lr=0.05)
    iters = 0
    curr_error = 10000
    loss_max = loss_fn_generator(x, y, initial_val, flat, feature_num)
    while iters < niters:
        optim.zero_grad()
        loss_res = loss_max(params_max)
        loss_res.backward()
        optim.step()
        curr_error = loss_res.item()
        iters += 1
    max_error = curr_error * -1
    assigns = (sigmoid(x @ params_max) + flat).cpu().detach().numpy()
    print(max_error, initial_val, assigns[assigns >= 0.02])
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
    if useCUDA:
        basis = torch.tensor(basis_list, requires_grad=True).cuda()
        x_t = torch.t(x).cuda()
        denom = torch.inverse(x_t @ x).cuda()
    else:
        basis = torch.tensor(basis_list, requires_grad=True)
        x_t = torch.t(x)
        denom = torch.inverse(x_t @ x)
    return (basis @ (denom @ (x_t @ y))).item()


def find_extreme_subgroups(dataset: pd.DataFrame, seed: int, target_column: str = 'two_year_recid'):
    """
    Given a dataset, finds the differential expressivity and maximal subset over all features.
    Saves that subset to a file.
    :param dataset: the pandas dataframe to use
    :param target_column:  Which column in that dataframe is the target.
    :return:  N/A.  Logs results.
    """

    if useCUDA:
        y = torch.tensor(dataset[target_column].values).float().cuda()
        x = torch.tensor(dataset.drop(target_column, axis=1).values.astype('float16')).float().cuda()
    else:
        y = torch.tensor(dataset[target_column].values).float()
        x = torch.tensor(dataset.drop(target_column, axis=1).values.astype('float16')).float()
    errors_and_weights = []
    for feature_num in range(x.shape[1]):
        print("Feature", feature_num, "of", x.shape[1])
        full_dataset = initial_value(x, y, feature_num)
        try:
            error, _, _ = train_and_return(x, y, feature_num, full_dataset, seed)
            if not (np.isnan(error)):
                errors_and_weights.append((error, feature_num))
                print(error, feature_num)
        except RuntimeError as e:
            print(e)
            continue
    errors_sorted = sorted(errors_and_weights, key=lambda elem: abs(elem[0]), reverse=True)
    print(errors_sorted[0])
    i_value = initial_value(x, y, errors_sorted[0][1])
    error, assigns, params = train_and_return(x, y, errors_sorted[0][1], i_value, seed)
    print(error, assigns[(assigns >= 0.002) & (assigns <= 1.0)])
    params_with_labels = np.array(
        sorted([[dataset.columns[i], float(param)] for i, param in enumerate(params)], key=lambda row: abs(row[1]),
               reverse=True))
    print(params_with_labels)
    with open(f'output_{dataset.columns[errors_sorted[0][1]]}_seed_{seed}.txt', 'w') as f:
        f.write("Initial Value: ")
        f.write(str(i_value))
        f.write("\n Final Value:")
        f.write(str(error))
    #np.savetxt(f"assignments_feature_{dataset.columns[errors_sorted[0][1]]}_error_{error}.csv", assigns, fmt='%.3f',
    #           delimiter=",")
    np.savetxt(f"params_feature_{dataset.columns[errors_sorted[0][1]]}_seed_{seed}_error_{error}.csv", params_with_labels,
               delimiter=",", fmt='%s: %s')
    print(dataset.columns[errors_sorted[0][1]])


if __name__ == "__main__":
    start = time.time()
    seeds = [0,1,2]
    for s in seeds:
        find_extreme_subgroups(compas_df, seed=s)
    print("Runtime:", '%.2f'%((time.time()-start)/3600), "Hours")
