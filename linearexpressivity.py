import torch
import numpy as np
from torch.special import expit as sigmoid
from torch.optim import Adam
from aif360.datasets import CompasDataset

compas_df = CompasDataset().convert_to_dataframe()[0]

torch.cuda.set_device('cuda:0')



def loss_fn_generator(x, y, flat, feature_num, minimize=False):
    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    basis = torch.tensor(basis_list, requires_grad=True).cuda()
    factor = -1
    if minimize:
        factor = 1

    def loss_fn(params):
        one_d = sigmoid(x @ params)
        diag = torch.diag(one_d + flat) * (float(x.shape[0]) / torch.sum(one_d + flat))
        x_t = torch.t(x)
        denom = torch.inverse(x_t @ diag @ x)
        return factor * (basis @ (denom @ (x_t @ diag @ y)))

    return loss_fn


def train_and_return(x, y, feature_num, initial_value):
    niters = 300
    torch.manual_seed(0)
    params_min = torch.randn(x.shape[1], requires_grad=True, device="cuda")
    flat_list = [0.0001 for _ in range(x.shape[0])]
    flat = torch.tensor(flat_list, requires_grad=True).cuda()
    optim = Adam(params=[params_min], lr=0.05)
    iters = 0
    curr_error = 10000
    loss_min = loss_fn_generator(x, y, flat, feature_num, minimize=True)
    while iters < niters:
        optim.zero_grad()
        loss_res = loss_min(params_min)
        loss_res.backward()
        optim.step()
        curr_error = loss_res.item()
        iters += 1
    min_error = curr_error
    params_max = torch.randn(x.shape[1], requires_grad=True, device="cuda")
    optim = Adam(params=[params_max], lr=0.05)
    iters = 0
    curr_error = 10000
    loss_max = loss_fn_generator(x, y, flat, feature_num, minimize=False)
    while iters < niters:
        optim.zero_grad()
        loss_res = loss_max(params_max)
        loss_res.backward()
        optim.step()
        curr_error = loss_res.item()
        iters += 1
    max_error = curr_error * -1
    if abs(max_error - initial_value) > abs(min_error - initial_value):
        print(max_error, initial_value, (sigmoid(x @ params_max) + flat).cpu().detach().numpy())
        return max_error - initial_value, (sigmoid(x @ params_max) + flat).cpu().detach().numpy()

    print(min_error, initial_value, (sigmoid(x @ params_min) + flat).cpu().detach().numpy())
    return min_error - initial_value, (sigmoid(x @ params_min) + flat).cpu().detach().numpy()


def initial_value(x, y, feature_num):
    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    basis = torch.tensor(basis_list, requires_grad=True).cuda()
    x_t = torch.t(x).cuda()
    denom = torch.inverse(x_t @ x).cuda()
    return (basis @ (denom @ (x_t @ y))).item()


def find_extreme_subgroups(dataset):
    y = torch.tensor(dataset["two_year_recid"].values).float().cuda()
    x = torch.tensor(dataset.drop("two_year_recid", axis=1).values).float().cuda()
    errors_and_weights = []
    for feature_num in range(int(x.shape[1])):
        full_dataset = initial_value(x, y, feature_num)
        try:
            error, _ = train_and_return(x, y, feature_num, full_dataset)
            if not (np.isnan(error)):
                errors_and_weights.append((error, feature_num))
                print(error, feature_num)
        except RuntimeError as e:
            print(e)
            continue
    errors_sorted = sorted(errors_and_weights, key=lambda elem: abs(elem[0]), reverse=True)
    print(errors_sorted[0])
    error, assigns = train_and_return(x, y, errors_sorted[0][1], initial_value(errors_sorted[0][1]))
    print(error, assigns)
    np.savetxt(f"assignments_feature_{errors_sorted[0][1]}:error_{error}.csv", assigns, fmt='%.3f', delimiter=",")
    print(dataset.columns[errors_sorted[0][1]])


find_extreme_subgroups(compas_df)