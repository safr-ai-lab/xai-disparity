import torch
from torch.special import expit as sigmoid
from torch.optim import Adam
from aif360.datasets import CompasDataset



compas_df = CompasDataset().convert_to_dataframe()[0]

torch.cuda.set_device('cuda:0')

y = torch.tensor(compas_df["two_year_recid"].values).float().cuda()
x = torch.tensor(compas_df.drop("two_year_recid", axis=1).values).float().cuda()




def loss_fn_generator(feature_num, minimize = False):
    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    basis = torch.tensor(basis_list, requires_grad=True).cuda()
    factor = -1
    if minimize:
        factor = 1

    def loss_fn(params):
        one_d = sigmoid(x @ params)
        diag = torch.diag(one_d)
        x_t = torch.t(x)
        denom = torch.inverse(x_t @ diag @ x)
        return factor*(basis @ (denom @ (x_t @ diag @ y)))

    return loss_fn


def train_and_return(feature_num, initial_value):
    params_min = torch.randn(x.shape[1], requires_grad=True,device="cuda")
    optim = Adam(params=[params_min], lr=0.05)
    iters = 0
    curr_error = 10000
    loss_min = loss_fn_generator(feature_num, minimize=True)
    while (curr_error > 1) and (iters < 100):
        optim.zero_grad()
        loss_res = loss_min(params_min)
        loss_res.backward()
        optim.step()
        curr_error = loss_res.item()
        iters += 1
    min_error = curr_error
    params_max = torch.randn(x.shape[1], requires_grad=True,device="cuda")
    optim = Adam(params=[params_max], lr=0.05)
    iters = 0
    curr_error = 10000
    loss_max = loss_fn_generator(feature_num, minimize=True)
    while (curr_error > 1) and (iters < 100):
        optim.zero_grad()
        loss_res = loss_max(params_max)
        loss_res.backward()
        optim.step()
        curr_error = loss_res.item()
        iters += 1
    max_error = curr_error
    if abs(max_error-initial_value) > abs(min_error-initial_value):
        return min_error, torch.diag(sigmoid(x @ params_min)).cpu().detach().numpy()
    return max_error, torch.diag(sigmoid(x @ params_min)).cpu().detach().numpy()


def initial_value(feature_num):
    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    basis = torch.tensor(basis_list, requires_grad=True).cuda()
    x_t = torch.t(x).cuda()
    denom = torch.inverse(x_t @ x).cuda()
    return (basis @ (denom @ (x_t @ y))).cuda()




errors_and_weights = []
for feature_num in range(x.shape[1]):
    full_dataset = initial_value(feature_num)
    try:
        error, _ = train_and_return(feature_num, full_dataset)
        errors_and_weights.append((error,feature_num))
        print(error, feature_num)
    except RuntimeError as e:
        print(e)
        continue
errors_sorted = sorted(errors_and_weights, key=lambda elem: abs(elem[0]), reverse=True)
print(errors_sorted[0])
print(compas_df.columns[errors_sorted[0][1]])


