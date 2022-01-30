import torch
from torch.special import expit as sigmoid
from torch.optim import SGD

x = torch.tensor([[1.,0.],[0.,1.], [1.,1.]], requires_grad=True)
y = torch.tensor([[1.25],[0.58], [2.47]], requires_grad=True)
feature_num = 0



def loss_fn_generator(feature_num):
    basis_list = [[0. for _ in range(x.shape[1])]]
    basis_list[0][feature_num] = 1.
    basis = torch.tensor(basis_list, requires_grad=True)

    def loss_fn(params):
        one_d = sigmoid(x @ params)
        diag = torch.diag(one_d)
        x_t = torch.t(x)
        denom = torch.inverse(x_t @ diag @ x)
        return basis @ (denom @ (x_t @ diag @ y))

    return loss_fn


params = torch.randn(x.shape[1], requires_grad=True)
optim = SGD(params=[params], lr=0.05)
errors_and_weights = []
for feature_num in range(x.shape[1]):
    iters = 0
    curr_error = 10000
    loss = loss_fn_generator(feature_num)
    while (curr_error > 1) and (iters<100):
        optim.zero_grad()
        loss_res = loss(params)
        loss_res.backward()
        optim.step()
        curr_error = loss_res.data[0][0]
        iters += 1
    errors_and_weights.append((curr_error, torch.diag(sigmoid(x @ params)).data, feature_num))
errors_sorted = sorted(errors_and_weights, key=lambda elem: elem[0], reverse=True)
print(errors_sorted[0])
