import torch
from torch.special import expit as sigmoid
from torch.nn.functional import one_hot
from torch.autograd import Function

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
        print(x_t @ diag @ y)
        print(denom @ (x_t @ diag @ y))
        """
        diag = torch.diag(one_d)
        denom = torch.inverse(x @ diag @ x_t)
        full_tensor = denom @ (x_t @ diag @ y)
        return basis @ full_tensor"""
        return basis @ (denom @ (x_t @ diag @ y))

    return loss_fn


loss = loss_fn_generator(0)
params=torch.tensor([1.,10.], requires_grad=True)
res = loss(params)
res.backward()

print(params.grad)
