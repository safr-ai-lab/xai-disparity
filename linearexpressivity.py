import torch
import pandas as pd
import numpy as np
from torch.special import expit as sigmoid
from torch.optim import Adam
from sklearn.model_selection import train_test_split

class NonseparableSolver:
    def __init__(self, df, target, sensitive_features, df_name, t_split, useCUDA, flatval, seed):
        self.df = df
        self.target = target
        self.sensitive_features = sensitive_features
        self.df_name = df_name
        self.t_split = t_split
        self.useCUDA = useCUDA
        self.flatval = flatval
        self.seed = seed

    def loss_fn_generator(self, x_0: torch.Tensor, y: torch.Tensor, feature_num: int, sensitives: torch.Tensor,
                          alpha: list, minimize: bool):
        """
        Factory for the loss function that pytorch runs will be optimizing in WLS
        :param x_0: the data tensor with intercept column
        :param y: the target tensor
        :param initial_val: expressivity over full dataset
        :param feature_num: Which feature in the data do we care about
        :param sensitives: tensor representing sensitive features
        :param alpha: desired subgroup size
        :param minimize: Boolean -- are we minimizing or maximizing
        :return: a loss function for our particular WLS problem.
        """
        x = self.remove_intercept_column(x_0)

        basis_list = [[0. for _ in range(x.shape[1])]]
        basis_list[0][feature_num] = 1.
        flat_list = [self.flatval for _ in range(x.shape[1])]

        if self.useCUDA:
            basis = torch.tensor(basis_list, requires_grad=True).cuda()
            flat = torch.tensor(flat_list, requires_grad=True).cuda()
        else:
            basis = torch.tensor(basis_list, requires_grad=True)
            flat = torch.tensor(flat_list, requires_grad=True)

        if minimize:
            sign = 1
        else:
            sign = -1

        # Look into derivation of gradient by hand and implementing it here instead.
        def loss_fn(params):
            # only train using sensitive features
            one_d = sigmoid(x_0 @ (sensitives * params))
            diag = torch.diag(one_d)
            x_t = torch.t(x)
            denom = torch.inverse((x_t @ diag @ x) + torch.diag(flat))
            coefficient = basis @ (denom @ (x_t @ diag @ y))

            size = torch.sum(one_d)/x.shape[0]
            size_penalty = 100000*(max(alpha[0]-size, 0) + max(size-alpha[1], 0))

            return size_penalty + .1 * sign * coefficient

        return loss_fn


    def train_and_return(self, x: torch.Tensor, y: torch.Tensor, feature_num: int, f_sensitive: list, alpha: list, minimize: bool):
        """
        Given an x, y, feature num, and the expressivity over the whole dataset,
        returns the differential expressivity and maximal subset for that feature
        :param x: The data tensor
        :param y: the target tensor
        :param feature_num: which feature to optimize, int.
        :param initial_val: What the expressivity over the whole dataset for the feature is.
        :param f_sensitive: indices of sensitive features
        :param alpha: target subgroup size
        :return: the differential expressivity and maximal subset weights.
        """
        niters = 1000
        # Set seed to const value for reproducibility
        torch.manual_seed(self.seed)
        s_list = [0. for _ in range(x.shape[1])]
        for f in f_sensitive:
            s_list[f] = 1.
        if self.useCUDA:
            sensitives = torch.tensor(s_list, requires_grad=True).cuda()
            params_max = torch.randn(x.shape[1], requires_grad=True, device="cuda")
        else:
            sensitives = torch.tensor(s_list, requires_grad=True)
            params_max = torch.randn(x.shape[1], requires_grad=True)

        optim = Adam(params=[params_max], lr=0.05)
        iters = 0
        curr_error = 10000
        s_record = []
        p_record = []
        loss_max = self.loss_fn_generator(x, y, feature_num, sensitives, alpha, minimize)
        while iters < niters:
            optim.zero_grad()
            loss_res = loss_max(params_max)
            loss_res.backward()
            optim.step()
            curr_error = loss_res.item()

            params_temp = sensitives * params_max
            size = np.sum((sigmoid(x @ params_temp)).cpu().detach().numpy())/x.shape[0]
            s_record.append(size)

            p_record.append(self.final_value(x, y, params_temp.cpu().detach().numpy(), feature_num)[0])
            iters += 1
        params_max = sensitives * params_max
        max_error = curr_error * -1
        assigns = (sigmoid(x @ params_max)).cpu().detach().numpy()
        #print('final train size: ',torch.sum(sigmoid(x @ params_max))/x.shape[0])
        return max_error, assigns, params_max.cpu().detach().numpy(), s_record, p_record

    def is_valid(self, assigns, alpha):
        """
        return: 1 if valid, 0 if invalid
        """
        size = np.mean(assigns)
        if size-alpha[0]>0 and alpha[1]-size>0:
            valid = 1
        else:
            valid = 0
        return valid

    def initial_value(self, x: torch.Tensor, y: torch.Tensor, feature_num: int) -> float:
        """
        Given a dataset, target, and feature number, returns the expressivity of that feature over the dataset.
        :param x: the data tensor
        :param y: the target tensor
        :param feature_num: the feature to test
        :return: the float value of expressivity over the dataset
        """
        basis_list = [[0. for _ in range(x.shape[1])]]
        basis_list[0][feature_num] = 1.
        flat_list = [self.flatval for _ in range(x.shape[1])]

        if self.useCUDA:
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

    def final_value(self, x_0: torch.Tensor, y: torch.Tensor, params: torch.Tensor, feature_num: int):
        """
        Given a defined subgroup function, returns the expressivity over the test data set
        :param x_0: the test data tensor
        :param y: the test target tensor
        :param params: tensor with coefficients defining the subgroup
        :param feature_num: the feature to test
        :return: the float value of expressivity over the dataset and subgroup assignments
        """
        x = self.remove_intercept_column(x_0)
        basis_list = [[0. for _ in range(x.shape[1])]]
        basis_list[0][feature_num] = 1.
        flat_list = [self.flatval for _ in range(x.shape[1])]

        if self.useCUDA:
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
        return (basis @ (denom @ (x_t @ diag @ y))).cpu().detach().numpy()[0], one_d.cpu().detach().numpy()

    def find_extreme_subgroups(self, dataset: pd.DataFrame, alpha: list, target_column: str, f_sensitive: list, t_split: float):
        """
        Given a dataset, finds the differential expressivity and maximal subset over all features.
        Saves that subset to a file.
        :param dataset: the pandas dataframe to use
        :param alpha: desired subgroup size
        :param target_column:  Which column in that dataframe is the target.
        :param f_sensitive: Which features are sensitive characteristics
        :return:  N/A.  Logs results.
        """
        out_df = pd.DataFrame()

        train_df, test_df = train_test_split(dataset, test_size=t_split, random_state=self.seed)

        if self.useCUDA:
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
            x_train_ni = self.remove_intercept_column(x_train)
            total_exp_train = self.initial_value(x_train_ni, y_train, feature_num)
            try:
                _, assigns_min, params_min, s_record_min, p_record_min = self.train_and_return(x_train, y_train, feature_num,
                                                                                          f_sensitive, alpha, minimize=True)
                _, assigns_max, params_max, s_record_max, p_record_max = self.train_and_return(x_train, y_train, feature_num,
                                                                                          f_sensitive, alpha, minimize=False)

                furthest_exp_min, _ = self.final_value(x_train, y_train, params_min, feature_num)
                furthest_exp_max, _ = self.final_value(x_train, y_train, params_max, feature_num)
                valid_min = self.is_valid(assigns_min, alpha)
                valid_max = self.is_valid(assigns_max, alpha)
                if valid_max * abs(furthest_exp_max - total_exp_train) > valid_min * abs(furthest_exp_min - total_exp_train):
                    assigns_train, params, s_record, p_record = assigns_max, params_max, s_record_max, p_record_max
                    furthest_exp_train = furthest_exp_max
                else:
                    assigns_train, params, s_record, p_record = assigns_min, params_min, s_record_min, p_record_min
                    furthest_exp_train = furthest_exp_min
                subgroup_size_train = np.mean(assigns_train)

                x_test_ni = self.remove_intercept_column(x_test)
                total_exp = self.initial_value(x_test_ni, y_test, feature_num)
                furthest_exp, assigns = self.final_value(x_test, y_test, params, feature_num)
                subgroup_size = np.mean(assigns)
                errors_and_weights.append((furthest_exp, feature_num))
                print(furthest_exp, feature_num)
                params_with_labels = {dataset.columns[i]: float(param) for (i, param) in enumerate(params)}
                out_df = pd.concat([out_df, pd.DataFrame.from_records([{'Feature': dataset.columns[feature_num],
                                                                        'Alpha': alpha,
                                                                        'F(D)': total_exp,
                                                                        'max(F(S))': furthest_exp,
                                                                        'Difference': abs(furthest_exp - total_exp),
                                                                        'Percent Change': 100*abs(furthest_exp - total_exp)/total_exp,
                                                                        'Subgroup Coefficients': params_with_labels,
                                                                        'Subgroup Size': subgroup_size,
                                                                        'F(D)_train': total_exp_train,
                                                                        'max(F(S))_train': furthest_exp_train,
                                                                        'Difference_train': abs(furthest_exp_train - total_exp_train),
                                                                        'Percent Change_train': 100*abs(furthest_exp_train - total_exp_train)/total_exp_train,
                                                                        'Subgroup Size_train': subgroup_size_train,
                                                                        'Size record': s_record,
                                                                        'WLS Penalties': p_record}])])
            except RuntimeError as e:
                print(e)
                continue
        errors_sorted = sorted(errors_and_weights, key=lambda elem: abs(elem[0]), reverse=True)
        print(errors_sorted[0])
        print(dataset.columns[errors_sorted[0][1]])

        return out_df

    def remove_intercept_column(self, x):
        mask = torch.arange(0, x.shape[1] - 1)
        x_cpu = x.cpu()
        out = torch.index_select(x_cpu, 1, mask)
        if self.useCUDA:
            out = out.cuda()
        return out

