from reg_oracle import ZeroPredictor, CostPredictor, RegOracle
import numpy as np


def fit_one_side(dataset, costs, minimize = False):
    left_predictor = CostPredictor()
    right_predictor = ZeroPredictor()
    left_predictor.fit(dataset, costs)
    reg_oracle = RegOracle(left_predictor, right_predictor, minimize=minimize)
    return reg_oracle


def cost_for_row_and_feature(row, feature_num, probability_f_i):

    #(2(x_{i, f_i}*Pr(f_i)-Pr(f_i))y_i)
    cost = 2*(row[feature_num]*probability_f_i - probability_f_i)*row[-1]
    return cost


def fit_means_dataset(dataset, feature_num):
    probability_f_i = dataset[dataset[feature_num] == 1].shape[0]/dataset.shape[0]
    costs = np.vectorize(lambda row: cost_for_row_and_feature(row, feature_num, probability_f_i))(dataset)
    return fit_one_side(dataset, costs, False), fit_one_side(dataset, costs, True)

