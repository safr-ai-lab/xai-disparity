from reg_oracle import ZeroPredictor, CostPredictor, RegOracle
import numpy as np
from typing import Callable, NewType


CostFuncGenType = NewType("CostFuncGenType", Callable[[np.ndarray, int], Callable[[np.ndarray], float]])


def fit_one_side(dataset, costs, minimize=False):
    left_predictor = CostPredictor()
    right_predictor = ZeroPredictor()
    left_predictor.fit(dataset, costs)
    reg_oracle = RegOracle(left_predictor, right_predictor, minimize=minimize)
    return reg_oracle


def fit_costs_dataset(dataset: np.ndarray, feature_num: int,
                      cost_func_gen: CostFuncGenType,
                      minimize=False):
    costs = np.vectorize(cost_func_gen(dataset, feature_num))(dataset)
    predictor = fit_one_side(dataset, costs, minimize=minimize)
    predictions, cost = predictor.predict(dataset)
    return predictions, cost


def extremize_costs_dataset(dataset: np.ndarray, cost_func_gen: CostFuncGenType):
    max_costs = np.array(
        [fit_costs_dataset(dataset, feature_num, cost_func_gen, minimize=False) for feature_num in range(dataset.shape[0])])
    min_costs = np.array(
        [fit_costs_dataset(dataset, feature_num, cost_func_gen, minimize=True) for feature_num in range(dataset.shape[0])])
    max_predictions, max_cost = sorted(max_costs, key=lambda x: x[1], reverse=True)[0]
    min_predictions, min_cost = sorted(min_costs, key=lambda x: x[1])[0]
    if abs(max_cost) >= abs(min_cost):
        return max_predictions, max_cost
    return min_predictions, min_cost


def means_cost_func(dataset: np.ndarray, feature_num: int) -> Callable[[np.ndarray], float]:
    probability_f_i = dataset[dataset[feature_num] == 1].shape[0] / dataset.shape[0]
    return lambda row: 2 * (row[feature_num] * probability_f_i - probability_f_i) * row[-1]


means_extremize = lambda dataset: extremize_costs_dataset(dataset, means_cost_func)