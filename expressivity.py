from reg_oracle import ZeroPredictor, CostPredictor, RegOracle
import numpy as np
from typing import Any, Callable, NewType
from random import uniform


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
    costs = []
    rows = []
    cost_func = cost_func_gen(dataset, feature_num)
    for row in dataset:
        costs.append(cost_func(row))
        rows.append(row[:-1])

    predictor = fit_one_side(np.array(rows), costs, minimize=minimize)
    predictions, cost = predictor.predict(dataset)
    return predictions, cost


def extremize_costs_dataset(dataset: np.ndarray, cost_func_gen: CostFuncGenType):
    max_costs = np.array(
        [(feature_num, fit_costs_dataset(dataset, feature_num, cost_func_gen, minimize=False)) for feature_num in range(dataset.shape[1]-1)])
    min_costs = np.array(
        [(feature_num, fit_costs_dataset(dataset, feature_num, cost_func_gen, minimize=True)) for feature_num in range(dataset.shape[1]-1)])
    feature_num, (max_predictions, max_cost) = sorted(max_costs, key=lambda x: x[1][1], reverse=True)[0]
    feature_num, (min_predictions, min_cost) = sorted(min_costs, key=lambda x: x[1][1])[0]
    if abs(max_cost) >= abs(min_cost):
        return feature_num, max_predictions, max_cost
    return feature_num, min_predictions, min_cost


def means_cost_func(dataset: np.ndarray, feature_num: int) -> Callable[[np.ndarray], float]:
    probability_f_i = dataset[dataset[..., feature_num] == 1].shape[0] / dataset.shape[0]
    return lambda row: 2 * (row[feature_num] * probability_f_i - probability_f_i) * row[-1]


means_extremize = lambda dataset: extremize_costs_dataset(dataset, means_cost_func)



def coinflip(threshold:float) -> int:
    return int(uniform(0, 1)<threshold)

num_list = 4000

poor_list = []
for x in range(num_list):
    poor_elem = [0, coinflip(0)]
    if poor_elem[1] == 1:
        poor_elem.append(coinflip(0.8))
    else:
        poor_elem.append(coinflip(0.4))
    poor_list.append(poor_elem)

rich_list = []
for x in range(num_list):
    rich_elem = [1, coinflip(0.5)]
    if rich_elem[1] == 1:
        rich_elem.append(coinflip(0.8))
    else:
        rich_elem.append(coinflip(0.4))
    rich_list.append(rich_elem)

poor_list.extend(rich_list)

poor_list = np.array(poor_list)

feature_num, predictions, cost = means_extremize(poor_list)

predictions = poor_list[np.array(predictions)]
print(set((elem[0], elem[1], elem[2]) for elem in predictions))

print(feature_num, cost)

