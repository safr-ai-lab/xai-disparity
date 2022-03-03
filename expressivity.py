from reg_oracle import ZeroPredictor, CostPredictor, RegOracle
import numpy as np
import pandas as pd
from typing import Union, Callable, NewType
from random import uniform
from lime.lime_tabular import LimeTabularExplainer

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


def total_cost(dataset, feature_num, cost_func_gen: CostFuncGenType):
    cost_func = cost_func_gen(dataset, feature_num)
    return sum(cost_func(row) for row in dataset)


def extremize_costs_dataset(dataset: Union[np.ndarray, pd.DataFrame], cost_func_gen: CostFuncGenType,
                            target_column=None):
    if target_column is None:
        max_costs = np.array(
            [(feature_num, fit_costs_dataset(dataset, feature_num, cost_func_gen, minimize=False),
              total_cost(dataset, feature_num, cost_func_gen)) for feature_num in range(dataset.shape[1] - 1)])
        min_costs = np.array(
            [(feature_num, fit_costs_dataset(dataset, feature_num, cost_func_gen, minimize=True),
              total_cost(dataset, feature_num, cost_func_gen)) for feature_num in range(dataset.shape[1] - 1)])
    else:
        numpy_ds = dataset.drop(target_column, axis=1).to_numpy()
        max_costs = np.array(
            [(feature_num, fit_costs_dataset(numpy_ds, feature_num, cost_func_gen, minimize=False),
              total_cost(numpy_ds, feature_num, cost_func_gen)) for feature_num in range(dataset.shape[1] - 1)])
        min_costs = np.array(
            [(feature_num, fit_costs_dataset(numpy_ds, feature_num, cost_func_gen, minimize=True),
              total_cost(numpy_ds, feature_num, cost_func_gen)) for feature_num in range(dataset.shape[1] - 1)])
    feature_num, (max_predictions, max_cost), total = sorted(max_costs, key=lambda x: (x[1][1] - x[2]), reverse=True)[0]
    feature_num, (min_predictions, min_cost), total = sorted(min_costs, key=lambda x: (x[1][1] - x[2]))[0]
    if abs(max_cost) >= abs(min_cost):
        return feature_num, max_predictions, max_cost - total, total
    return feature_num, min_predictions, min_cost - total, total


def means_cost_func(dataset: np.ndarray, feature_num: int) -> Callable[[np.ndarray], float]:
    probability_f_i = dataset[dataset[..., feature_num] == 1].shape[0] / dataset.shape[0]
    return lambda row: (row[feature_num] - probability_f_i) * row[-1] / (probability_f_i * (1 - probability_f_i))


def lime_cost_func_generator(classifier):
    def cost_func(dataset, feature_num):
        lime_exp = LimeTabularExplainer(dataset)

        def internal_cost(row):
            explanation = lime_exp.explain_instance(row, classifier.predict_proba, num_features=row.shape[1])
            return explanation.as_list()[feature_num][1]

        return internal_cost

    return cost_func


means_extremize = lambda dataset: extremize_costs_dataset(dataset, means_cost_func)


def coinflip(threshold: float) -> int:
    return int(uniform(0, 1) < threshold)


if __name__ == "__main__":

    num_list = 4000

    poor_list = []
    for x in range(num_list):
        poor_elem = [0, coinflip(0)]
        if poor_elem[1] == 1:
            poor_elem.append(coinflip(0.6))
        else:
            poor_elem.append(coinflip(0.6))
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

    feature_num, predictions, cost, total = means_extremize(poor_list)

    predictions = poor_list[np.array(predictions)]
    print(set((elem[0], elem[1], elem[2]) for elem in predictions))

    print(feature_num, cost, total)
