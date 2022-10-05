from reg_oracle import ZeroPredictor, ExpPredictor, RegOracle
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Union, Callable, NewType
import pandas as pd
import numpy as np
from aif360.datasets import CompasDataset, BankDataset
import re
import time


class LimeExpFunc:
    def __init__(self, classifier, dataset):
        self.classifier = classifier
        self.dataset = dataset
        self.exps = []
        self.lime_exp = LimeTabularExplainer(self.dataset)

    # Populate exps with expressivity dictionaries
    # exps[n][i] returns expressivity of feature i in datapoint n
    def populate_exps(self):
        i = 0
        for row in self.dataset:
            print('Computing ', i)
            exp_i = self.lime_exp.explain_instance(row, self.classifier.predict_proba, num_features=row.shape[0]).as_list()
            exp_dict = {}
            # Clean up LIME output and return dict with key=feature, value=expressivity
            for e in exp_i:
                parts = re.split(r"[\<=\<\>=\>]", e[0].replace(" ", ""))
                for p in parts:
                    if '.' not in p and len(p)>0:
                        feature = int(p)
                exp_dict[feature] = e[1]
            self.exps.append(exp_dict)
            i += 1

    # Given feature and row, return the computed expressivities
    def get_exp(self, row, feature):
        if len(self.exps) == 0:
            print("Populating feature expressivities")
            self.populate_exps()
        return self.exps[row][feature]


ExpFuncGenType = NewType("CostFuncGenType", Callable[[np.ndarray, int], Callable[[np.ndarray], float]])

def fit_one_side(dataset, exps, minimize=False):
    left_predictor = ExpPredictor()
    right_predictor = ZeroPredictor()
    left_predictor.fit(dataset, exps)
    reg_oracle = RegOracle(left_predictor, right_predictor, minimize=minimize)
    return reg_oracle

def fit_exps_dataset(dataset: np.ndarray, feature_num: int,
                      exp_func: ExpFuncGenType,
                      minimize=False):
    exps = []
    for i in range(len(dataset)):
        exps.append(exp_func.get_exp(i, feature_num))

    predictor = fit_one_side(dataset, exps, minimize=minimize)
    predictions, exp = predictor.predict(dataset)
    return predictions, exp

def full_dataset_expressivity(exp_func, feature_num):
    total = 0
    for row in exp_func.exps:
        total += row[feature_num]
    return total


def extremize_exps_dataset(dataset: Union[np.ndarray, pd.DataFrame], exp_func, target_column):
    # Populate expressivities for the dataset
    print("Populating expressivity values")
    exp_func.populate_exps()

    numpy_ds = dataset.drop(target_column, axis=1).to_numpy()

    out_df = pd.DataFrame(columns=['Feature', 'F(D)', 'max(F(S))', 'Difference', 'Subgroup Size', 'Subgroup Coefficients'])

    for feature_num in range(len(numpy_ds[0])):
        total = full_dataset_expressivity(exp_func, feature_num)
        max_pred, max_exp = fit_exps_dataset(numpy_ds, feature_num, exp_func, minimize=False)
        min_pred, min_exp = fit_exps_dataset(numpy_ds, feature_num, exp_func, minimize=True)
        if abs(max_exp-total) > abs(min_exp-total):
            furthest_exp = max_exp
            predictions = max_pred
        else:
            furthest_exp = min_exp
            predictions = min_pred
        subgroup_size = predictions.count(1)/len(predictions)

        # Train logistic regression model on the classification of the points
        if len(set(predictions)) == 1:
            params = np.zeros(len(numpy_ds[0]))
        else:
            subgroup_model = LogisticRegression(solver='lbfgs', max_iter=200).fit(numpy_ds, predictions)
            params = subgroup_model.coef_[0]
        params_with_labels = {dataset.columns[i]: float(param) for (i, param) in enumerate(params)}

        out_df = pd.concat([out_df, pd.DataFrame.from_records([{'Feature': dataset.columns[feature_num],
                                                                'F(D)': total,
                                                                'max(F(S))': furthest_exp,
                                                                'Difference': abs(furthest_exp - total),
                                                                'Subgroup Coefficients': params_with_labels,
                                                                'Subgroup Size': subgroup_size}])])

    # max_exps = np.array(
    #     [(feature_num, fit_exps_dataset(numpy_ds, feature_num, exp_func, minimize=False),
    #       full_dataset_expressivity(exp_func, feature_num)) for feature_num in range(len(numpy_ds[0]))])
    # min_exps = np.array(
    #     [(feature_num, fit_exps_dataset(numpy_ds, feature_num, exp_func, minimize=True),
    #       full_dataset_expressivity(exp_func, feature_num)) for feature_num in range(len(numpy_ds[0]))])

    # feature_num, (max_predictions, max_exp), total = sorted(max_exps, key=lambda k: (k[1][1] - k[2]), reverse=True)[0]
    # feature_num, (min_predictions, min_exp), total = sorted(min_exps, key=lambda k: (k[1][1] - k[2]))[0]
    # if abs(max_exp) >= abs(min_exp):
    #     return feature_num, max_predictions, max_exp, total
    # return feature_num, min_predictions, min_exp, total

    return out_df


def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    return x, y

# df = CompasDataset().convert_to_dataframe()[0]
# target = 'two_year_recid'
# sensitive = ['age','race','sex','age_cat=25 - 45','age_cat=Greater than 45','age_cat=Less than 25']
# df_name = 'compas'

df = pd.read_csv('data/student/student_cleaned.csv').head(10)
target = 'G3'
sensitive_features = ['sex_M', 'Pstatus_T', 'Dalc', 'Walc', 'health']
df_name = 'student'

# Sort columns so target is at end
new_cols = [col for col in df.columns if col != target] + [target]
df = df[new_cols]

classifier = RandomForestClassifier()
x, y = split_out_dataset(df, target)
classifier.fit(x, y)

start = time.time()
out = extremize_exps_dataset(df, LimeExpFunc(classifier, x), target_column=target)
out.to_csv("testrun.csv")
print("Runtime:", '%.2f'%((time.time()-start)/3600), "Hours")


