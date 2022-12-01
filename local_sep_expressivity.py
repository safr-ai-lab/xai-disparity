from reg_oracle import ZeroPredictor, ExpPredictor, RegOracle
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Union, Callable, NewType
import pandas as pd
import numpy as np
from aif360.datasets import CompasDataset, BankDataset
import re
import time
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('--dummy', action='store_true')
args = parser.parse_args()
dummy = args.dummy


class LimeExpFunc:
    def __init__(self, classifier, dataset, seed):
        self.classifier = classifier
        self.dataset = dataset
        self.exps = []
        self.lime_exp = LimeTabularExplainer(self.dataset, random_state=seed)

    # Populate exps with expressivity dictionaries
    # exps[n][i] returns expressivity of feature i in datapoint n
    def populate_exps(self):
        i = 0
        for row in self.dataset:
            #print('Computing ', i)
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


ExpFuncGenType = NewType("ExpFuncGenType", Callable[[np.ndarray, int], Callable[[np.ndarray], float]])

def fit_one_side(dataset, exps, minimize=False):
    left_predictor = ZeroPredictor()
    right_predictor = ExpPredictor()
    right_predictor.fit(dataset, exps)
    # flip the predictors
    reg_oracle = RegOracle(left_predictor, right_predictor, minimize=minimize)
    return reg_oracle

def fit_exps_dataset(dataset: np.ndarray, feature_num: int, exp_func: ExpFuncGenType, minimize=False):
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

def partial_dataset_expressivity(exp_func, feature_num, membership):
    """
    Computes expressivity over part of dataset that is member of group
    :param exp_func: cost expressivity class used
    :param feature_num: feature we are intersted in
    :param membership: binary array of length equal to exp_fun.exps. Denotes membership in group
    :return: total expressivity over these rows
    """
    total = 0
    for i in range(len(exp_func.exps)):
        total += exp_func.exps[i][feature_num] * membership[i]
    return total

def split_out_dataset(dataset, target_column, f_sensitive):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    sensitive_ds = dataset[f_sensitive].to_numpy()
    return x, y, sensitive_ds

def extremize_exps_dataset(dataset: pd.DataFrame, exp_func_type: ExpFuncGenType, target_column: str, f_sensitive: list, seed: int):
    # train test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    train_x, train_y, sensitive_train = split_out_dataset(train_df, target, f_sensitive)
    test_x, test_y, sensitive_test = split_out_dataset(test_df, target, f_sensitive)
    classifier = RandomForestClassifier(random_state=seed)
    classifier.fit(train_x, train_y)

    exp_func = exp_func_type(classifier, train_x, seed)
    print("Populating train expressivity values")
    exp_func.populate_exps()

    exp_func_test = exp_func_type(classifier, test_x, seed)
    print("Populating test expressivity values")
    exp_func_test.populate_exps()

    # numpy_ds is now train_x/test_x
    # sensitive_ds is now sensitive_train/sensitive_test
    # numpy_ds = dataset.drop(target_column, axis=1).to_numpy()
    # sensitive_ds = dataset[f_sensitive].to_numpy()

    out_df = pd.DataFrame(columns=['Feature', 'F(D)', 'max(F(S))', 'Difference', 'Subgroup Size', 'Subgroup Coefficients',
                                   'Direction', 'F(D)_train', 'max(F(S))_train', 'Difference_train', 'Subgroup Size_train'])

    for feature_num in range(len(train_x[0])):
        total_train = full_dataset_expressivity(exp_func, feature_num)
        max_pred, max_exp = fit_exps_dataset(train_x, feature_num, exp_func, minimize=False)
        min_pred, min_exp = fit_exps_dataset(train_x, feature_num, exp_func, minimize=True)
        if abs(max_exp-total_train) > abs(min_exp-total_train):
            furthest_exp_train = max_exp
            predictions_train = max_pred
            direction = 'maximize'
        else:
            furthest_exp_train = min_exp
            predictions_train = min_pred
            direction = 'minimize'
        subgroup_size_train = predictions_train.count(1)/len(predictions_train)

        # Train logistic regression model on the classification of the points
        if len(set(predictions_train)) == 1:
            params_with_labels = {f: 0 for f in f_sensitive}
            print("zeros")
            subgroup_model = ZeroPredictor()
        else:
            subgroup_model = LogisticRegression(solver='lbfgs', max_iter=200, random_state=seed).fit(sensitive_train, predictions_train)
            params = subgroup_model.coef_[0]
            print(params)
            params_with_labels = {dataset[f_sensitive].columns[i]: float(param) for (i, param) in enumerate(params)}

        total = full_dataset_expressivity(exp_func_test, feature_num)

        predictions_test = subgroup_model.predict(sensitive_test)
        subgroup_size = np.sum(predictions_test)/len(predictions_test)

        furthest_exp = partial_dataset_expressivity(exp_func_test, feature_num, predictions_test)

        out_df = pd.concat([out_df, pd.DataFrame.from_records([{'Feature': dataset.columns[feature_num],
                                                                'F(D)': total,
                                                                'max(F(S))': furthest_exp,
                                                                'Difference': abs(furthest_exp - total),
                                                                'Subgroup Coefficients': params_with_labels,
                                                                'Subgroup Size': subgroup_size,
                                                                'Direction': direction,
                                                                'F(D)_train': total_train,
                                                                'max(F(S))_train': furthest_exp_train,
                                                                'Difference_train': abs(furthest_exp_train - total_train),
                                                                'Subgroup Size_train': subgroup_size_train}])])
    return out_df


def run_system(df, target, sensitive_features, df_name, dummy=False):
    if dummy:
        df[target] = df[target].sample(frac=1).values
        df_name = 'dummy_' + df_name

    # Sort columns so target is at end
    new_cols = [col for col in df.columns if col != target] + [target]
    df = df[new_cols]

    seeds = [0]
    for s in seeds:
        np.random.seed(s)
        print("Running", df_name, ", Seed =", s)
        start = time.time()

        out = extremize_exps_dataset(dataset=df, exp_func_type=LimeExpFunc, target_column=target,
                                     f_sensitive=sensitive_features, seed=s)
        date = datetime.today().strftime('%m_%d')
        out.to_csv(f'output/sep/{df_name}_LIME_output_seed{s}_{date}.csv')
        print("Runtime:", '%.2f'%((time.time()-start)/3600), "Hours")
    return 1


df = pd.read_csv('data/student/student_cleaned.csv')
target = 'G3'
sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
df_name = 'student'
run_system(df, target, sensitive_features, df_name, dummy)

# df = CompasDataset().convert_to_dataframe()[0]
# target = 'two_year_recid'
# sensitive_features = ['age','race','sex','age_cat=25 - 45','age_cat=Greater than 45','age_cat=Less than 25']
# df_name = 'compas'
# run_system(df, target, sensitive_features, df_name, dummy)

# df = BankDataset().convert_to_dataframe()[0]
# target = 'y'
# sensitive_features = ['age', 'marital=married', 'marital=single', 'marital=divorced']
# df_name = 'bank'
# run_system(df, target, sensitive_features, df_name, dummy)
#
# df = pd.read_csv('data/folktables/ACSIncome_MI_2018_sampled.csv')
# target = 'PINCP'
# sensitive_features = ['AGEP', 'SEX', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0', 'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0',
#                       'RAC1P_3.0', 'RAC1P_4.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0', 'RAC1P_9.0']
# df_name = 'folktables'
# run_system(df, target, sensitive_features, df_name, dummy)



