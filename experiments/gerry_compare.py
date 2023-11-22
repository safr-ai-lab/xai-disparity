import gerryfair
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import json
import numpy as np
import warnings
from aif360.datasets import BankDataset
import torch
from torch.special import expit as sigmoid

def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    return x, y


def get_gerry_fid(data_path, sensitive_features, target, imp_file):
    dataset = pd.read_csv(data_path)
    seed = 0

    train_df, test_df = train_test_split(dataset, test_size=.2, random_state=seed)

    y_train = train_df[target].reset_index(drop=True)
    y_test = test_df[target].reset_index(drop=True)
    X_train = train_df.drop([target], axis=1).reset_index(drop=True)
    X_test = test_df.drop([target], axis=1).reset_index(drop=True)
    X_prime_train = train_df[sensitive_features].reset_index(drop=True)
    X_prime_test = test_df[sensitive_features].reset_index(drop=True)

    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    auditor = gerryfair.model.Auditor(X_prime_test, y_test, 'FP')
    [violated_group, fairness_violation] = auditor.audit(predictions)

    test_imps = []
    with open(imp_file, 'r') as f:
        test_temp = list(map(json.loads, f))[0]
    for e_list in test_temp:
        test_imps.append({int(k): v for k, v in e_list.items()})

    results_df = pd.DataFrame()
    features = []
    sizes = []
    FDs = []
    Fss = []
    avg_diffs = []
    mags = []

    for f in range(len(X_test.columns)):
        features.append(X_test.columns[f])
        overall = 0
        sub = 0
        for i in range(X_test.shape[0]):
            overall += test_imps[i][f]
            if violated_group[i] == 1:
                sub += test_imps[i][f]
        FDs.append(overall / X_test.shape[0])
        Fss.append(sub / sum(violated_group))
        avg_diffs.append(abs(overall / X_test.shape[0] - sub / sum(violated_group)))
        sizes.append(np.mean(violated_group))
        mags.append(abs(np.log10(abs((sub / sum(violated_group)) / ((overall + .00001) / X_test.shape[0])))))

    row = pd.DataFrame({"Feature": features, "Size": sizes, "avg(F(D))": FDs,
                        "avg(F(S))": Fss, "avg diff": avg_diffs, "Magnitude Change": mags})
    results_df = results_df.append(row)
    return results_df.sort_values('avg diff', key=abs, ascending=False)

dataset = '../data/compas/compas_recid.csv'
target = 'two_year_recid'
sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
imp_file = '../input/imps/compas_recid_shap_seed0_test'

# dataset = BankDataset().convert_to_dataframe()[0]
# target = 'y'
# sensitive_features = ['age', 'marital=married', 'marital=single', 'marital=divorced']
# imp_file = '../input/imps/bank_test_lime_seed0'

# dataset = '../data/folktables/ACSIncome_MI_2018_new.csv'
# target = 'PINCP'
# sensitive_features = ['AGEP', 'SEX', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0', 'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0',
#                       'RAC1P_3.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0', 'RAC1P_9.0']
# imp_file = '../input/imps/folktables_test_shap_seed0'

result = get_gerry_fid(dataset, sensitive_features, target, imp_file)

print(result)
