from lime_exp_func import LimeExpFunc
from shap_exp_func import ShapExpFunc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from aif360.datasets import BankDataset
import argparse
import pandas as pd
import json
import time
import sys

parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('exp_method', type=str)
parser.add_argument('seed', type=int, default=0)
args = parser.parse_args()
exp_method = args.exp_method
seed = args.seed

if exp_method == 'lime':
    expFunc = LimeExpFunc
elif exp_method == 'shap':
    expFunc = ShapExpFunc
else:
    sys.exit('Exp method not recognized')

def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    #sensitive_ds = dataset[f_sensitive].to_numpy()
    return x, y

# df = pd.read_csv('data/student/student_cleaned.csv')
# target = 'G3'
# t_split = .5
# sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
# df_name = 'student'

df = pd.read_csv('data/compas/compas_recid.csv')
target = 'two_year_recid'
t_split = .5
sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
df_name = 'compas_recid'

# df = pd.read_csv('data/compas/compas_cleaned_decile.csv')
# target = 'decile_score'
# t_split = .5
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
# df_name = 'compas_decile'

# df = pd.read_csv('data/compas/compas_decile_stripped.csv')
# target = 'decile_score'
# t_split = .5
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
# df_name = 'compas_decile'

# df = BankDataset().convert_to_dataframe()[0]
# target = 'y'
# t_split = .5
# sensitive_features = ['age', 'marital=married', 'marital=single', 'marital=divorced']
# df_name = 'bank'

# df = pd.read_csv('data/folktables/ACSIncome_MI_2018_new.csv')
# target = 'PINCP'
# t_split = .5
# sensitive_features = ['AGEP', 'SEX', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0', 'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0',
#                       'RAC1P_3.0', 'RAC1P_4.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0', 'RAC1P_9.0']
# df_name = 'folktables'

print('starting', df_name)
new_cols = [col for col in df.columns if col != target] + [target]
df = df[new_cols]
train_df, test_df = train_test_split(df, test_size=t_split, random_state=seed)
x_train, y_train = split_out_dataset(train_df, target)
x_test, y_test = split_out_dataset(test_df, target)
print('training classifier')
#classifier = RandomForestClassifier(random_state=seed)
classifier = LogisticRegression(random_state=seed,max_iter=1000)
classifier.fit(x_train, y_train)

start = time.time()
exp_func_train = expFunc(classifier, x_train, seed)
print("Populating train expressivity values")
exp_func_train.populate_exps()
print("runtime train: ", time.time()-start)

with open(f'data/exps/{df_name}_train_{exp_method}LR_seed{seed}', 'w') as fout:
    json.dump(exp_func_train.exps, fout)

start = time.time()
exp_func_test = expFunc(classifier, x_test, seed)
print("Populating test expressivity values")
exp_func_test.populate_exps()
print("runtime test: ", time.time()-start)

with open(f'data/exps/{df_name}_test_{exp_method}LR_seed{seed}_lr', 'w') as fout:
    json.dump(exp_func_test.exps, fout)

