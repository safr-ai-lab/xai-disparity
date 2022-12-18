#from constrained_opt import split_out_dataset
from lime_exp_func import LimeExpFunc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from aif360.datasets import BankDataset
import argparse
import pandas as pd
import json
import time

def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    #sensitive_ds = dataset[f_sensitive].to_numpy()
    return x, y


# parser = argparse.ArgumentParser(description='Process exps and save to file')
# parser.add_argument('seed', type=int)
# args = parser.parse_args()
seed = 0

# df = pd.read_csv('data/student/student_cleaned.csv')
# target = 'G3'
# sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
# df_name = 'student'

# df = pd.read_csv('data/compas/compas_cleaned.csv')
# target = 'two_year_recid'
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
# df_name = 'compas_recid'

# df = pd.read_csv('data/compas/compas_cleaned_decile.csv')
# target = 'decile_score'
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']
# df_name = 'compas_decile'

df = BankDataset().convert_to_dataframe()[0]
target = 'y'
sensitive_features = ['age', 'marital=married', 'marital=single', 'marital=divorced']
df_name = 'bank'

# df = pd.read_csv('data/folktables/ACSIncome_MI_2018_new.csv')
# target = 'PINCP'
# sensitive_features = ['AGEP', 'SEX', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0', 'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0',
#                       'RAC1P_3.0', 'RAC1P_4.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0', 'RAC1P_9.0']
# df_name = 'folktables'

print('starting')
new_cols = [col for col in df.columns if col != target] + [target]
df = df[new_cols]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
x_train, y_train = split_out_dataset(train_df, target)
x_test, y_test = split_out_dataset(test_df, target)
print('training classifier')
classifier = RandomForestClassifier(random_state=seed)
classifier.fit(x_train, y_train)

start = time.time()
exp_func_train = LimeExpFunc(classifier, x_train, seed)
print("Populating train expressivity values")
exp_func_train.populate_exps()
print("runtime train: ", time.time()-start)

start = time.time()
exp_func_test = LimeExpFunc(classifier, x_test, seed)
print("Populating test expressivity values")
exp_func_test.populate_exps()
print("runtime test: ", time.time()-start)

with open(f'data/exps/{df_name}_train_seed{seed}', 'w') as fout:
    json.dump(exp_func_train.exps, fout)

with open(f'data/exps/{df_name}_test_seed{seed}', 'w') as fout:
    json.dump(exp_func_test.exps, fout)

