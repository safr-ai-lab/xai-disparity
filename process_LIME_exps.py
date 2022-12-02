from constrained_opt import split_out_dataset
from lime_exp_func import LimeExpFunc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import json
import time


parser = argparse.ArgumentParser(description='Process exps and save to file')
parser.add_argument('seed', type=int)
args = parser.parse_args()
seed = args.seed

df = pd.read_csv('data/student/student_cleaned.csv')
target = 'G3'
sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
df_name = 'student'

new_cols = [col for col in df.columns if col != target] + [target]
df = df[new_cols]
train_df, test_df = train_test_split(df, test_size=0.5, random_state=seed)
x_train, y_train, sensitive_train = split_out_dataset(train_df, target, sensitive_features)
x_test, y_test, sensitive_test = split_out_dataset(test_df, target, sensitive_features)
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

