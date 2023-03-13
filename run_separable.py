from notions.lime_imp_func import LimeImpFunc
from notions.shap_imp_func import ShapImpFunc
from notions.grad_imp_func import GradImpFunc
import pandas as pd
import time
import argparse
from datetime import datetime
import sys
import json
from constrained_opt import SeparableSolver
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    return x, y


parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--file', type=str, required=False)
args = parser.parse_args()
imp_method = args.method
imp_file = args.file

if imp_method == 'lime':
    impFunc = LimeImpFunc
elif imp_method == 'shap':
    impFunc = ShapImpFunc
elif imp_method == 'grad':
    impFunc = GradImpFunc
else:
    sys.exit('Exp method not recognized')


# User: Define dataset here
df = pd.read_csv('data/student/student_cleaned.csv')
target = 'G3'
t_split = .5
sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
df_name = 'student'

# Sorting columns so target is at end
new_cols = [col for col in df.columns if col != target] + [target]
df = df[new_cols]
f_sensitive = list(df.columns.get_indexer(sensitive_features))

# User: Define classifier here
seed = 0
train_df, test_df = train_test_split(df, test_size=t_split, random_state=seed)
x_train, y_train = split_out_dataset(train_df, target)
x_test, y_test = split_out_dataset(test_df, target)
classifier = RandomForestClassifier(random_state=seed)
classifier.fit(x_train, y_train)

# User: Define alpha range here
a = [.01, .05]


# Read in importances file if specified, else compute importances
imp_func_train = impFunc(classifier, x_train, y_train, seed)
imp_func_test = impFunc(classifier, x_test, y_test, seed)

try:
    with open(f'{imp_file}_train', 'r') as f:
        train_temp = list(map(json.loads, f))[0]
    for e_list in train_temp:
        imp_func_train.imps.append({int(k): v for k, v in e_list.items()})
    with open(f'{imp_file}_test', 'r') as f:
        test_temp = list(map(json.loads, f))[0]
    for e_list in test_temp:
        imp_func_test.imps.append({int(k): v for k, v in e_list.items()})
except:
    print("Importance file not specified or found...")
    print("Populating train importance values")
    imp_func_train.populate_imps()
    with open(f'input/imps/{df_name}_{imp_method}_seed{seed}_train', 'w') as fout:
        json.dump(imp_func_train.imps, fout)

    print("Populating test importance values")
    imp_func_test.populate_imps()
    with open(f'input/imps/{df_name}_{imp_method}_seed{seed}_test', 'w') as fout:
        json.dump(imp_func_test.imps, fout)


print("Running", df_name, ", Alphas =", a)
start = time.time()
solver = SeparableSolver(df, (x_train, y_train), (x_test, y_test), imp_func_train, imp_func_test,
                         f_sensitive, a, seed)
final_df = solver.extremize_imps_dataset()
print("Runtime:", '%.2f' % ((time.time() - start) / 3600), "Hours")
date = datetime.today().strftime('%m_%d')
final_df.to_csv(f'output/{df_name}_{imp_method}_output_{date}_alpha{a}.csv')