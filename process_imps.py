from notions.lime_imp_func import LimeImpFunc
from notions.shap_imp_func import ShapImpFunc
from notions.grad_imp_func import GradImpFunc
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import json
import time
import sys
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('imp_method', type=str)
parser.add_argument('seed', type=int, default=0)
args = parser.parse_args()
imp_method = args.imp_method
seed = args.seed

if imp_method == 'lime':
    impFunc = LimeImpFunc
elif imp_method == 'shap':
    impFunc = ShapImpFunc
elif imp_method == 'grad':
    impFunc = GradImpFunc
else:
    sys.exit('Exp method not recognized')

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    return x, y

df = pd.read_csv('data/student/student_cleaned.csv')
target = 'G3'
t_split = .5
sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
df_name = 'student'

print('starting', df_name)
new_cols = [col for col in df.columns if col != target] + [target]
df = df[new_cols]
train_df, test_df = train_test_split(df, test_size=t_split, random_state=seed)
x_train, y_train = split_out_dataset(train_df, target)
x_test, y_test = split_out_dataset(test_df, target)
print('training classifier')
classifier = RandomForestClassifier(random_state=seed)
# classifier = sklearn.linear_model.LogisticRegression(random_state=seed,max_iter=1000)
# classifier.fit(x_train, y_train)


start = time.time()
imp_func_train = impFunc(classifier, x_train, seed)
print("Populating train importance values")
imp_func_train.populate_imps()
print("runtime train: ", time.time()-start)

with open(f'input/imps/{df_name}_{imp_method}_seed{seed}_train', 'w') as fout:
    json.dump(imp_func_train.imps, fout)

start = time.time()
imp_func_test = impFunc(classifier, x_test, seed)
print("Populating test importance values")
imp_func_test.populate_imps()
print("runtime test: ", time.time()-start)

with open(f'input/imps/{df_name}_{imp_method}_seed{seed}_test', 'w') as fout:
    json.dump(imp_func_test.imps, fout)

