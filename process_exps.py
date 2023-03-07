from notions.lime_exp_func import LimeExpFunc
from notions.shap_exp_func import ShapExpFunc
from notions.grad_exp_func import GradExpFunc
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.model_selection import train_test_split
from aif360.datasets import BankDataset
import argparse
import pandas as pd
import json
import time
import sys
import torch
import torch.nn as nn

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
elif exp_method == 'grad':
    expFunc = GradExpFunc
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
    #sensitive_ds = dataset[f_sensitive].to_numpy()
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
#classifier = RandomForestClassifier(random_state=seed)
classifier = sklearn.linear_model.LogisticRegression(random_state=seed,max_iter=1000)
classifier.fit(x_train, y_train)

# else:
#     epochs = 200000
#     input_dim = x_train.shape[1] # Two inputs x1 and x2
#     output_dim = max(y_train)+1
#     learning_rate = 0.01
#
#     model = LogisticRegression(input_dim,output_dim)
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
#     X_train, X_test = torch.Tensor(x_train),torch.Tensor(x_test)
#     Y_train, Y_test = torch.Tensor(y_train),torch.Tensor(y_test)

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

