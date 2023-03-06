from notions.lime_imp_func import LimeImpFunc
from notions.shap_imp_func import ShapImpFunc
from notions.grad_imp_func import GradImpFunc
import pandas as pd
import time
import argparse
from datetime import datetime
import sys
from constrained_opt import SeparableSolver

parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('imp_method', type=str)
args = parser.parse_args()
imp_method = args.imp_method

if imp_method == 'lime':
    impFunc = LimeImpFunc
elif imp_method == 'shap':
    impFunc = ShapImpFunc
elif imp_method == 'grad':
    impFunc = GradImpFunc
else:
    sys.exit('Exp method not recognized')

df = pd.read_csv('data/student/student_cleaned.csv')
target = 'G3'
t_split = .5
sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
df_name = 'student'

# Sort columns so target is at end
new_cols = [col for col in df.columns if col != target] + [target]
df = df[new_cols]

f_sensitive = list(df.columns.get_indexer(sensitive_features))

a = [.01, .05]
print("Running", df_name, ", Alphas =", a)
start = time.time()
solver = SeparableSolver(df, target, sensitive_features, imp_method, df_name, t_split)
final_df = solver.extremize_imps_dataset(dataset=df, imp_func=impFunc, target_column=target,
                                  f_sensitive=f_sensitive, alphas=a, t_split=t_split)
print("Runtime:", '%.2f' % ((time.time() - start) / 3600), "Hours")
date = datetime.today().strftime('%m_%d')
final_df.to_csv(f'output/{df_name}_{imp_method}_output_{date}_alpha{a}.csv')