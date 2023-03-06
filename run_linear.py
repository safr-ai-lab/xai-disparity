import torch
import pandas as pd
import numpy as np
import time
from datetime import datetime
import argparse
from linearexpressivity import NonseparableSolver

parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
useCUDA = args.cuda

# Enable GPU if desired. Sometimes returns false values
if useCUDA:
    torch.cuda.set_device('cuda:0')
else:
    torch.device('cuda:0')

df = pd.read_csv('data/student/student_cleaned.csv')
target = 'G3'
t_split = .5
sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']
df_name = 'student'
seed = 0
# regularization term of .0001 works well for smaller datasets (<5k data points)
# For larger datasets, CUDA can sometimes return funky results
# Values ~.01 don't substantially affect results and work for larger datasets
flatval = .0001

# Add intercept column at the end
df['Intercept'] = np.ones(df.shape[0])

# Sort columns so target is at end
new_cols = [col for col in df.columns if col != target] + [target]
df = df[new_cols]

# Get indices of sensitive features. Append a 1 for intercept
f_sensitive = list(df.columns.get_indexer(sensitive_features))
f_sensitive.append(df.shape[1]-2)

print(df.shape[1])
final_df = pd.DataFrame()

start = time.time()
solver = NonseparableSolver(df, target, sensitive_features, df_name, t_split, useCUDA, flatval, seed)
#alphas = [[.01,.05],[.05,.1],[.1,.15],[.15,.2]]
alphas = [[.1,.15]]
for a in alphas:
    print("Running", df_name, ", Alpha =", a)
    out = solver.find_extreme_subgroups(df, alpha=a, target_column=target, f_sensitive=f_sensitive, t_split=t_split)
    final_df = pd.concat([final_df, out])

date = datetime.today().strftime('%m_%d')
fname = f'output/{df_name}_output_{date}.csv'
final_df.to_csv(fname)
print("Runtime:", '%.2f'%((time.time()-start)/3600), "Hours")

