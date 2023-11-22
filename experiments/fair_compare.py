import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import ast
import matplotlib.pyplot as plt
import numpy as np
import json
from aif360.datasets import BankDataset

def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    return x, y

def in_g(x, g, value):
    for feature in g:
        value += g[feature]*x[feature]
    if value > 0:
        return False
    else:
        return True

def get_E(scores, Y_pred, Y_true, r):
    count, trues = 0, 0
    acc_count, conf_count = 0, 0
    for i in range(len(scores)):
        if scores[i] >= r[0] and scores[i] < r[1]:
            count += 1
            trues += Y_true[i]
            if Y_pred[i] == Y_true[i]:
                acc_count += 1
            conf_count += scores[i]
    if count == 0:
        cals = 0
        acc = 0
        conf = 0
    else:
        cals = trues / count
        acc = acc_count / count
        conf = conf_count / count

    return cals, abs(acc - conf) * count / len(scores)

def zero_groups(df):
    print('Groups: ',df.shape[0],'| 0 Groups: ', df[df['Subgroup Size']<.001].shape[0], '| 0 Features:', df[(df['F(D)']==0) & (df['F(D)_train']==0)].shape[0])
    print('Percent Zero Groups: ', (df[df['Subgroup Size']<.001].shape[0]-df[(df['F(D)']==0) & (df['F(D)_train']==0)].shape[0])/df.shape[0])
    return 1

def clean_sep(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'],axis=1)
    if 'Percent Change' not in df.columns:
        df['Percent Change'] = 100*abs(df['max(F(S))']-df['F(D)'])/(abs(df['F(D)'])+.0001)
        df['Percent Change train'] = 100*abs(df['max(F(S))_train']-df['F(D)_train'])/(abs(df['F(D)_train'])+.0001)
    if 'avg diff' not in df.columns:
        df['avg diff'] = abs(df['avg(F(D))']-df['avg(F(S))'])
    df['Percent Change avg'] = 100*df['avg diff']/(abs(df['avg(F(D))'])+.0001)
    df['Magnitude Change'] = abs(np.log10(abs(df['avg(F(S))']/df['avg(F(D))'])))
    if isinstance(df.loc[0]['Subgroup Coefficients'],str):
        df['Subgroup Coefficients'] = df['Subgroup Coefficients'].apply(lambda x: json.loads(x.replace("\'","\"")))
        df['Subgroup Coefficients'] = df['Subgroup Coefficients'].apply(lambda x: {k: v for k, v in sorted(x.items(), key=lambda item: abs(item[1]), reverse=True)})
    zero_groups(df)
    print("**************")
    return df[df['Subgroup Size'] != 0].sort_values('avg diff', key=abs, ascending=False)

def clean_nonsep(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'],axis=1)
    if isinstance(df.loc[0]['Subgroup Coefficients'],str):
        df['Subgroup Coefficients'] = df['Subgroup Coefficients'].apply(lambda x: json.loads(x.replace("\'","\"")))
        df['Subgroup Coefficients'] = df['Subgroup Coefficients'].apply(lambda x: {k: v for k, v in sorted(x.items(), key=lambda item: abs(item[1]), reverse=True)})
    df['Magnitude Change'] = abs(np.log10(abs(df['max(F(S))']/df['F(D)'])))
    zero_groups(df)
    print("**************")
    return df[df['Subgroup Size'] != 0].sort_values('Difference', key=abs, ascending=False)

# 02_16 uses new termination condition
# separable/final is old termination condition

# out_df = clean_sep(pd.read_csv('../output/02_16/student_lime_output.csv'))
# df = pd.read_csv('../data/student/student_cleaned.csv')
# target = 'G3'
# t_split = .5
# sensitive_features = ['sex_M', 'Pstatus_T', 'address_U', 'Dalc', 'Walc', 'health']

#out_df = clean_sep(pd.read_csv('../output/02_16/compas_recid_lime_output.csv'))
#out_df = clean_sep(pd.read_csv('../output/02_16/compas_recid_shap_output.csv'))
#out_df = clean_sep(pd.read_csv('../output/separable/final/compas_recid_gradLR_output.csv'))
# out_df = clean_nonsep(pd.read_csv('../output/nonseparable/final/compas_recid_output.csv'))
# df = pd.read_csv('../data/compas/compas_recid.csv')
# target = 'two_year_recid'
# t_split = .2
# sensitive_features = ['age','sex_Male','race_African-American','race_Asian','race_Caucasian','race_Hispanic','race_Native American','race_Other']

# out_df = clean_sep(pd.read_csv('../output/02_10/bank_lime_output.csv'))
# df = BankDataset().convert_to_dataframe()[0]
# target = 'y'
# t_split = .2
# sensitive_features = ['age', 'marital=married', 'marital=single', 'marital=divorced']

#out_df = clean_sep(pd.read_csv('../output/02_10/folktables_lime_output.csv'))
out_df = clean_sep(pd.read_csv('../output/02_16/folktables_shap_output.csv'))
# out_df = clean_nonsep(pd.read_csv('../output/nonseparable/final/folktables_output.csv'))
df = pd.read_csv('../data/folktables/ACSIncome_MI_2018_new.csv')
target = 'PINCP'
t_split = .2
sensitive_features = ['AGEP', 'SEX', 'MAR_1.0', 'MAR_2.0', 'MAR_3.0', 'MAR_4.0', 'MAR_5.0', 'RAC1P_1.0', 'RAC1P_2.0',
                      'RAC1P_3.0', 'RAC1P_4.0', 'RAC1P_5.0', 'RAC1P_6.0', 'RAC1P_7.0', 'RAC1P_8.0', 'RAC1P_9.0']

new_cols = [col for col in df.columns if col != target] + [target]
df = df[new_cols]
f_sensitive = list(df.columns.get_indexer(sensitive_features))

k = 0
seed = 0
train_df, test_df = train_test_split(df, test_size=t_split, random_state=seed)
x_train, y_train = split_out_dataset(train_df, target)
classifier = RandomForestClassifier(random_state=seed)
classifier.fit(x_train, y_train)

print(out_df.iloc[k]['Feature'])
#g = ast.literal_eval(out_df.iloc[k]['Subgroup Coefficients'])
g = out_df.iloc[k]['Subgroup Coefficients']
intercept = g['Intercept']
del g['Intercept']
count = 0
for i in range(test_df.shape[0]):
    count += in_g(test_df.iloc[i], g, intercept)
print("Size in out", out_df.iloc[k]['Subgroup Size'])
print("Size found", count/test_df.shape[0])

subgroup = [in_g(test_df.iloc[i], g, intercept) for i in range(test_df.shape[0])]
x_test, y_test = split_out_dataset(test_df, target)
g_df = test_df[subgroup]
gx_test, gy_test = split_out_dataset(g_df, target)

tn, fp, fn, tp = confusion_matrix(y_test, classifier.predict(x_test)).ravel()
length_full = len(y_test)

gtn, gfp, gfn, gtp = confusion_matrix(gy_test, classifier.predict(gx_test)).ravel()
length_g = len(gy_test)

#print(f"Overall")
#print(f"Y=1 rate: {sum(classifier.predict(x_test))/length_full}  |  TPR: {tp/length_full}  |  FPR: {fp/length_full}")
#print("Subgroup")
#print(f"Y=1 rate: {sum(classifier.predict(gx_test))/length_g}  |  TPR: {gtp/length_g}  |  FPR: {gfp/length_g}")

#ranges = [[.0, .1], [.1, .2], [.2, .3], [.3, .4], [.4, .5], [.5, .6], [.6, .7], [.7, .8], [.8, .9], [.9, 1.]]
ranges = [[.0, .2],[.2, .4], [.4, .6],[.6, .8], [.8, 1.]]
scores = [c[1] for c in classifier.predict_proba(x_test)]

Es = []
ece = 0
for r in ranges:
    cals, ece_r = get_E(scores, classifier.predict(x_test), y_test, r)
    Es.append(cals)
    ece += ece_r

gscores = [c[1] for c in classifier.predict_proba(gx_test)]

gEs = []
gece = 0
for r in ranges:
    cals, ece_r = get_E(gscores, classifier.predict(gx_test), gy_test, r)
    gEs.append(cals)
    gece += ece_r

print("DELTAS")
print("Y=1 diff:", sum(classifier.predict(gx_test))/length_g - sum(classifier.predict(x_test))/length_full)
print("TPR diff:", gtp/length_g - tp/length_full)
print("FPR diff:", gfp/length_g - fp/length_full)
print("ECE diff:", gece-ece)

# x_ticks = [(x[0]+x[1])/2 for x in ranges]
# fig = plt.figure(figsize=(10, 4))
# f = 18
#
# plt.subplot(1, 2, 1)
# plt.title("Overall", fontsize=f)
# plt.ylabel("Predicted Risk", fontsize=f)
# plt.xlabel("True Risk", fontsize=f)
# plt.plot([0, 1],[0, 1], linestyle='dashed', alpha=.3)
# plt.plot(x_ticks, Es)
#
# plt.subplot(1, 2, 2)
# plt.title("Subgroup", fontsize=f)
# plt.xlabel("True Risk", fontsize=f)
# plt.plot([0, 1],[0, 1], linestyle='dashed', alpha=.3)
# plt.plot(x_ticks, gEs)
#
# plt.tight_layout()
# plt.show()
