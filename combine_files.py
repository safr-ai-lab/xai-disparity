import pandas as pd
import json
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser(description='Locally separable run')
parser.add_argument('name', action='store', type=str)
args = parser.parse_args()
name = args.name

def clean_df(df, split_adjust=1):
    df = df.drop(['Unnamed: 0'],axis=1)
    df['F(D)'] = df['F(D)'].apply(lambda x: x*split_adjust)
    df['max(F(S))'] = df['max(F(S))'].apply(lambda x: x*split_adjust)
    df['Subgroup Coefficients'] = df['Subgroup Coefficients'].apply(lambda x: json.loads(x.replace("\'","\"")))
    df['Subgroup Coefficients'] = df['Subgroup Coefficients'].apply(lambda x: {k: v for k, v in sorted(x.items(), key=lambda item: abs(item[1]), reverse=True)})
    return df

final_df = pd.DataFrame()

for file in glob.glob(f"{name}*"):
    df = clean_df(pd.read_csv(file))
    final_df = pd.concat([final_df, df])

final_df.sort_values('Percent Change',ascending=False)

final_df.to_csv(f'{name}_combined.csv', index=False)

