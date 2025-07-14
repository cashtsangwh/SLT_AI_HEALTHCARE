# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 22:50:35 2025

@author: User
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

import random

## Handling unreasonable value

df = pd.read_csv("./data/processed_data.csv")
df.loc[(df["ASBP"] < 30) | (df["ASBP"] > 400), ["ASBP"]] = np.nan
df.loc[(df["ADBP"] > 400) | (df["ADBP"] < 20), ["ADBP"]] = np.nan
df.loc[(df["HR"] < 0) | (df["HR"] > 300), ["HR"]] = np.nan
df.loc[(df["SPO2"] < 0 ) | (df["SPO2"] > 100), ["SPO2"]] = np.nan
df.loc[(df["RESP"] < 0 ) | (df["RESP"] > 100), ["RESP"]] = np.nan

print(df.max(axis=0))

## Fill missing value for each admission case
hadm_ids = df["HADM_ID"].unique().tolist()

data = []
for hadm in hadm_ids:
    temp = df.loc[df["HADM_ID"] == hadm]
    temp = temp.interpolate(method='linear')
    temp = temp.bfill()
    temp = temp.ffill()
    if temp.isna().values.any():
        continue
    data.append(temp)

df = pd.concat(data)


## Checking whether HOUR is a factor

groups = [(0,4), (4,8), (4,8), (8, 12), (12, 16), (16, 20), (20, 24)]

sbp_time_df = pd.DataFrame()
for l, u in groups:
    sbp_time_df[f"{l}-{u}"] = df.loc[(l <= df["HOUR"]) & (df["HOUR"] < u)]["ASBP"].iloc[:300000].values


dbp_time_df = pd.DataFrame()
for l, u in groups:
    dbp_time_df[f"{l}-{u}"] = df.loc[(l <= df["HOUR"]) & (df["HOUR"] < u)]["ADBP"].iloc[:300000].values


plt.figure(figsize=(12,6))
sns.violinplot(x="variable", y="value", data=pd.melt(dbp_time_df))
plt.title("Blood Pressure Diastolic in Different Time Period")
plt.xlabel("Time Period")
plt.ylabel("Blood Pressure Diastolic")


plt.show()
plt.figure(figsize=(12,6))
sns.violinplot(x="variable", y="value", data=pd.melt(sbp_time_df))
plt.title("Blood Pressure Systolic in Different Time Period")
plt.xlabel("Time Period")
plt.ylabel("Blood Pressure Systolic")
plt.show()



f_statistic, p_value = stats.f_oneway(*[sbp_time_df[f"{l}-{u}"] for l, u in groups])

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")


f_statistic, p_value = stats.f_oneway(*[dbp_time_df[f"{l}-{u}"] for l, u in groups])

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")


asbp_data = df.loc[:, ["ASBP", "HOUR", "HR", "SPO2", "RESP"] ]
asbp_corr = asbp_data.corr()
sns.heatmap(asbp_corr)
plt.show()

adbp_data = df.loc[:, ["ADBP", "HOUR", "HR", "SPO2", "RESP"] ]
adbp_corr = adbp_data.corr()
sns.heatmap(adbp_corr)
plt.show()


## Spliting dataset

random.shuffle(hadm_ids)
split_idx = int(len(hadm_ids) * 0.8) # 80% for training and 20% for validation
train = hadm_ids[:split_idx]
valid = hadm_ids[split_idx:]

train_dataset = df.loc[df["HADM_ID"].isin(train)]
valid_dataset = df.loc[df["HADM_ID"].isin(valid)]

train_dataset.to_csv("./data/TRAIN.csv", index=False)
valid_dataset.to_csv("./data/VALID.csv", index=False)



