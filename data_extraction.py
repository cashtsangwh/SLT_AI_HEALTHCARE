import pandas as pd
import os
import numpy as np
from collections import defaultdict


DATAPATH = "D:\\Dataset\\mimic-iii-clinical-database-1.4"

item = pd.read_csv(os.path.join(DATAPATH, "D_ITEMS.csv.gz"))
item = item[~item["LABEL"].isna()]
item_numeric = item.loc[item["PARAM_TYPE"] == "Numeric"]

sbp_id = item.loc[item["LABEL"].str.lower().str.contains("blood pressure systolic")]["ITEMID"].values.tolist()
dbp_id = item.loc[item["LABEL"].str.lower().str.contains("blood pressure diastolic")]["ITEMID"].values.tolist()
hr_id = item.loc[item["LABEL"] == "Heart Rate"]["ITEMID"].values.tolist()
o2sat_id = item.loc[(item["LABEL"] == "SpO2") | 
                    (item["LABEL"] == "O2 saturation pulseoxymetry") | 
                    (item["LABEL"] == "Arterial O2 Saturation") ]["ITEMID"].values.tolist()
resp_id = item.loc[item["LABEL"] == "Respiratory Rate"]["ITEMID"].values.tolist()

itemid_dict = {"ASBP": sbp_id,
           "ADBP": dbp_id,
           "HR": hr_id,
           "SPO2": o2sat_id,
           "RESP": resp_id,
           }

tables_dict = defaultdict(list)

events = pd.read_csv(os.path.join(DATAPATH, "CHARTEVENTS.csv.gz"), chunksize=1000000)


for event in events:
    
    for label, itemid in itemid_dict.items():
        df = event.loc[event["ITEMID"].isin(itemid)]
        if len(df) > 0:
            tables_dict[label].append(df)



for key in tables_dict.keys():
    tables_dict[key] = pd.concat(tables_dict[key])
    df = tables_dict[key]
    df[key] = df.pop("VALUENUM")
    chart_time = df.pop("CHARTTIME")
    df["DATE"] = pd.to_datetime(chart_time).dt.date
    df["HOUR"] = pd.to_datetime(chart_time).dt.hour
    df = df.loc[:,["SUBJECT_ID", "HADM_ID", "DATE", "HOUR", key]]
    df = df.groupby(["SUBJECT_ID", "HADM_ID", "DATE", "HOUR"]).mean().reset_index()
    tables_dict[key] = df


merge_df = pd.merge(tables_dict["ASBP"], tables_dict["ADBP"], on=["SUBJECT_ID", "HADM_ID", "DATE", "HOUR"], how="outer")

for key, table in tables_dict.items():
    if key != "ASBP" and key != "ADBP":
        merge_df = pd.merge(merge_df, table, on=["SUBJECT_ID", "HADM_ID", "DATE", "HOUR"], how="left")

merge_df.to_csv("./data/processed_data.csv")
