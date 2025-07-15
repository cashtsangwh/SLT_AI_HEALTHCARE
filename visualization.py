import pandas as pd
from matplotlib import pyplot as plt
import random

df = pd.read_csv("./data/processed_data.csv")
hadm_ids = df["HADM_ID"].unique().tolist()
plot_num = 3
random.shuffle(hadm_ids)

for hadm in hadm_ids:
    case = df.loc[df["HADM_ID"] == hadm]
    if not case.isna().values.any() and 400 > len(case) > 200: #
        plt.figure(figsize=(16,8))
        plt.plot(case["ASBP"].tolist(), label="Systolic BP")
        plt.plot(case["ADBP"].tolist(), label="Diastolic BP")
        plt.plot(case["HR"].tolist(), label="Heart Rate")
        plt.plot(case["SPO2"].tolist(), label="O2 Sat")
        plt.plot(case["RESP"].tolist(), label="Resp Rate")
        
        plt.legend()
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title(f"Measurement of Admission Cases {hadm}")
        plt.show()
        
        plot_num -= 1
        if plot_num == 0:
            break
        