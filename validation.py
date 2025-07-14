import torch
from model import Transformer
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(input_dim=6, embed_dim=128, num_heads=8, mlp_dim=128, 
                    num_layers=3, dropout=0.1).to(device)
model.load_state_dict(torch.load("./final_checkpoint.ckpt"))
model.eval()
valid_df = pd.read_csv("./data/VALID.csv")
hadm_ids = valid_df["HADM_ID"].unique().tolist()

case = 0
sbp_valid_loss = 0
dbp_valid_loss = 0
N = 0

for hadm_id in hadm_ids:
    data = valid_df.loc[valid_df["HADM_ID"] == hadm_id]
    if len(data) > 1:
        
        x = data.loc[:,["ASBP", "ADBP", "HOUR", "HR", "SPO2", "RESP"]].iloc[:-1,:].values / 100
        y = data.loc[:,["ASBP", "ADBP", "HOUR", "HR", "SPO2", "RESP"]].iloc[1:,:].values
        
        x = torch.from_numpy(x).float().to(device)[None, ...]
        with torch.no_grad():
            pred = model(x, attn_mask=True)[0]
        
        sbp = pred[:,0].detach().cpu().numpy() * 100
        dbp = pred[:,1].detach().cpu().numpy() *100
        
        mae_sbp = np.abs(sbp - y[:,0]).mean()
        mae_dbp = np.abs((dbp - y[:,1])).mean()
        
        sbp_valid_loss += (mae_sbp - sbp_valid_loss) / (N + 1)
        dbp_valid_loss += (mae_dbp - dbp_valid_loss) / (N + 1)
        
        N += 1
        
        if len(data) > 100 and len(data) < 200 and case < 5:
            case += 1
            plt.figure(figsize=(12,6))
            plt.plot(sbp , label="Model")
            plt.plot(y[:,0], label="True")
            plt.ylim(0, 200)
            plt.title("Blood Pressure Systolic")
            plt.legend()
            plt.show()
            
            print("L1 Loss SBP", np.abs(sbp - y[:,0]).mean())
            
            plt.figure(figsize=(12,6))
            plt.plot(dbp , label="Model")
            plt.plot(y[:,1], label="True")
            plt.ylim(0, 150)
            plt.title("Blood Pressure Diastolic")
            plt.legend()
            plt.show()
            
            print("L1 Loss DBP", np.abs((dbp - y[:,1])).mean())

print("SBP MAE:", sbp_valid_loss)
print("DBP MAE:", dbp_valid_loss)
        
    
        
    
    