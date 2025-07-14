from model import Transformer
from load_data import BPTrainDataset, BPValidDataset, collate_fn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm 

train_loss_hist = []

def train(device="cuda", epochs=30):
    
    global train_loss_hist
    
    data, valid = BPTrainDataset(), BPValidDataset()
    dataloader = DataLoader(data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    validloader = DataLoader(valid, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    model = Transformer(input_dim=6, embed_dim=128, num_heads=8, mlp_dim=128, 
                 num_layers=3, dropout=0.1).to(device)
    l1loss = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    min_valid_loss, last_valid_loss = torch.inf, 0
    
    for i in range(epochs):
        
        train_loss, valid_loss = 0, 0
        model.train()        
        pbar = tqdm(enumerate(dataloader))
        for j, batch in pbar:
            x, pad_mask, seq_len = batch
            x = x.to(device)
            n, s, d = x.shape
            
            pred = model(x, True, pad_mask.to(device)) # N x seq_len x dim
            pred_mask = pad_mask.clone()
            pred_mask[torch.arange(n), seq_len-1] = True

            target = torch.cat([x[:,1:,:].clone(), torch.zeros(n, 1, d).to(device)], dim=1)
            ## more weight on BP
            loss = 0.5 * l1loss(pred[~pred_mask], target[~pred_mask]) + \
                   1.5 * l1loss(pred[...,:2][~pred_mask], target[...,:2][~pred_mask])
            
            train_loss_hist.append(loss.item())
            
            train_loss += (loss.item() - train_loss)/(j+1)
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
                        
            pbar.set_postfix({"TRAIN_LOSS": train_loss, "VALID_LOSS": last_valid_loss})
    

        model.eval()
        
        for j, batch in enumerate(validloader):
            
            with torch.no_grad():
                x, pad_mask, seq_len = batch
                x = x.to(device)
                ## pad_mask n x seq_len
                n, s, d = x.shape
                
                pred = model(x, padding_mask=pad_mask.to(device)) # N x seq_len x dim
                
                pred_mask = pad_mask.clone()
                pred_mask[torch.arange(n), seq_len-1] = True
                target = torch.cat([x[:,1:,:].clone(), torch.zeros(n, 1, d).to(device)], dim=1)
                
                loss = 0.5 *l1loss(pred[~pred_mask], target[~pred_mask]) + \
                       1.5 *l1loss(pred[...,:2][~pred_mask], target[...,:2][~pred_mask]) ## more weight on BP
                
                valid_loss += (loss.item() - valid_loss)/(j+1)
        
        last_valid_loss = valid_loss

        if valid_loss < min_valid_loss:
            torch.save(model.state_dict(), "checkpoint.ckpt")
            min_valid_loss = valid_loss
        
    
    
        
if __name__ == "__main__":
    train()
        
        
    