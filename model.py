
import torch
from torch.nn import functional as F

class MLP(torch.nn.Module):
    
    def __init__(self, features, hidden_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(features, hidden_dim)
        self.linear_2 = torch.nn.Linear(hidden_dim, features)
        self.norm = torch.nn.LayerNorm(features)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        res = x
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x) + res
        return self.norm(x)

class PositionalEncoding(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ## x: N x seq_len x embed_dim
        device = x.device
        N, S, D = x.shape
        pos = torch.arange(0,S).to(device) # seq_len
        pos_encode = torch.arange(0, D, 2).view(-1,1).repeat(1,2).view(-1).to(device) ## embed_dim
        pos_encode = torch.pow(10000, pos_encode / D)
        pos_encode = pos[:, None] / pos_encode[None, :] # seq_len x embed_dim
        pos_encode[:, ::2] = torch.sin(pos_encode[:, ::2])
        pos_encode[:, 1::2] = torch.cos(pos_encode[:, 1::2])
        
        return pos_encode[None,:,:] + x
        
        
class TransformerLayer(torch.nn.Module):
    
    def __init__(self, embed_dim=128, num_heads=1, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, 
                                                num_heads=num_heads, 
                                                dropout=dropout, 
                                                batch_first=True)
        self.mlp = MLP(embed_dim, mlp_dim)
    
    def forward(self, x:torch.Tensor, attn_mask=None, key_padding_mask=None) -> torch.Tensor:
        # x in dim N x seq_len x embed_dim
        res = x
 
        x, _ = self.attn(x,x,x, 
                      attn_mask=attn_mask, 
                      need_weights=False, 
                      key_padding_mask=key_padding_mask)
        
        x = x + res
        x = self.norm(x)
        return self.mlp(x)
    

class Transformer(torch.nn.Module):
    
    def __init__(self, input_dim=5, embed_dim=128, num_heads=4, mlp_dim=128, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding()
        self.trans_layers = torch.nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
            )
        self.out_layer = torch.nn.Linear(embed_dim, input_dim)
    
    def forward(self, x, attn_mask=True, padding_mask=None):
        device = x.device
        N, S, D = x.shape
        
        embed = self.proj(x)
        embed = self.pos_encoding(embed)
        
        if attn_mask:
            attn_mask = self.generate_attn_mask(S).to(device)
        else:
            attn_mask = None
        for layer in self.trans_layers:
            embed = layer(embed, attn_mask, padding_mask)
            
        out = self.out_layer(embed)
        
        return out 
    
    def generate_attn_mask(self, size):
        return torch.triu(torch.full((size, size), -torch.inf), diagonal=1)
    
    
    
if __name__ == "__main__":
    
    x = torch.randn(4, 200, 5)
    model = Transformer(5,)
    model(x)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        