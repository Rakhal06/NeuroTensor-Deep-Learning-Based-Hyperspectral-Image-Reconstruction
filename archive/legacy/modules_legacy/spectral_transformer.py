import torch
import torch.nn as nn
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_alibi_bias(seq_len, num_heads):
    """
    Returns fixed ALiBi distance bias.
    Shape: [1, num_heads, seq_len, seq_len]
    """
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
            
    slopes = torch.tensor(get_slopes(num_heads))
    
    context_position = torch.arange(seq_len)[:, None]
    memory_position = torch.arange(seq_len)[None, :]
    relative_position = memory_position - context_position 
    # bidirectional ALiBi
    relative_position = -torch.abs(relative_position).float()
    
    alibi = slopes[:, None, None] * relative_position[None, None, :, :]
    return alibi

def get_sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class SpectralTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, d_ffn=512, seq_len=260, rank=20):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_heads = n_heads
        
        self.embedding = nn.Linear(rank, d_model)
        self.register_buffer('pe', get_sinusoidal_pe(seq_len, d_model))
        
        alibi = get_alibi_bias(seq_len, n_heads).squeeze(0) # [num_heads, seq_len, seq_len]
        self.register_buffer('alibi', alibi)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=0.0, batch_first=True),
                'norm2': nn.LayerNorm(d_model),
                'ffn1': nn.Linear(d_model, d_ffn),
                'act': nn.GELU(),
                'ffn2': nn.Linear(d_ffn, d_model)
            })
            self.layers.append(layer)
            
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, rank)
        
    def forward(self, x, need_weights=False):
        B, C, L = x.shape
        x = x.permute(0, 2, 1) # [B, L, C]
        
        x = self.embedding(x)
        x = x + self.pe
        
        # Expand ALiBi
        attn_mask = self.alibi.repeat(B, 1, 1) # [B*heads, L, L]
        
        attn_weights_list = []
        
        for layer in self.layers:
            norm1 = layer['norm1'](x)
            attn_out, attn_weights = layer['attn'](norm1, norm1, norm1, attn_mask=attn_mask, need_weights=need_weights)
            x = x + attn_out
            
            norm2 = layer['norm2'](x)
            ffn_out = layer['ffn2'](layer['act'](layer['ffn1'](norm2)))
            x = x + ffn_out
            
            if need_weights:
                attn_weights_list.append(attn_weights)
                
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        x = x.permute(0, 2, 1) # [B, C, L]
        
        if need_weights:
            return x, attn_weights_list[-1]
        return x

class ControlSpectralTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, d_ffn=512, seq_len=260, rank=20):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Linear(rank, d_model)
        self.register_buffer('pe', get_sinusoidal_pe(seq_len, d_model))
        
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                # Two linear layers (256 width) match MHSA's 66,048 parameters very closely (65,920).
                # No cross-token mixing, purely point-wise linear operation.
                'control_mix1': nn.Linear(d_model, 256),
                'control_mix2': nn.Linear(256, d_model),
                'norm2': nn.LayerNorm(d_model),
                'ffn1': nn.Linear(d_model, d_ffn),
                'act': nn.GELU(),
                'ffn2': nn.Linear(d_ffn, d_model)
            })
            self.layers.append(layer)
            
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, rank)
        
    def forward(self, x, need_weights=False):
        B, C, L = x.shape
        x = x.permute(0, 2, 1)
        
        x = self.embedding(x)
        x = x + self.pe
        
        for layer in self.layers:
            norm1 = layer['norm1'](x)
            mix_out = layer['control_mix2'](layer['control_mix1'](norm1))
            x = x + mix_out
            
            norm2 = layer['norm2'](x)
            ffn_out = layer['ffn2'](layer['act'](layer['ffn1'](norm2)))
            x = x + ffn_out
            
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        x = x.permute(0, 2, 1)
        
        if need_weights:
            return x, None
        return x
