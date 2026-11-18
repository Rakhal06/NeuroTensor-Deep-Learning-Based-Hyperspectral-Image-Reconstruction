import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralSelfAttention(nn.Module):
    def __init__(self, d_model=128, seq_len=260, heads=2):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.heads = heads
        
        # Precompute fixed sinusoidal positional encoding
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dim: [1, seq_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Attention components
        self.norm1 = nn.LayerNorm(d_model)
        # batch_first=True makes input/output [batch, seq, d_model]
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        
        # FFN components
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, d_model)
        )

    def forward(self, x, need_weights=False):
        # x is [B, C, L] = [1, 128, 260]
        # Transpose to [B, L, C]
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pe
        
        # Pre-norm + Attention + Residual
        x_norm = self.norm1(x)
        # mha returns (attn_output, attn_weights)
        attn_out, attn_weights = self.mha(x_norm, x_norm, x_norm, need_weights=need_weights)
        x = x + attn_out
        
        # Pre-norm + FFN + Residual
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        
        # Transpose back to [B, C, L]
        x = x.transpose(1, 2)
        
        if need_weights:
            return x, attn_weights
        return x

class SpectralPointwiseControl(nn.Module):
    def __init__(self, d_model=128, seq_len=260):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Positional encoding (same as attention to be fair)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Pointwise mixing (replaces MHA)
        self.norm1 = nn.LayerNorm(d_model)
        # Using hidden=256 closely matches MHA(128, 2 heads) params
        self.pointwise_mixing = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, d_model)
        )
        
        # FFN components (identical to attention)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, d_model)
        )

    def forward(self, x, need_weights=False):
        # x is [B, C, L]
        x = x.transpose(1, 2)
        x = x + self.pe
        
        x_norm = self.norm1(x)
        # Pointwise mixing instead of cross-token attention
        mix_out = self.pointwise_mixing(x_norm)
        x = x + mix_out
        
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        
        x = x.transpose(1, 2)
        
        if need_weights:
            # Control has no attention weights
            return x, None
        return x

class HybridSpectralNet(nn.Module):
    def __init__(self, original_spec_net, variant='hybrid'):
        super().__init__()
        # Ensure it's the expected Sequential
        if not isinstance(original_spec_net, nn.Sequential):
            raise ValueError("Expected original_spec_net to be an nn.Sequential")
            
        modules = list(original_spec_net.children())
        
        # The body is everything except the last module
        self.body = nn.Sequential(*modules[:-1])
        
        # The final projection is the last module
        self.final_proj = modules[-1]
        
        # Instantiate the correct block
        if variant == 'hybrid':
            self.attention_block = SpectralSelfAttention(d_model=128, seq_len=260, heads=2)
        elif variant == 'control':
            self.attention_block = SpectralPointwiseControl(d_model=128, seq_len=260)
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward(self, x, need_weights=False):
        x = self.body(x)
        
        if need_weights and hasattr(self.attention_block, 'forward'):
            x, weights = self.attention_block(x, need_weights=True)
        else:
            x = self.attention_block(x)
            weights = None
            
        x = self.final_proj(x)
        
        if need_weights:
            return x, weights
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, input_shape=(1, 20, 260), variant='hybrid'):
    """
    Very rough analytical FLOPs estimation for the attention/control blocks.
    (Body FLOPs are identical, so we just calculate the difference block)
    """
    B, C, L = input_shape
    # The block takes input of shape [1, 128, 260]
    D = 128
    
    flops = 0
    # Positional encoding addition: B * L * D
    flops += B * L * D
    
    # LayerNorm 1: mean, var, scale, shift ~ 4 operations per element
    flops += 4 * B * L * D
    
    if variant == 'hybrid':
        # MHA FLOPs (simplified)
        # Q, K, V projections: 3 * B * L * D * D
        flops += 3 * B * L * D * D
        # QK^T: B * Heads * L * L * (D/Heads) = B * L^2 * D
        flops += B * L * L * D
        # Softmax: ~ 3 operations per element
        flops += 3 * B * 2 * L * L
        # Attention * V: B * Heads * L * L * (D/Heads) = B * L^2 * D
        flops += B * L * L * D
        # Output projection: B * L * D * D
        flops += B * L * D * D
    elif variant == 'control':
        # Pointwise mixing (D -> 256 -> D)
        # Linear 1: B * L * D * 256
        flops += B * L * D * 256
        # GELU: ~ 4 ops per element
        flops += 4 * B * L * 256
        # Linear 2: B * L * 256 * D
        flops += B * L * 256 * D
        
    # Residual 1
    flops += B * L * D
    
    # LayerNorm 2
    flops += 4 * B * L * D
    
    # FFN (D -> 256 -> D)
    flops += B * L * D * 256
    flops += 4 * B * L * 256
    flops += B * L * 256 * D
    
    # Residual 2
    flops += B * L * D
    
    return flops
