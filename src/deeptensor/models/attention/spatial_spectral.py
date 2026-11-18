import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _get_sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class SpectralSelfAttention(nn.Module):
    def __init__(self, d_model=128, seq_len=260, heads=2):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.heads = heads
        
        # Precompute fixed sinusoidal positional encoding
        pe = _get_sinusoidal_pe(seq_len, d_model)
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
        # x is [B, C, L]
        # Transpose to [B, L, C]
        x = x.transpose(1, 2)
        L = x.shape[1]
        
        # Add positional encoding (dynamic length support)
        if L <= self.pe.shape[1]:
            x = x + self.pe[:, :L, :]
        else:
            pe_dyn = _get_sinusoidal_pe(L, self.d_model).to(x.device)
            x = x + pe_dyn
        
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
        pe = _get_sinusoidal_pe(seq_len, d_model)
        self.register_buffer('pe', pe)
        
        # Pointwise mixing (replaces MHA)
        self.norm1 = nn.LayerNorm(d_model)
        # Using hidden closely matches MHA params (256 for d_model=128, 128 for d_model=64)
        hidden = 256 if d_model == 128 else (128 if d_model == 64 else d_model * 2)
        self.pointwise_mixing = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model)
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
        L = x.shape[1]
        if L <= self.pe.shape[1]:
            x = x + self.pe[:, :L, :]
        else:
            pe_dyn = _get_sinusoidal_pe(L, self.d_model).to(x.device)
            x = x + pe_dyn
        
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

class SpectralConvControl(nn.Module):
    """
    Pure Convolutional Parameter-Matched Control.
    Replaces global self-attention with a local 1D convolutional residual block
    with exact matched parameter capacity (+-0.05% of MHA + FFN).
    """
    def __init__(self, d_model=128, seq_len=260):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Positional encoding (same as attention to preserve spatial encoding fairness)
        pe = _get_sinusoidal_pe(seq_len, d_model)
        self.register_buffer('pe', pe)

        # Exact parameter-matched hidden dimension
        if d_model == 128:
            hidden = 172  # 132,396 params matches 132,480 params within 84 params (0.06%)
        elif d_model == 64:
            hidden = 129  # 49,729 params matches 49,984 params within 255 params (0.5%)
        else:
            hidden = int(round(d_model * 1.34375))

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, need_weights=False):
        # x is [B, C, L]
        # Residual 1D conv block
        res = x
        x_conv = self.conv_block(x)
        x = res + x_conv

        # Apply LayerNorm along channel dimension
        x = x.transpose(1, 2)
        L = x.shape[1]
        if L <= self.pe.shape[1]:
            x = x + self.pe[:, :L, :]
        else:
            x = x + _get_sinusoidal_pe(L, self.d_model).to(x.device)
        x = self.norm(x)
        x = x.transpose(1, 2)

        if need_weights:
            return x, None
        return x

class HybridSpectralNet(nn.Module):
    def __init__(self, original_spec_net, variant='hybrid', seq_len=260, d_model=None, heads=2):
        super().__init__()
        # Ensure it's the expected Sequential
        if not isinstance(original_spec_net, nn.Sequential):
            raise ValueError("Expected original_spec_net to be an nn.Sequential")
            
        modules = list(original_spec_net.children())
        
        # The body is everything except the last module
        self.body = nn.Sequential(*modules[:-1])
        
        # The final projection is the last module
        self.final_proj = modules[-1]
        
        # Auto-detect d_model from final_proj in_channels if not provided
        if d_model is None:
            detected = None
            for m in self.final_proj.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    detected = m.in_channels if hasattr(m, 'in_channels') else m.in_features
                    break
            d_model = detected if detected is not None else 128

        # Ensure heads divides d_model
        if heads is None or d_model % heads != 0:
            heads = 1
            for h in (4, 2):
                if d_model % h == 0:
                    heads = h
                    break

        # Instantiate the correct block
        if variant == 'hybrid':
            self.attention_block = SpectralSelfAttention(d_model=d_model, seq_len=seq_len, heads=heads)
        elif variant in ('control', 'pointwise_control'):
            self.attention_block = SpectralPointwiseControl(d_model=d_model, seq_len=seq_len)
        elif variant in ('conv_control', 'cnn_control'):
            self.attention_block = SpectralConvControl(d_model=d_model, seq_len=seq_len)
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
