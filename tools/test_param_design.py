import torch
import torch.nn as nn

def design_tri_branch(d_model=128):
    # 1. Router: Channel-Aware Global Context Gate (CAGC)
    d_route = 16
    router = nn.Sequential(
        nn.Linear(d_model, d_route),
        nn.GELU(),
        nn.Linear(d_route, 3 * d_model)
    )
    router_params = sum(p.numel() for p in router.parameters())
    
    # 2. Branch 1: CNN (Depthwise-Separable 1D conv)
    # Target ~41k params
    # Conv1d with channels C_mid
    # Let's check depthwise separable:
    # dw1 (k=3): d_model * 3 + d_model
    # pw1: d_model * c_mid + c_mid
    # dw2 (k=5): c_mid * 5 + c_mid
    # pw2: c_mid * d_model + d_model
    # To get ~40k params:
    # pw1 + pw2 = 2 * d_model * c_mid. If d_model=128, 2 * 128 * c_mid = 256 * c_mid.
    # If c_mid = 158: 256 * 158 = 40,448 params!
    c_mid_cnn = 158
    b1_norm = nn.GroupNorm(8, d_model)
    b1 = nn.Sequential(
        nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
        nn.Conv1d(d_model, c_mid_cnn, kernel_size=1),
        nn.GELU(),
        nn.Conv1d(c_mid_cnn, c_mid_cnn, kernel_size=5, padding=2, groups=c_mid_cnn),
        nn.Conv1d(c_mid_cnn, d_model, kernel_size=1)
    )
    b1_params = sum(p.numel() for p in b1.parameters()) + sum(p.numel() for p in b1_norm.parameters())
    
    # 3. Branch 2: Pointwise Mixing (1x1 linear bottleneck)
    # Target ~41k params
    # LayerNorm: 2 * d_model = 256
    # Linear 1: d_model * c_mid_pw + c_mid_pw
    # Linear 2: c_mid_pw * d_model + d_model
    # Total = 256 * c_mid_pw + c_mid_pw + d_model + 256
    # If c_mid_pw = 158: 257 * 158 + 128 + 256 = 40,990 params!
    c_mid_pw = 158
    b2_norm = nn.LayerNorm(d_model)
    b2 = nn.Sequential(
        nn.Linear(d_model, c_mid_pw),
        nn.GELU(),
        nn.Linear(c_mid_pw, d_model)
    )
    b2_params = sum(p.numel() for p in b2.parameters()) + sum(p.numel() for p in b2_norm.parameters())
    
    # 4. Branch 3: Attention (Pre-LN MHSA + FFN)
    # Target ~41k params
    # LayerNorm1: 256
    # MHA: heads=2, d_k=32 -> embed_dim = 64
    # in_proj from 128 to 64:
    # in_proj: 128 -> 64 (proj_in), MHA(64, heads=2), proj_out: 64 -> 128
    # proj_in: 128 * 64 + 64 = 8,256
    # MHA(64): 3 * 64 * 64 + 3*64 + 64*64 + 64 = 12,288 + 192 + 4,096 + 64 = 16,640
    # proj_out: 64 * 128 + 128 = 8,320
    # LayerNorm2: 256
    # FFN(128 -> 64 -> 128): 128*64 + 64 + 64*128 + 128 = 16,576
    # Total B3 = 8,256 + 16,640 + 8,320 + 256 + 16,576 = 50k
    # Let's tune:
    
    print(f"Router ({d_route}): {router_params:,} params")
    print(f"Branch 1 CNN (c_mid={c_mid_cnn}): {b1_params:,} params")
    print(f"Branch 2 PW (c_mid={c_mid_pw}): {b2_params:,} params")
    
    # Alternatively, what if MHA operates directly at d_model=128?
    # If MHA(heads=2) at d_model=128 is 66,048, then MHA alone is 66k.
    # What if MHA is projected, or what if the Tri-Branch module replaces the existing Hybrid block?
    
design_tri_branch(128)

