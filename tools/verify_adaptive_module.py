import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAwareGlobalContextGate(nn.Module):
    def __init__(self, d_model=128, d_route=16):
        super().__init__()
        self.d_model = d_model
        self.d_route = d_route
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_route),
            nn.GELU(),
            nn.Linear(d_route, 3 * d_model),
        )
        
    def forward(self, x, tau=1.0):
        # x: [B, C, L]
        B, C, L = x.shape
        s = x.mean(dim=-1)  # [B, C]
        u = self.mlp(s)      # [B, 3 * C]
        u = u.view(B, C, 3)  # [B, C, 3]
        gates = F.softmax(u / tau, dim=-1)  # [B, C, 3]
        return gates

class LowRankChannelPositionGate(nn.Module):
    def __init__(self, d_model=128, d_route=16):
        super().__init__()
        self.d_model = d_model
        self.d_route = d_route
        self.chan_gate = nn.Sequential(
            nn.Linear(d_model, d_route),
            nn.GELU(),
            nn.Linear(d_route, 3 * d_model),
        )
        self.pos_gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, 3, kernel_size=1),
        )
        
    def forward(self, x, tau=1.0):
        # x: [B, C, L]
        B, C, L = x.shape
        s = x.mean(dim=-1)  # [B, C]
        u_chan = self.chan_gate(s).view(B, C, 1, 3)  # [B, C, 1, 3]
        u_pos = self.pos_gate(x).permute(0, 2, 1).view(B, 1, L, 3)  # [B, 1, L, 3]
        u_total = u_chan + u_pos  # [B, C, L, 3]
        gates = F.softmax(u_total / tau, dim=-1)  # [B, C, L, 3]
        return gates

class LocalCNNBranch(nn.Module):
    def __init__(self, d_model=128, c_cnn=143):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=4, num_channels=d_model)
        self.dw1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pw1 = nn.Conv1d(d_model, c_cnn, kernel_size=1)
        self.act = nn.GELU()
        self.dw2 = nn.Conv1d(c_cnn, c_cnn, kernel_size=5, padding=2, groups=c_cnn)
        self.pw2 = nn.Conv1d(c_cnn, d_model, kernel_size=1)
        
    def forward(self, x):
        h = self.gn(x)
        h = self.dw1(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.dw2(h)
        h = self.pw2(h)
        return h

class PointwiseBranch(nn.Module):
    def __init__(self, d_model=128, c_pw=135):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, c_pw)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(c_pw, d_model)
        
    def forward(self, x):
        xt = x.transpose(1, 2)
        h = self.linear2(self.act(self.linear1(self.norm(xt))))
        return h.transpose(1, 2)

class GlobalAttentionBranch(nn.Module):
    def __init__(self, d_model=128, d_attn=64, heads=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, d_attn)
        self.mha = nn.MultiheadAttention(d_attn, heads, batch_first=True)
        self.proj_out = nn.Linear(d_attn, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_attn),
            nn.GELU(),
            nn.Linear(d_attn, d_model),
        )
        
    def forward(self, x, pe=None, need_weights=False):
        xt = x.transpose(1, 2)
        if pe is not None:
            xt = xt + pe
        z = self.proj_in(self.norm1(xt))
        attn_out, attn_weights = self.mha(z, z, z, need_weights=need_weights)
        a = self.proj_out(attn_out)
        ffn_out = self.ffn(self.norm2(a))
        out = (a + ffn_out).transpose(1, 2)
        if need_weights:
            return out, attn_weights
        return out

class AdaptiveTriBranchBlock(nn.Module):
    def __init__(self, d_model=128, router_type='cagc', tau=1.0):
        super().__init__()
        self.d_model = d_model
        self.router_type = router_type
        self.tau = tau
        
        if d_model == 128:
            d_route = 16
            c_cnn = 143
            c_pw = 135
            d_attn = 64
        elif d_model == 64:
            d_route = 8
            c_cnn = 113
            c_pw = 113
            d_attn = 40
        else:
            raise ValueError(f"Unsupported d_model: {d_model}")
            
        if router_type == 'cagc':
            self.router = ChannelAwareGlobalContextGate(d_model=d_model, d_route=d_route)
        elif router_type == 'lr_cpg':
            self.router = LowRankChannelPositionGate(d_model=d_model, d_route=d_route)
        elif router_type == 'static':
            self.router = None
        else:
            raise ValueError(f"Unknown router_type: {router_type}")
            
        self.cnn_branch = LocalCNNBranch(d_model=d_model, c_cnn=c_cnn)
        self.pw_branch = PointwiseBranch(d_model=d_model, c_pw=c_pw)
        self.attn_branch = GlobalAttentionBranch(d_model=d_model, d_attn=d_attn, heads=2)
        self.last_gates = None
        
    def forward(self, x, pe=None, tau=None, need_weights=False):
        if tau is None:
            tau = self.tau
            
        B, C, L = x.shape
        f_cnn = self.cnn_branch(x)
        f_pw = self.pw_branch(x)
        if need_weights:
            f_attn, attn_w = self.attn_branch(x, pe=pe, need_weights=True)
        else:
            f_attn = self.attn_branch(x, pe=pe, need_weights=False)
            attn_w = None
            
        if self.router_type == 'static':
            g_cnn = 1.0 / 3.0
            g_pw = 1.0 / 3.0
            g_attn = 1.0 / 3.0
            f_fused = g_cnn * f_cnn + g_pw * f_pw + g_attn * f_attn
            self.last_gates = None
        elif self.router_type == 'cagc':
            gates = self.router(x, tau=tau)  # [B, C, 3]
            self.last_gates = gates.detach()
            g_cnn = gates[..., 0].unsqueeze(-1)   # [B, C, 1]
            g_pw = gates[..., 1].unsqueeze(-1)    # [B, C, 1]
            g_attn = gates[..., 2].unsqueeze(-1)  # [B, C, 1]
            f_fused = g_cnn * f_cnn + g_pw * f_pw + g_attn * f_attn
        elif self.router_type == 'lr_cpg':
            gates = self.router(x, tau=tau)  # [B, C, L, 3]
            self.last_gates = gates.detach()
            g_cnn = gates[..., 0]   # [B, C, L]
            g_pw = gates[..., 1]    # [B, C, L]
            g_attn = gates[..., 2]  # [B, C, L]
            f_fused = g_cnn * f_cnn + g_pw * f_pw + g_attn * f_attn
            
        y = x + f_fused
        if need_weights:
            return y, attn_w
        return y
        
    def get_router_entropy(self, gates=None):
        if gates is None:
            gates = self.last_gates
        if gates is None:
            return 0.0
        # Compute mean routing weights across batch and channels (and pos if lr_cpg)
        if gates.ndim == 3:  # [B, C, 3]
            g_mean = gates.mean(dim=(0, 1))  # [3]
        elif gates.ndim == 4:  # [B, C, L, 3]
            g_mean = gates.mean(dim=(0, 1, 2))  # [3]
        entropy = -torch.sum(g_mean * torch.log(g_mean + 1e-8))
        return entropy

block128 = AdaptiveTriBranchBlock(d_model=128, router_type='cagc')
x = torch.randn(1, 128, 50, requires_grad=True)
y = block128(x)
loss = y.sum()
loss.backward()

zero_grads = [name for name, p in block128.named_parameters() if p.grad is None or (p.grad == 0).all()]
print("CAGC Block 128 Total params:", sum(p.numel() for p in block128.parameters()))
print("CAGC Block 128 Zero grads:", zero_grads)
print("Router entropy:", block128.get_router_entropy().item())
