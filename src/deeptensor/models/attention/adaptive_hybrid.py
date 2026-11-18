"""
DeepTensor++ Adaptive Hybrid DeepTensor v1 Architecture Core
============================================================
Implements the Tri-Branch Adaptive Hybrid module combining:
  1. Local Convolutional Branch (Depthwise-Separable 1D ResBlock, k=3, 5)
  2. Pointwise Channel/Rank Mixing Branch (1x1 Linear bottleneck + GELU)
  3. Global Dependency Modelling Branch (Pre-LN Multi-Head Self-Attention + FFN)
Conditioned via an Input-Dependent Routing Mechanism:
  - Channel-Aware Global Context Gate (CAGC) with cosine temperature annealing
  - Optional Low-Rank Channel-Position Gate (LR-CPG) as position-aware ablation
  - Static Tri-Hybrid baseline (fixed uniform weights g1=g2=g3=1/3)

Designed with exact mathematical parameter budgeting to match historical
DeepTensor++ hybrid module capacities down to 0 parameter delta:
  - d_model = 128: exactly 132,480 module parameters (Figs 09, 10, 12)
  - d_model = 64:  exactly 49,984 module parameters (Fig 11)
"""

import math
from typing import Optional, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cosine_temperature(
    epoch: int,
    max_epochs: int,
    tau_start: float = 2.0,
    tau_min: float = 0.5,
) -> float:
    """
    Compute cosine temperature annealing schedule for router softmax.

    tau(t) = tau_min + 0.5 * (tau_start - tau_min) * (1 + cos(pi * t / T))

    Args:
        epoch: Current optimization epoch t in [0, max_epochs].
        max_epochs: Total optimization epochs T.
        tau_start: Initial exploratory temperature (default 2.0).
        tau_min: Minimum asymptotic temperature (default 0.5).

    Returns:
        Current scalar temperature tau.
    """
    if max_epochs <= 0:
        return tau_min
    progress = min(max(float(epoch) / float(max_epochs), 0.0), 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tau_min + (tau_start - tau_min) * cosine_decay


def _get_sinusoidal_pe(seq_len: int, d_model: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Generate fixed sinusoidal positional encodings [1, seq_len, d_model]."""
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class ChannelAwareGlobalContextGate(nn.Module):
    """
    Channel-Aware Global Context Gate (CAGC Router).

    Squeezes temporal/spectral sequence via Global Average Pooling:
        s = (1/L) sum_{t=1}^L x_{:, t} in R^C
    Maps through a two-layer bottleneck MLP:
        u = W2 * GELU(W1 * s + b1) + b2, with u in R^{3C} -> reshaped to R^{C x 3}
    Computes per-channel temperature-scaled softmax:
        g_b(c) = exp(u_{c, b} / tau) / sum_j exp(u_{c, j} / tau)
    """

    def __init__(self, d_model: int = 128, d_route: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_route = d_route
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_route),
            nn.GELU(),
            nn.Linear(d_route, 3 * d_model),
        )

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        Forward pass for CAGC router.

        Args:
            x: Input tensor [B, C, L].
            tau: Softmax temperature parameter (tau > 0).

        Returns:
            Routing gates tensor [B, C, 3], where dim -1 corresponds to
            [CNN, Pointwise, Attention] and sums to 1.0 for every channel.
        """
        tau = max(float(tau), 1e-4)
        B, C, L = x.shape
        s = x.mean(dim=-1)  # [B, C]
        u = self.mlp(s)      # [B, 3 * C]
        u = u.view(B, C, 3)  # [B, C, 3]
        gates = F.softmax(u / tau, dim=-1)  # [B, C, 3]
        return gates


class LowRankChannelPositionGate(nn.Module):
    """
    Low-Rank Factorized Channel-Position Gate (LR-CPG Router).

    Position-aware ablation extending CAGC by decomposing full C x L x 3 gating
    into rank-1 channel and position terms:
        G(c, t, b) = Softmax_b( U_chan(c, b) + U_pos(t, b) )

    U_chan is generated from sequence-averaged context s in R^C via MLP.
    U_pos is generated via lightweight depthwise-separable 1D convolution over x.
    """

    def __init__(self, d_model: int = 128, d_route: int = 16):
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

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        Forward pass for LR-CPG router.

        Args:
            x: Input tensor [B, C, L].
            tau: Softmax temperature parameter.

        Returns:
            Position-aware routing gates [B, C, L, 3], summing to 1.0 across dim -1.
        """
        tau = max(float(tau), 1e-4)
        B, C, L = x.shape
        s = x.mean(dim=-1)  # [B, C]
        u_chan = self.chan_gate(s).view(B, C, 1, 3)  # [B, C, 1, 3]
        # pos_gate outputs [B, 3, L] -> transpose to [B, L, 3] -> view [B, 1, L, 3]
        u_pos = self.pos_gate(x).permute(0, 2, 1).unsqueeze(1)  # [B, 1, L, 3]
        u_total = u_chan + u_pos  # [B, C, L, 3] via broadcast addition
        gates = F.softmax(u_total / tau, dim=-1)  # [B, C, L, 3]
        return gates


class LocalCNNBranch(nn.Module):
    """
    Branch 1: Local Convolutional Feature Extraction (F_CNN).

    Depthwise-separable 1D residual block:
      GroupNorm(4, C) -> DWConv(k=3, pad=1) -> PWConv(1x1, C -> c_cnn) -> GELU ->
      DWConv(k=5, pad=2) -> PWConv(1x1, c_cnn -> C)
    """

    def __init__(self, d_model: int = 128, c_cnn: int = 143):
        super().__init__()
        self.d_model = d_model
        self.c_cnn = c_cnn
        self.gn = nn.GroupNorm(num_groups=4, num_channels=d_model)
        self.dw1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pw1 = nn.Conv1d(d_model, c_cnn, kernel_size=1)
        self.act = nn.GELU()
        self.dw2 = nn.Conv1d(c_cnn, c_cnn, kernel_size=5, padding=2, groups=c_cnn)
        self.pw2 = nn.Conv1d(c_cnn, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass [B, C, L] -> [B, C, L]."""
        h = self.gn(x)
        h = self.dw1(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.dw2(h)
        h = self.pw2(h)
        return h


class PointwiseBranch(nn.Module):
    """
    Branch 2: Pointwise Channel & Rank Mixing (F_PW).

    Computes position-independent cross-channel/rank feature interaction:
      LayerNorm(C) -> Linear(C -> c_pw) -> GELU -> Linear(c_pw -> C)
    """

    def __init__(self, d_model: int = 128, c_pw: int = 135):
        super().__init__()
        self.d_model = d_model
        self.c_pw = c_pw
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, c_pw)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(c_pw, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass [B, C, L] -> [B, C, L]."""
        xt = x.transpose(1, 2)  # [B, L, C]
        h = self.linear2(self.act(self.linear1(self.norm(xt))))
        return h.transpose(1, 2)


class GlobalAttentionBranch(nn.Module):
    """
    Branch 3: Global Dependency Modelling (F_Attn).

    Pre-LN bottleneck Multi-Head Self-Attention with projection bottleneck:
      ProjIn(C -> d_attn) -> MultiheadAttention(d_attn, heads=heads) ->
      ProjOut(d_attn -> C) -> Pre-LN FFN(C -> d_attn -> C)
    """

    def __init__(self, d_model: int = 128, d_attn: int = 64, heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_attn = d_attn
        self.heads = heads

        self.norm1 = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, d_attn)
        self.mha = nn.MultiheadAttention(embed_dim=d_attn, num_heads=heads, batch_first=True)
        self.proj_out = nn.Linear(d_attn, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_attn),
            nn.GELU(),
            nn.Linear(d_attn, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass for global attention branch.

        Args:
            x: Input tensor [B, C, L].
            pe: Optional positional encoding [1, L, C] or [B, L, C].
            need_weights: Whether to return attention map weights.

        Returns:
            Branch output [B, C, L], and optionally attention weights.
        """
        xt = x.transpose(1, 2)  # [B, L, C]
        if pe is not None:
            xt = xt + pe
        z = self.proj_in(self.norm1(xt))  # [B, L, d_attn]
        attn_out, attn_weights = self.mha(z, z, z, need_weights=need_weights)  # [B, L, d_attn]
        a = self.proj_out(attn_out)  # [B, L, C]
        ffn_out = self.ffn(self.norm2(a))  # [B, L, C]
        out = (a + ffn_out).transpose(1, 2)  # [B, C, L]

        if need_weights:
            return out, attn_weights
        return out


class AdaptiveTriBranchBlock(nn.Module):
    """
    Unified Adaptive Tri-Branch DeepTensor Block.

    Combines:
      - Local CNN Branch
      - Pointwise Mixing Branch
      - Global Attention Branch
    under an Input-Conditioned Router with weighted residual recombination:
      y = x + sum_{b=1}^3 g_b * F_b(x)

    Parameter Budgets:
      - d_model = 128: exactly 132,480 trainable parameters (Figs 09, 10, 12)
      - d_model = 64:  exactly 49,984 trainable parameters (Fig 11)
    """

    def __init__(
        self,
        d_model: int = 128,
        seq_len: int = 260,
        router_type: str = 'cagc',
        pe_type: str = 'sin_pe',
        tau: float = 1.0,
        heads: int = 2,
        gating_floor: float = 0.0,
    ):
        """
        Args:
            d_model: Latent feature dimension (128 or 64).
            seq_len: Canonical sequence length L for positional encoding.
            router_type: 'cagc' (Channel-Aware Global Context), 'lr_cpg' (Low-Rank Channel-Position),
                         'static' (fixed 1/3 equal weighting), 'cnn_only' (fixed [1, 0, 0]),
                         'pointwise_only' (fixed [0, 1, 0]), or 'attention_only' (fixed [0, 0, 1]).
            pe_type: 'sin_pe' (sinusoidal encoding), 'no_pe' (zeros), or 'phys_pe' (external coordinates).
            tau: Initial router temperature.
            heads: Number of attention heads (default 2).
            gating_floor: Minimum gate weight constraint epsilon in [0, 1/3) preventing starvation.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.router_type = router_type.lower()
        self.pe_type = pe_type.lower()
        self.tau = float(tau)
        self.heads = heads
        self.gating_floor = float(gating_floor)

        # Derive exact parameter-matched architectural dimensions
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
            raise ValueError(f"Unsupported d_model: {d_model}. Expected 128 or 64.")

        # Instantiate Router
        if self.router_type == 'cagc':
            self.router = ChannelAwareGlobalContextGate(d_model=d_model, d_route=d_route)
        elif self.router_type == 'lr_cpg':
            self.router = LowRankChannelPositionGate(d_model=d_model, d_route=d_route)
        elif self.router_type in ('static', 'equal', 'cnn_only', 'pointwise_only', 'pw_only', 'attention_only', 'attn_only'):
            self.router = None
        else:
            raise ValueError(
                f"Unknown router_type: {router_type}. Expected 'cagc', 'lr_cpg', 'static', "
                f"'cnn_only', 'pointwise_only', or 'attention_only'."
            )

        # Instantiate Three Branches
        self.cnn_branch = LocalCNNBranch(d_model=d_model, c_cnn=c_cnn)
        self.pw_branch = PointwiseBranch(d_model=d_model, c_pw=c_pw)
        self.attn_branch = GlobalAttentionBranch(d_model=d_model, d_attn=d_attn, heads=heads)

        # Precompute sinusoidal positional encoding buffer
        pe = _get_sinusoidal_pe(seq_len, d_model)
        self.register_buffer('pe', pe)

        # Cache for telemetry and regularizer computation
        self.last_gates: Optional[torch.Tensor] = None
        self.last_entropy: float = 0.0

    def set_temperature(self, tau: float) -> None:
        """Update router temperature."""
        self.tau = max(float(tau), 1e-4)

    def _resolve_pe(self, x: torch.Tensor, custom_pe: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Resolve positional encoding tensor based on pe_type and sequence length."""
        if self.pe_type == 'no_pe':
            return None
        if self.pe_type == 'phys_pe':
            return custom_pe

        # Standard sinusoidal PE
        L = x.shape[2]
        if L <= self.pe.shape[1]:
            return self.pe[:, :L, :].to(device=x.device, dtype=x.dtype)
        else:
            return _get_sinusoidal_pe(L, self.d_model, device=x.device).to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        custom_pe: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass for the Adaptive Tri-Branch Block.

        Args:
            x: Input tensor [B, C, L].
            custom_pe: Optional positional encoding or physical coordinates [1, L, C].
            tau: Optional override temperature for this forward call.
            need_weights: Whether to return attention weights.

        Returns:
            Output tensor [B, C, L], and optionally attention weights.
        """
        current_tau = self.tau if tau is None else max(float(tau), 1e-4)
        pe = self._resolve_pe(x, custom_pe)
        B, C, L = x.shape

        # Branch isolation clamped modes (fast path)
        if self.router_type == 'cnn_only':
            f_cnn = self.cnn_branch(x)
            f_fused = f_cnn
            self.last_gates = torch.tensor([1.0, 0.0, 0.0], device=x.device, dtype=x.dtype).expand(B, C, 3)
            self.last_entropy = 0.0
            attn_w = None
        elif self.router_type in ('pointwise_only', 'pw_only'):
            f_pw = self.pw_branch(x)
            f_fused = f_pw
            self.last_gates = torch.tensor([0.0, 1.0, 0.0], device=x.device, dtype=x.dtype).expand(B, C, 3)
            self.last_entropy = 0.0
            attn_w = None
        elif self.router_type in ('attention_only', 'attn_only'):
            if need_weights:
                f_attn, attn_w = self.attn_branch(x, pe=pe, need_weights=True)
            else:
                f_attn = self.attn_branch(x, pe=pe, need_weights=False)
                attn_w = None
            f_fused = f_attn
            self.last_gates = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype).expand(B, C, 3)
            self.last_entropy = 0.0
        else:
            # Tri-branch computation
            f_cnn = self.cnn_branch(x)
            f_pw = self.pw_branch(x)
            if need_weights:
                f_attn, attn_w = self.attn_branch(x, pe=pe, need_weights=True)
            else:
                f_attn = self.attn_branch(x, pe=pe, need_weights=False)
                attn_w = None

            if self.router_type in ('static', 'equal'):
                # Fixed equal weights 1/3
                g_cnn = 1.0 / 3.0
                g_pw = 1.0 / 3.0
                g_attn = 1.0 / 3.0
                f_fused = g_cnn * f_cnn + g_pw * f_pw + g_attn * f_attn
                self.last_gates = torch.full((B, C, 3), 1.0 / 3.0, device=x.device, dtype=x.dtype)
                self.last_entropy = math.log(3.0)
            elif self.router_type == 'cagc':
                # Channel-Aware Global Context [B, C, 3]
                gates = self.router(x, tau=current_tau)
                if self.gating_floor > 0.0:
                    eps = self.gating_floor
                    gates = (1.0 - 3.0 * eps) * gates + eps
                self.last_gates = gates
                g_cnn = gates[..., 0].unsqueeze(-1)   # [B, C, 1]
                g_pw = gates[..., 1].unsqueeze(-1)    # [B, C, 1]
                g_attn = gates[..., 2].unsqueeze(-1)  # [B, C, 1]
                f_fused = g_cnn * f_cnn + g_pw * f_pw + g_attn * f_attn
            elif self.router_type == 'lr_cpg':
                # Low-Rank Channel-Position [B, C, L, 3]
                gates = self.router(x, tau=current_tau)
                if self.gating_floor > 0.0:
                    eps = self.gating_floor
                    gates = (1.0 - 3.0 * eps) * gates + eps
                self.last_gates = gates
                g_cnn = gates[..., 0]   # [B, C, L]
                g_pw = gates[..., 1]    # [B, C, L]
                g_attn = gates[..., 2]  # [B, C, L]
                f_fused = g_cnn * f_cnn + g_pw * f_pw + g_attn * f_attn

        y = x + f_fused

        if need_weights:
            return y, attn_w
        return y

    def compute_router_entropy(self, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Shannon entropy of the average routing distribution across channels:
            g_bar = (1 / (B * C)) sum_{b, c} g_{b, c} in R^3
            H(g_bar) = - sum_{j=1}^3 g_bar_j * log(g_bar_j + 1e-8)

        Returns:
            Scalar PyTorch tensor with non-negative Shannon entropy H in [0, ln(3)].
        """
        if gates is None:
            gates = self.last_gates
        if gates is None:
            return torch.tensor(math.log(3.0), device=self.pe.device)

        if gates.ndim == 3:  # [B, C, 3]
            g_bar = gates.mean(dim=(0, 1))  # [3]
        elif gates.ndim == 4:  # [B, C, L, 3]
            g_bar = gates.mean(dim=(0, 1, 2))  # [3]
        else:
            g_bar = gates.mean(dim=list(range(gates.ndim - 1)))

        entropy = -torch.sum(g_bar * torch.log(g_bar + 1e-8))
        return entropy

    def compute_channel_entropy(self, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute mean Shannon entropy across individual channels:
            H_c = - sum_{b=1}^3 w_{c, b} * log(w_{c, b} + 1e-8)
            H_mean = (1 / C) sum_c H_c

        Returns:
            Scalar PyTorch tensor with mean channel entropy in [0, ln(3)].
        """
        if gates is None:
            gates = self.last_gates
        if gates is None:
            return torch.tensor(math.log(3.0), device=self.pe.device)

        h_per_channel = -torch.sum(gates * torch.log(gates + 1e-8), dim=-1)
        return h_per_channel.mean()

    def compute_negative_entropy_loss(
        self,
        gates: Optional[torch.Tensor] = None,
        per_channel: bool = True,
    ) -> torch.Tensor:
        """
        Compute negative entropy loss for loss minimization:
            L_entropy = -H(g)
        Minimizing this loss maximizes entropy, preventing branch collapse.
        """
        if per_channel:
            return -self.compute_channel_entropy(gates)
        return -self.compute_router_entropy(gates)

    def get_routing_telemetry(self) -> Dict[str, Any]:
        """
        Return structured telemetry metrics for logging and monitoring.

        Returns:
            Dictionary with mean gate weights, scalar entropy, and current temperature.
        """
        telemetry: Dict[str, Any] = {
            "router_type": self.router_type,
            "tau": self.tau,
            "pe_type": self.pe_type,
        }
        if self.last_gates is not None:
            with torch.no_grad():
                if self.last_gates.ndim == 3:
                    g_mean = self.last_gates.mean(dim=(0, 1)).cpu().tolist()
                elif self.last_gates.ndim == 4:
                    g_mean = self.last_gates.mean(dim=(0, 1, 2)).cpu().tolist()
                else:
                    g_mean = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
                entropy = float(self.compute_router_entropy().item())
                telemetry["mean_gate_weights"] = {
                    "cnn": g_mean[0],
                    "pointwise": g_mean[1],
                    "attention": g_mean[2],
                }
                telemetry["entropy"] = entropy
        else:
            telemetry["mean_gate_weights"] = {"cnn": 1.0 / 3.0, "pointwise": 1.0 / 3.0, "attention": 1.0 / 3.0}
            telemetry["entropy"] = math.log(3.0)
        return telemetry


class AdaptiveHybridSpectralNet(nn.Module):
    """
    Wrapper module for integrating AdaptiveTriBranchBlock into DeepTensor factor networks.

    Decomposes an existing 1D sequential network (e.g. Skip1D) into:
      - self.body: Feature downsampling and spatial/temporal processing
      - self.adaptive_block: AdaptiveTriBranchBlock (CNN + PW + Attn with CAGC router)
      - self.final_proj: Output projection to rank channels
    """

    def __init__(
        self,
        original_spec_net: nn.Sequential,
        variant: str = 'adaptive_hybrid_v1',
        router_type: str = 'cagc',
        pe_type: str = 'sin_pe',
        seq_len: int = 260,
        d_model: Optional[int] = None,
        heads: int = 2,
        tau: float = 1.0,
        gating_floor: float = 0.0,
    ):
        super().__init__()
        if not isinstance(original_spec_net, nn.Sequential):
            raise ValueError("Expected original_spec_net to be an nn.Sequential")

        modules = list(original_spec_net.children())
        self.body = nn.Sequential(*modules[:-1])
        self.final_proj = modules[-1]

        # Auto-detect d_model from final_proj in_channels if not provided
        if d_model is None:
            detected = None
            for m in self.final_proj.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    detected = m.in_channels if hasattr(m, 'in_channels') else m.in_features
                    break
            d_model = detected if detected is not None else 128

        # Parse variant overrides
        variant_lower = variant.lower()
        if variant_lower in ('static_tri_hybrid', 'static', 'equal'):
            router_type = 'static'
        elif variant_lower in ('adaptive_hybrid_lr_cpg', 'lr_cpg'):
            router_type = 'lr_cpg'
        elif variant_lower in ('cnn_only_tri_hybrid', 'cnn_only'):
            router_type = 'cnn_only'
        elif variant_lower in ('pointwise_only_tri_hybrid', 'pointwise_only', 'pw_only'):
            router_type = 'pointwise_only'
        elif variant_lower in ('attention_only_tri_hybrid', 'attention_only', 'attn_only'):
            router_type = 'attention_only'

        self.adaptive_block = AdaptiveTriBranchBlock(
            d_model=d_model,
            seq_len=seq_len,
            router_type=router_type,
            pe_type=pe_type,
            tau=tau,
            heads=heads,
            gating_floor=gating_floor,
        )

    def set_temperature(self, tau: float) -> None:
        """Update temperature in adaptive block."""
        self.adaptive_block.set_temperature(tau)

    def get_routing_telemetry(self) -> Dict[str, Any]:
        """Retrieve routing telemetry from adaptive block."""
        return self.adaptive_block.get_routing_telemetry()

    def compute_router_entropy(self, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Shannon entropy from adaptive block."""
        return self.adaptive_block.compute_router_entropy(gates)

    def compute_channel_entropy(self, gates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute mean channel Shannon entropy from adaptive block."""
        return self.adaptive_block.compute_channel_entropy(gates)

    def compute_negative_entropy_loss(
        self,
        gates: Optional[torch.Tensor] = None,
        per_channel: bool = True,
    ) -> torch.Tensor:
        """Compute negative entropy loss for regularization."""
        return self.adaptive_block.compute_negative_entropy_loss(gates, per_channel=per_channel)

    def forward(
        self,
        x: torch.Tensor,
        custom_pe: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward pass through body -> adaptive tri-branch block -> final projection."""
        x = self.body(x)
        if need_weights:
            x, weights = self.adaptive_block(x, custom_pe=custom_pe, tau=tau, need_weights=True)
        else:
            x = self.adaptive_block(x, custom_pe=custom_pe, tau=tau, need_weights=False)
            weights = None
        x = self.final_proj(x)

        if need_weights:
            return x, weights
        return x


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

