"""
DeepTensor++ Unified Model Factory
==================================
Centralized factory module for constructing, configuring, inspecting, and serializing
DeepTensor baseline CNN, Hybrid CNN-Attention, and Transformer-dominant architectures
for both 2-factor matrix (U x V) and 3-way PARAFAC/CP tensor (U (x) V (x) W) factorizations.
"""

import os
from typing import Optional, Dict, Any, Union, Tuple, List

import torch
import torch.nn as nn

from .deeptensor import DeepTensor, DeepTensorCP, SimpleForward2D, SimpleForward1D, SimpleForwardMLP, UNetND
from .deep_prior import get_net
from .deep_decoder import decodernw, decodernw1d
from .attention import (
    HybridSpectralNet,
    SpectralSelfAttention,
    SpectralPointwiseControl,
    SpectralTransformer,
    ControlSpectralTransformer,
    AdaptiveHybridSpectralNet,
    AdaptiveTriBranchBlock,
)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_spatial_net(
    n_inputs: int,
    rank: int,
    net_type: str = 'skip',
    pad: str = 'reflection',
    upsample_mode: str = 'bilinear',
    skip_n33d: int = 128,
    skip_n33u: int = 128,
    num_scales: int = 5,
    init_nconv: int = 128,
    **kwargs: Any,
) -> nn.Module:
    """
    Construct spatial factor generator network U(H, W).

    Args:
        n_inputs: Number of input channels (latent dimension, typically rank).
        rank: Number of output channels (low-rank components).
        net_type: 'skip' (U-Net with skip connections), 'unet' (UNetND), 'simple2d', or 'dd' (Deep Decoder).
        pad: Padding mode ('reflection' or 'zero').
        upsample_mode: Upsampling mode ('bilinear', 'nearest').
        skip_n33d: Channels in downsampling path for skip network.
        skip_n33u: Channels in upsampling path for skip network.
        num_scales: Number of scales/levels in skip network.
        init_nconv: Initial filter width for UNetND.

    Returns:
        Instantiated PyTorch neural network.
    """
    net_type_lower = net_type.lower()
    # Filter out spectral/attention specific kwargs that spatial get_net does not accept
    spatial_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ('router_type', 'pe_type', 'tau', 'heads')
    }
    if net_type_lower in ('skip', 'dip', 'cnn'):
        return get_net(
            n_inputs,
            'skip',
            pad,
            upsample_mode=upsample_mode,
            skip_n33d=skip_n33d,
            skip_n33u=skip_n33u,
            num_scales=num_scales,
            n_channels=rank,
            **spatial_kwargs,
        )
    elif net_type_lower == 'unet':
        return UNetND(n_inputs, rank, ndim=2, init_nconv=init_nconv)
    elif net_type_lower == 'simple2d':
        nconvs = kwargs.get('nconvs', [128, 128, 128])
        return SimpleForward2D(n_inputs, rank, nconvs)
    elif net_type_lower in ('dd', 'deep_decoder'):
        nchans = kwargs.get('nchans', [init_nconv] * 5)
        return decodernw(rank, nchans)
    else:
        raise ValueError(f"Unknown spatial net_type '{net_type}'. Expected one of: 'skip', 'unet', 'simple2d', 'dd'.")


def build_spectral_net(
    n_inputs: int,
    rank: int,
    seq_len: int,
    net_type: str = 'skip1d',
    pad: str = 'reflection',
    upsample_mode: str = 'linear',
    skip_n33d: int = 128,
    skip_n33u: int = 128,
    num_scales: int = 5,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ffn: int = 512,
    init_nconv: int = 128,
    **kwargs: Any,
) -> nn.Module:
    """
    Construct spectral / temporal / 1D factor generator network V(L).

    Args:
        n_inputs: Number of input channels (latent dimension, typically rank).
        rank: Number of output channels (low-rank components).
        seq_len: Sequence length L (spectral wavelengths, temporal frames, or slice count).
        net_type:
            - 'skip1d' / 'cnn' / 'baseline': Standard 1D CNN Skip network.
            - 'hybrid' / 'hybrid_attention': U-Net body + Multi-Head Self-Attention + FFN.
            - 'hybrid_control': Parameter-matched pointwise control for hybrid attention.
            - 'transformer' / 'spectral_transformer': Pre-LN Transformer with ALiBi relative bias.
            - 'transformer_control': Parameter-matched pointwise control for Transformer.
            - 'unet1d': 1D UNetND.
            - 'dd1d' / 'deep_decoder1d': 1D Deep Decoder.
        pad: Padding mode ('reflection' or 'zero').
        upsample_mode: Upsampling mode ('linear', 'nearest').
        skip_n33d: Downsampling channels.
        skip_n33u: Upsampling channels.
        num_scales: Number of scales.
        d_model: Latent feature dimension for attention/transformer blocks.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ffn: Feed-forward expansion dimension.
        init_nconv: Initial filter count for UNetND.

    Returns:
        Instantiated PyTorch neural network.
    """
    net_type_lower = net_type.lower()
    clean_get_net_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ('router_type', 'pe_type', 'tau', 'heads', 'gating_floor')
    }
    
    if net_type_lower in ('skip1d', 'cnn', 'baseline', 'dip'):
        return get_net(
            n_inputs,
            'skip1d',
            pad,
            upsample_mode=upsample_mode,
            skip_n33d=skip_n33d,
            skip_n33u=skip_n33u,
            num_scales=num_scales,
            n_channels=rank,
            **clean_get_net_kwargs,
        )
    elif net_type_lower in ('hybrid', 'hybrid_attention'):
        base_net = get_net(
            n_inputs,
            'skip1d',
            pad,
            upsample_mode=upsample_mode,
            skip_n33d=skip_n33d,
            skip_n33u=skip_n33u,
            num_scales=num_scales,
            n_channels=rank,
            **clean_get_net_kwargs,
        )
        return HybridSpectralNet(
            base_net,
            variant='hybrid',
            seq_len=seq_len,
            d_model=skip_n33u,
            heads=kwargs.get('heads', 2),
        )
    elif net_type_lower in ('hybrid_control', 'control_hybrid', 'pointwise_control'):
        base_net = get_net(
            n_inputs,
            'skip1d',
            pad,
            upsample_mode=upsample_mode,
            skip_n33d=skip_n33d,
            skip_n33u=skip_n33u,
            num_scales=num_scales,
            n_channels=rank,
            **clean_get_net_kwargs,
        )
        return HybridSpectralNet(
            base_net,
            variant='control',
            seq_len=seq_len,
            d_model=skip_n33u,
        )
    elif net_type_lower in ('cnn_control', 'conv_control', 'hybrid_cnn_control'):
        base_net = get_net(
            n_inputs,
            'skip1d',
            pad,
            upsample_mode=upsample_mode,
            skip_n33d=skip_n33d,
            skip_n33u=skip_n33u,
            num_scales=num_scales,
            n_channels=rank,
            **clean_get_net_kwargs,
        )
        return HybridSpectralNet(
            base_net,
            variant='cnn_control',
            seq_len=seq_len,
            d_model=skip_n33u,
        )
    elif net_type_lower in ('transformer', 'spectral_transformer'):
        return SpectralTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ffn=d_ffn,
            seq_len=seq_len,
            rank=rank,
        )
    elif net_type_lower in ('transformer_control', 'control_transformer', 'control'):
        return ControlSpectralTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ffn=d_ffn,
            seq_len=seq_len,
            rank=rank,
        )
    elif net_type_lower in (
        'adaptive_hybrid', 'adaptive_hybrid_v1', 'adaptive', 'adaptive_tri_hybrid',
        'static_tri_hybrid', 'static_hybrid', 'static',
        'adaptive_hybrid_lr_cpg', 'lr_cpg',
        'cnn_only_tri_hybrid', 'cnn_only',
        'pointwise_only_tri_hybrid', 'pointwise_only', 'pw_only',
        'attention_only_tri_hybrid', 'attention_only', 'attn_only',
    ):
        base_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ('router_type', 'pe_type', 'tau', 'heads', 'gating_floor')
        }
        base_net = get_net(
            n_inputs,
            'skip1d',
            pad,
            upsample_mode=upsample_mode,
            skip_n33d=skip_n33d,
            skip_n33u=skip_n33u,
            num_scales=num_scales,
            n_channels=rank,
            **base_kwargs,
        )

        # Infer router_type if not explicitly passed
        router_type = kwargs.get('router_type', None)
        if router_type is None:
            if net_type_lower in ('static_tri_hybrid', 'static_hybrid', 'static'):
                router_type = 'static'
            elif net_type_lower in ('adaptive_hybrid_lr_cpg', 'lr_cpg'):
                router_type = 'lr_cpg'
            elif net_type_lower in ('cnn_only_tri_hybrid', 'cnn_only'):
                router_type = 'cnn_only'
            elif net_type_lower in ('pointwise_only_tri_hybrid', 'pointwise_only', 'pw_only'):
                router_type = 'pointwise_only'
            elif net_type_lower in ('attention_only_tri_hybrid', 'attention_only', 'attn_only'):
                router_type = 'attention_only'
            else:
                router_type = 'cagc'

        return AdaptiveHybridSpectralNet(
            base_net,
            variant=net_type_lower,
            router_type=router_type,
            pe_type=kwargs.get('pe_type', 'sin_pe'),
            seq_len=seq_len,
            d_model=skip_n33u,
            heads=kwargs.get('heads', 2),
            tau=kwargs.get('tau', 1.0),
            gating_floor=kwargs.get('gating_floor', 0.0),
        )
    elif net_type_lower == 'unet1d':
        return UNetND(n_inputs, rank, ndim=1, init_nconv=init_nconv)
    elif net_type_lower in ('dd1d', 'deep_decoder1d'):
        nchans = kwargs.get('nchans', [init_nconv] * 5)
        return decodernw1d(rank, nchans)
    else:
        raise ValueError(
            f"Unknown spectral net_type '{net_type}'. Expected one of: "
            "'skip1d', 'cnn', 'baseline', 'hybrid', 'hybrid_control', "
            "'transformer', 'transformer_control', 'adaptive_hybrid', "
            "'static_tri_hybrid', 'adaptive_hybrid_lr_cpg', 'unet1d', 'dd1d'."
        )


def build_deeptensor_model(
    rank: int,
    seq_len: int,
    spatial_type: str = 'skip',
    spectral_type: str = 'skip1d',
    n_inputs: Optional[int] = None,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ffn: int = 512,
    skip_n33d: int = 128,
    skip_n33u: int = 128,
    num_scales: int = 5,
    **kwargs: Any,
) -> DeepTensor:
    """
    Construct a complete 2-factor DeepTensor model (U x V) with chosen spatial and spectral backbones.

    Args:
        rank: Decomposition rank R.
        seq_len: Length of spectral/temporal dimension L.
        spatial_type: Spatial network backbone ('skip', 'unet', 'simple2d', 'dd').
        spectral_type: Spectral network backbone ('skip1d', 'hybrid', 'transformer', 'control').
        n_inputs: Latent input channels (defaults to rank).
        d_model: Transformer/attention feature dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ffn: Feed-forward expansion dimension.
        skip_n33d: Conv channels in downsampler.
        skip_n33u: Conv channels in upsampler.
        num_scales: Number of multiscale stages.

    Returns:
        Configured DeepTensor instance.
    """
    if n_inputs is None:
        n_inputs = rank

    spatial_net = build_spatial_net(
        n_inputs=n_inputs,
        rank=rank,
        net_type=spatial_type,
        skip_n33d=skip_n33d,
        skip_n33u=skip_n33u,
        num_scales=num_scales,
        **kwargs,
    )
    spectral_net = build_spectral_net(
        n_inputs=n_inputs,
        rank=rank,
        seq_len=seq_len,
        net_type=spectral_type,
        skip_n33d=skip_n33d,
        skip_n33u=skip_n33u,
        num_scales=num_scales,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        **kwargs,
    )
    return DeepTensor(spatial_net=spatial_net, spectral_net=spectral_net, rank=rank)


def build_deeptensor_cp_model(
    rank: int,
    lengths: Tuple[int, int, int],
    u_type: str = 'skip1d',
    v_type: str = 'skip1d',
    w_type: str = 'skip1d',
    n_inputs: Optional[int] = None,
    learnable_core: bool = True,
    normalize_core: bool = True,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ffn: int = 512,
    skip_n33d: int = 128,
    skip_n33u: int = 128,
    num_scales: int = 5,
    **kwargs: Any,
) -> DeepTensorCP:
    """
    Construct a complete 3-way DeepTensorCP model (U (x) V (x) W) with chosen factor backbones.

    Args:
        rank: Decomposition rank R.
        lengths: Tuple (L_u, L_v, L_w) of lengths for each mode (e.g. (192, 168, 160)).
        u_type: Architecture for mode 1 generator.
        v_type: Architecture for mode 2 generator.
        w_type: Architecture for mode 3 generator (e.g. 'hybrid', 'transformer', 'skip1d').
        n_inputs: Latent input channels (defaults to rank).
        learnable_core: Whether the rank core vector is trainable.
        normalize_core: Whether to divide core by rank during reconstruction.
        d_model: Attention/transformer feature dimension.
        n_heads: Attention heads.
        n_layers: Transformer layers.
        d_ffn: Transformer FFN dimension.

    Returns:
        Configured DeepTensorCP instance.
    """
    if n_inputs is None:
        n_inputs = rank

    lu, lv, lw = lengths

    u_net = build_spectral_net(
        n_inputs=n_inputs,
        rank=rank,
        seq_len=lu,
        net_type=u_type,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        skip_n33d=skip_n33d,
        skip_n33u=skip_n33u,
        num_scales=num_scales,
        **kwargs,
    )
    v_net = build_spectral_net(
        n_inputs=n_inputs,
        rank=rank,
        seq_len=lv,
        net_type=v_type,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        skip_n33d=skip_n33d,
        skip_n33u=skip_n33u,
        num_scales=num_scales,
        **kwargs,
    )
    w_net = build_spectral_net(
        n_inputs=n_inputs,
        rank=rank,
        seq_len=lw,
        net_type=w_type,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        skip_n33d=skip_n33d,
        skip_n33u=skip_n33u,
        num_scales=num_scales,
        **kwargs,
    )

    return DeepTensorCP(
        u_net=u_net,
        v_net=v_net,
        w_net=w_net,
        rank=rank,
        learnable_core=learnable_core,
        normalize_core=normalize_core,
    )


def save_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint safely to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics or {},
        'config': config or {},
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load model checkpoint safely from disk and return metadata."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
    try:
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
