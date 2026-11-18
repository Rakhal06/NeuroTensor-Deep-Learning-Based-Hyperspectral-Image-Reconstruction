"""
Phase 2: Comprehensive Parameter Verification across All Four Applications
=========================================================================
Instantiates complete DeepTensor / DeepTensorCP models and AdaptiveTriBranchBlock
for Figures 09, 10, 11, and 12, comparing directly against historical hybrid budgets.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import torch
from deeptensor.models.factory import (
    build_deeptensor_model,
    build_deeptensor_cp_model,
    count_parameters,
)
from deeptensor.models.attention import (
    AdaptiveTriBranchBlock,
    ChannelAwareGlobalContextGate,
    LowRankChannelPositionGate,
    LocalCNNBranch,
    PointwiseBranch,
    GlobalAttentionBranch,
)

def verify_modules():
    print("=" * 80)
    print("PHASE 2: ADAPTIVE TRI-BRANCH MODULE PARAMETER VERIFICATION")
    print("=" * 80)
    
    # 1. d_model = 128 (Figs 09, 10, 12)
    b128 = AdaptiveTriBranchBlock(d_model=128, router_type='cagc')
    r128_params = count_parameters(b128.router)
    cnn128_params = count_parameters(b128.cnn_branch)
    pw128_params = count_parameters(b128.pw_branch)
    at128_params = count_parameters(b128.attn_branch)
    tot128 = count_parameters(b128)
    target128 = 132480
    
    print("\n--- Configuration A: d_model = 128 (Figs 09, 10, 12) ---")
    print(f"Router (CAGC, d_route=16)       : {r128_params:>8} params ({r128_params/tot128*100:6.2f}%)")
    print(f"Branch 1 (Local CNN, c_cnn=143) : {cnn128_params:>8} params ({cnn128_params/tot128*100:6.2f}%)")
    print(f"Branch 2 (Pointwise, c_pw=135)  : {pw128_params:>8} params ({pw128_params/tot128*100:6.2f}%)")
    print(f"Branch 3 (Attention, d_attn=64) : {at128_params:>8} params ({at128_params/tot128*100:6.2f}%)")
    print(f"Total Block Parameters          : {tot128:>8} params")
    print(f"Historical Target Budget        : {target128:>8} params")
    print(f"Parameter Discrepancy (Delta)   : {tot128 - target128:>8} params ({(tot128 - target128)/target128*100:+.4f}%)")
    assert tot128 == target128, f"Mismatch in d_model=128: {tot128} vs {target128}"
    
    # 2. d_model = 64 (Fig 11)
    b64 = AdaptiveTriBranchBlock(d_model=64, router_type='cagc')
    r64_params = count_parameters(b64.router)
    cnn64_params = count_parameters(b64.cnn_branch)
    pw64_params = count_parameters(b64.pw_branch)
    at64_params = count_parameters(b64.attn_branch)
    tot64 = count_parameters(b64)
    target64 = 49984
    
    print("\n--- Configuration B: d_model = 64 (Fig 11) ---")
    print(f"Router (CAGC, d_route=8)        : {r64_params:>8} params ({r64_params/tot64*100:6.2f}%)")
    print(f"Branch 1 (Local CNN, c_cnn=113) : {cnn64_params:>8} params ({cnn64_params/tot64*100:6.2f}%)")
    print(f"Branch 2 (Pointwise, c_pw=113)  : {pw64_params:>8} params ({pw64_params/tot64*100:6.2f}%)")
    print(f"Branch 3 (Attention, d_attn=40) : {at64_params:>8} params ({at64_params/tot64*100:6.2f}%)")
    print(f"Total Block Parameters          : {tot64:>8} params")
    print(f"Historical Target Budget        : {target64:>8} params")
    print(f"Parameter Discrepancy (Delta)   : {tot64 - target64:>8} params ({(tot64 - target64)/target64*100:+.4f}%)")
    assert tot64 == target64, f"Mismatch in d_model=64: {tot64} vs {target64}"

    # 3. LR-CPG Position-Aware Ablation Check
    b128_lrcpg = AdaptiveTriBranchBlock(d_model=128, router_type='lr_cpg')
    tot128_lrcpg = count_parameters(b128_lrcpg)
    pos_overhead = tot128_lrcpg - tot128
    print("\n--- Position-Aware LR-CPG Ablation ---")
    print(f"LR-CPG Block Parameters         : {tot128_lrcpg:>8} params")
    print(f"Position Gate Overhead          : +{pos_overhead} params ({pos_overhead/tot128*100:.2f}% of module budget)")

def verify_full_models():
    print("\n" + "=" * 80)
    print("PHASE 2: FULL-MODEL FACTORY PARAMETER VERIFICATION")
    print("=" * 80)

    # 1. Figure 09: ICVL Hyperspectral Denoising (Rank=20, seq_len=260)
    m09_base = build_deeptensor_model(rank=20, seq_len=260, spatial_type='skip', spectral_type='skip1d')
    m09_hyb = build_deeptensor_model(rank=20, seq_len=260, spatial_type='skip', spectral_type='hybrid')
    m09_adapt = build_deeptensor_model(rank=20, seq_len=260, spatial_type='skip', spectral_type='adaptive_hybrid')
    p09_base = count_parameters(m09_base)
    p09_hyb = count_parameters(m09_hyb)
    p09_adapt = count_parameters(m09_adapt)
    
    print("\n[Figure 09: ICVL HSI (Rank R=20, seq_len=260)]")
    print(f"  Baseline Model Params        : {p09_base:>10}")
    print(f"  Target Hybrid Model Params   : {p09_hyb:>10}")
    print(f"  Adaptive Hybrid Model Params : {p09_adapt:>10}")
    print(f"  Delta vs Target Hybrid       : {p09_adapt - p09_hyb:>10} params ({(p09_adapt - p09_hyb)/p09_hyb*100:+.4f}%)")
    assert p09_adapt == p09_hyb, f"Fig 09 mismatch: {p09_adapt} != {p09_hyb}"
    assert p09_hyb == 3137808, f"Fig 09 hybrid target mismatch: {p09_hyb} != 3137808"

    # 2. Figure 10: Yale Face B (Rank=2000, CP lengths=(192, 168, 160))
    m10_base = build_deeptensor_cp_model(rank=2000, lengths=(192, 168, 160), u_type='skip1d', v_type='skip1d', w_type='skip1d')
    m10_hyb = build_deeptensor_cp_model(rank=2000, lengths=(192, 168, 160), u_type='skip1d', v_type='skip1d', w_type='hybrid')
    m10_adapt = build_deeptensor_cp_model(rank=2000, lengths=(192, 168, 160), u_type='skip1d', v_type='skip1d', w_type='adaptive_hybrid')
    p10_base = count_parameters(m10_base)
    p10_hyb = count_parameters(m10_hyb)
    p10_adapt = count_parameters(m10_adapt)
    
    print("\n[Figure 10: Yale Face B (Rank R=2000, lengths=(192, 168, 160))]")
    print(f"  Baseline Model Params        : {p10_base:>10}")
    print(f"  Target Hybrid Model Params   : {p10_hyb:>10}")
    print(f"  Adaptive Hybrid Model Params : {p10_adapt:>10}")
    print(f"  Delta vs Target Hybrid       : {p10_adapt - p10_hyb:>10} params ({(p10_adapt - p10_hyb)/p10_hyb*100:+.4f}%)")
    assert p10_adapt == p10_hyb, f"Fig 10 mismatch: {p10_adapt} != {p10_hyb}"
    assert p10_hyb == 5602988, f"Fig 10 hybrid target mismatch: {p10_hyb} != 5602988"

    # 3. Figure 11: Video CS (CACTI) (Rank=32, seq_len=64, skip_n33d=64, skip_n33u=64)
    m11_base = build_deeptensor_model(rank=32, seq_len=64, spatial_type='skip', spectral_type='skip1d',
                                      skip_n33d=64, skip_n33u=64, d_model=64)
    m11_hyb = build_deeptensor_model(rank=32, seq_len=64, spatial_type='skip', spectral_type='hybrid',
                                     skip_n33d=64, skip_n33u=64, d_model=64)
    m11_adapt = build_deeptensor_model(rank=32, seq_len=64, spatial_type='skip', spectral_type='adaptive_hybrid',
                                       skip_n33d=64, skip_n33u=64, d_model=64)
    p11_base = count_parameters(m11_base)
    p11_hyb = count_parameters(m11_hyb)
    p11_adapt = count_parameters(m11_adapt)
    
    print("\n[Figure 11: Video CS CACTI (Rank R=32, seq_len=64, d_model=64)]")
    print(f"  Baseline Model Params        : {p11_base:>10}")
    print(f"  Target Hybrid Model Params   : {p11_hyb:>10}")
    print(f"  Adaptive Hybrid Model Params : {p11_adapt:>10}")
    print(f"  Delta vs Target Hybrid       : {p11_adapt - p11_hyb:>10} params ({(p11_adapt - p11_hyb)/p11_hyb*100:+.4f}%)")
    assert p11_adapt == p11_hyb, f"Fig 11 mismatch: {p11_adapt} != {p11_hyb}"
    assert p11_hyb == 834632, f"Fig 11 hybrid target mismatch: {p11_hyb} != 834632"

    # 4. Figure 12: 3D PET/CT Radon CS (Rank=1000, lengths=(128, 128, 56), CP model)
    m12_base = build_deeptensor_cp_model(rank=1000, lengths=(128, 128, 56), u_type='skip1d', v_type='skip1d', w_type='skip1d')
    m12_hyb = build_deeptensor_cp_model(rank=1000, lengths=(128, 128, 56), u_type='skip1d', v_type='skip1d', w_type='hybrid')
    m12_adapt = build_deeptensor_cp_model(rank=1000, lengths=(128, 128, 56), u_type='skip1d', v_type='skip1d', w_type='adaptive_hybrid')
    p12_base = count_parameters(m12_base)
    p12_hyb = count_parameters(m12_hyb)
    p12_adapt = count_parameters(m12_adapt)
    
    print("\n[Figure 12: 3D PET/CT Radon (Rank R=1000, lengths=(128, 128, 56), d_model=128)]")
    print(f"  Baseline Model Params        : {p12_base:>10}")
    print(f"  Target Hybrid Model Params   : {p12_hyb:>10}")
    print(f"  Adaptive Hybrid Model Params : {p12_adapt:>10}")
    print(f"  Delta vs Target Hybrid       : {p12_adapt - p12_hyb:>10} params ({(p12_adapt - p12_hyb)/p12_hyb*100:+.4f}%)")
    assert p12_adapt == p12_hyb, f"Fig 12 mismatch: {p12_adapt} != {p12_hyb}"
    assert p12_hyb == 4050988, f"Fig 12 hybrid target mismatch: {p12_hyb} != 4050988"

if __name__ == "__main__":
    verify_modules()
    verify_full_models()
    print("\n" + "=" * 80)
    print("ALL PARAMETER VERIFICATION CHECKS PASSED: ZERO PARAMETER DELTA")
    print("=" * 80)
