"""
DeepTensor++ Adaptive Hybrid Architecture Unit Tests
===================================================
Comprehensive test suite validating Phase 1 implementation and Phase 2 parameter verification:
  1. Forward-pass shapes for all four application dimensions.
  2. Backpropagation through all three branches.
  3. Nonzero branch gradient checks.
  4. Router softmax sums to one.
  5. Gate values remain finite and within [0, 1].
  6. Temperature changes behave correctly.
  7. Batch size and sequence-length variations.
  8. CPU and CUDA compatibility when available.
  9. Mixed precision compatibility where supported.
  10. NaN and Inf detection.
  11. Parameter counts (exact matching).
  12. Residual and output-shape correctness.
  13. Position-aware gate behavior (LR-CPG ablation).
  14. Save/load state_dict compatibility.
"""

import os
import sys
import math
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor.models.factory import (
    build_deeptensor_model,
    build_deeptensor_cp_model,
    build_spectral_net,
    count_parameters,
)
from deeptensor.models.attention import (
    AdaptiveTriBranchBlock,
    AdaptiveHybridSpectralNet,
    ChannelAwareGlobalContextGate,
    LowRankChannelPositionGate,
    LocalCNNBranch,
    PointwiseBranch,
    GlobalAttentionBranch,
    get_cosine_temperature,
)


class TestAdaptiveHybrid(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)

    def test_01_forward_shapes_all_four_application_dimensions(self):
        """1. Forward-pass shapes for all four application dimensions."""
        # Figure 09: ICVL HSI (128 channels, seq_len=260)
        b09 = AdaptiveTriBranchBlock(d_model=128, seq_len=260).to(self.device)
        x09 = torch.randn(1, 128, 260, device=self.device)
        y09 = b09(x09)
        self.assertEqual(y09.shape, (1, 128, 260))

        # Figure 10: Yale Face B (128 channels, seq_len=160)
        b10 = AdaptiveTriBranchBlock(d_model=128, seq_len=160).to(self.device)
        x10 = torch.randn(1, 128, 160, device=self.device)
        y10 = b10(x10)
        self.assertEqual(y10.shape, (1, 128, 160))

        # Figure 11: Video CS CACTI (64 channels, seq_len=64)
        b11 = AdaptiveTriBranchBlock(d_model=64, seq_len=64).to(self.device)
        x11 = torch.randn(1, 64, 64, device=self.device)
        y11 = b11(x11)
        self.assertEqual(y11.shape, (1, 64, 64))

        # Figure 12: 3D PET/CT Radon (128 channels, seq_len=56)
        b12 = AdaptiveTriBranchBlock(d_model=128, seq_len=56).to(self.device)
        x12 = torch.randn(1, 128, 56, device=self.device)
        y12 = b12(x12)
        self.assertEqual(y12.shape, (1, 128, 56))

        # Full DeepTensor (2-factor) for Fig 09
        m09 = build_deeptensor_model(rank=20, seq_len=260, spectral_type='adaptive_hybrid').to(self.device)
        im09 = torch.randn(1, 20, 64, 64, device=self.device)
        sp09 = torch.randn(1, 20, 260, device=self.device)
        out09 = m09(im09, sp09)
        self.assertEqual(out09.shape, (1, 64 * 64, 260))

        # Full DeepTensorCP (3-way) for Fig 10
        m10 = build_deeptensor_cp_model(
            rank=2000, lengths=(192, 168, 160), u_type='skip1d', v_type='skip1d', w_type='adaptive_hybrid'
        ).to(self.device)
        u10 = torch.randn(1, 2000, 192, device=self.device)
        v10 = torch.randn(1, 2000, 168, device=self.device)
        w10 = torch.randn(1, 2000, 160, device=self.device)
        out10 = m10(u10, v10, w10)
        self.assertEqual(out10.shape, (192, 168, 160))

    def test_02_backpropagation_all_three_branches(self):
        """2. Backpropagation through all three branches and input tensor."""
        for d_model, seq_len in [(128, 260), (64, 64)]:
            block = AdaptiveTriBranchBlock(d_model=d_model, seq_len=seq_len).to(self.device)
            x = torch.randn(1, d_model, seq_len, device=self.device, requires_grad=True)
            y = block(x)
            loss = y.sum()
            loss.backward()

            self.assertIsNotNone(x.grad)
            self.assertEqual(x.grad.shape, x.shape)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(torch.count_nonzero(x.grad).item(), 0)

    def test_03_nonzero_branch_gradients(self):
        """3. Verify all parameters in CNN, PW, Attn, and Router receive nonzero gradients."""
        for d_model in [128, 64]:
            block = AdaptiveTriBranchBlock(d_model=d_model, seq_len=50).to(self.device)
            x = torch.randn(2, d_model, 50, device=self.device, requires_grad=True)
            y = block(x)
            loss = (y ** 2).sum()
            loss.backward()

            zero_grad_params = []
            for name, param in block.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"Param {name} has None grad!")
                    self.assertTrue(torch.isfinite(param.grad).all(), f"Param {name} has NaN/Inf grad!")
                    nonzero_count = torch.count_nonzero(param.grad).item()
                    if nonzero_count == 0:
                        zero_grad_params.append(name)

            self.assertEqual(
                zero_grad_params,
                [],
                f"Parameters with zero gradient in d_model={d_model}: {zero_grad_params}"
            )

    def test_04_router_softmax_sums_to_one(self):
        """4. Verify router softmax weights sum to 1.0 across branches."""
        # Test CAGC router
        cagc = ChannelAwareGlobalContextGate(d_model=128, d_route=16).to(self.device)
        x = torch.randn(3, 128, 75, device=self.device)
        gates = cagc(x, tau=1.0)  # [3, 128, 3]
        sums = gates.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

        # Test LR-CPG router
        lrcpg = LowRankChannelPositionGate(d_model=128, d_route=16).to(self.device)
        gates_lr = lrcpg(x, tau=1.0)  # [3, 128, 75, 3]
        sums_lr = gates_lr.sum(dim=-1)
        self.assertTrue(torch.allclose(sums_lr, torch.ones_like(sums_lr), atol=1e-5))

    def test_05_gate_values_finite_and_in_range(self):
        """5. Gate values remain finite and strictly within [0, 1]."""
        block = AdaptiveTriBranchBlock(d_model=128, router_type='cagc').to(self.device)
        x = torch.randn(2, 128, 100, device=self.device)
        _ = block(x)
        gates = block.last_gates
        self.assertIsNotNone(gates)
        self.assertTrue(torch.isfinite(gates).all())
        self.assertTrue((gates >= 0.0).all())
        self.assertTrue((gates <= 1.0).all())

    def test_06_temperature_changes_behavior(self):
        """6. High temperature approaches uniform (1/3); low temperature sharpens."""
        block = AdaptiveTriBranchBlock(d_model=128, router_type='cagc').to(self.device)
        x = torch.randn(1, 128, 80, device=self.device)

        # High temperature -> uniform (1/3, 1/3, 1/3)
        _ = block(x, tau=100.0)
        gates_high = block.last_gates
        mean_high = gates_high.mean(dim=(0, 1))
        self.assertTrue(torch.allclose(mean_high, torch.full_like(mean_high, 1.0 / 3.0), atol=1e-2))
        ent_high = block.compute_router_entropy().item()
        self.assertAlmostEqual(ent_high, math.log(3.0), places=2)

        # Low temperature -> sharpened logits, lower entropy
        _ = block(x, tau=0.1)
        gates_low = block.last_gates
        ent_low = block.compute_router_entropy().item()
        self.assertLess(ent_low, ent_high)

        # Test cosine temperature schedule function
        t_0 = get_cosine_temperature(0, 100, tau_start=2.0, tau_min=0.5)
        t_50 = get_cosine_temperature(50, 100, tau_start=2.0, tau_min=0.5)
        t_100 = get_cosine_temperature(100, 100, tau_start=2.0, tau_min=0.5)
        self.assertAlmostEqual(t_0, 2.0, places=5)
        self.assertAlmostEqual(t_50, 1.25, places=5)
        self.assertAlmostEqual(t_100, 0.5, places=5)

    def test_07_batch_and_sequence_length_variations(self):
        """7. Verify robustness across varying batch sizes and sequence lengths."""
        block = AdaptiveTriBranchBlock(d_model=128, seq_len=100).to(self.device)
        for B in [1, 2, 4]:
            for L in [16, 64, 100, 256, 512]:
                x = torch.randn(B, 128, L, device=self.device)
                y = block(x)
                self.assertEqual(y.shape, (B, 128, L))
                self.assertTrue(torch.isfinite(y).all())

    def test_08_cpu_and_cuda_compatibility(self):
        """8. Module executes correctly on CPU and CUDA (if available)."""
        # Test on CPU
        block_cpu = AdaptiveTriBranchBlock(d_model=128).to('cpu')
        x_cpu = torch.randn(1, 128, 50, device='cpu')
        y_cpu = block_cpu(x_cpu)
        self.assertEqual(y_cpu.device.type, 'cpu')
        self.assertEqual(y_cpu.shape, (1, 128, 50))

        # Test on CUDA if available
        if torch.cuda.is_available():
            block_cuda = AdaptiveTriBranchBlock(d_model=128).to('cuda')
            x_cuda = torch.randn(1, 128, 50, device='cuda')
            y_cuda = block_cuda(x_cuda)
            self.assertEqual(y_cuda.device.type, 'cuda')
            self.assertEqual(y_cuda.shape, (1, 128, 50))

    def test_09_mixed_precision_compatibility(self):
        """9. Mixed precision (autocast) forward and backward stability."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for AMP testing")

        block = AdaptiveTriBranchBlock(d_model=128).to('cuda')
        x = torch.randn(1, 128, 64, device='cuda', requires_grad=True)
        optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            y = block(x)
            loss = y.mean()

        scaler = torch.amp.GradScaler('cuda')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        self.assertTrue(torch.isfinite(y).all())
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_10_nan_and_inf_detection(self):
        """10. Check that normal and extreme inputs produce no NaN or Inf."""
        block = AdaptiveTriBranchBlock(d_model=128).to(self.device)

        # Standard random input
        x_norm = torch.randn(2, 128, 64, device=self.device)
        y_norm = block(x_norm)
        self.assertFalse(torch.isnan(y_norm).any())
        self.assertFalse(torch.isinf(y_norm).any())

        # Large magnitude input (+- 50.0)
        x_large = torch.randn(2, 128, 64, device=self.device) * 50.0
        y_large = block(x_large)
        self.assertFalse(torch.isnan(y_large).any())
        self.assertFalse(torch.isinf(y_large).any())

    def test_11_exact_parameter_counts(self):
        """11. Parameter counts match theoretical derivations down to 0 parameter delta."""
        # d_model = 128
        b128 = AdaptiveTriBranchBlock(d_model=128, router_type='cagc')
        r128 = count_parameters(b128.router)
        cnn128 = count_parameters(b128.cnn_branch)
        pw128 = count_parameters(b128.pw_branch)
        at128 = count_parameters(b128.attn_branch)
        tot128 = count_parameters(b128)

        self.assertEqual(r128, 8592, f"Router params {r128} != 8592")
        self.assertEqual(cnn128, 38505, f"CNN params {cnn128} != 38505")
        self.assertEqual(pw128, 35079, f"PW params {pw128} != 35079")
        self.assertEqual(at128, 50304, f"Attn params {at128} != 50304")
        self.assertEqual(tot128, 132480, f"Total 128 params {tot128} != 132480 (Delta: {tot128 - 132480})")

        # d_model = 64
        b64 = AdaptiveTriBranchBlock(d_model=64, router_type='cagc')
        r64 = count_parameters(b64.router)
        cnn64 = count_parameters(b64.cnn_branch)
        pw64 = count_parameters(b64.pw_branch)
        at64 = count_parameters(b64.attn_branch)
        tot64 = count_parameters(b64)

        self.assertEqual(r64, 2248, f"Router params {r64} != 2248")
        self.assertEqual(cnn64, 15703, f"CNN params {cnn64} != 15703")
        self.assertEqual(pw64, 14769, f"PW params {pw64} != 14769")
        self.assertEqual(at64, 17264, f"Attn params {at64} != 17264")
        self.assertEqual(tot64, 49984, f"Total 64 params {tot64} != 49984 (Delta: {tot64 - 49984})")

    def test_12_residual_and_output_correctness(self):
        """12. Residual connection preserves identity when branches output zero."""
        block = AdaptiveTriBranchBlock(d_model=128, router_type='static').to(self.device)
        x = torch.randn(1, 128, 50, device=self.device)

        # Zero out branch weights
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()

        # If branch weights and biases are 0, output is exactly x
        y = block(x)
        self.assertTrue(torch.allclose(y, x, atol=1e-5))

    def test_13_position_aware_gate_behavior(self):
        """13. Position-aware gate (LR-CPG) produces position-dependent routing."""
        block = AdaptiveTriBranchBlock(d_model=128, router_type='lr_cpg').to(self.device)
        x = torch.randn(1, 128, 64, device=self.device)
        y = block(x)
        self.assertEqual(y.shape, (1, 128, 64))

        gates = block.last_gates
        self.assertEqual(gates.shape, (1, 128, 64, 3))
        # Ensure position-dependent variance across sequence dimension
        pos_std = gates.std(dim=2).mean().item()
        self.assertGreater(pos_std, 1e-6, "LR-CPG should produce spatially varying gates")

    def test_14_save_load_state_dict_compatibility(self):
        """14. Save/load state_dict produces identical outputs."""
        block1 = AdaptiveTriBranchBlock(d_model=128, router_type='cagc').to(self.device)
        x = torch.randn(1, 128, 60, device=self.device)
        y1 = block1(x)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name

        try:
            torch.save(block1.state_dict(), temp_path)
            block2 = AdaptiveTriBranchBlock(d_model=128, router_type='cagc').to(self.device)
            block2.load_state_dict(torch.load(temp_path, weights_only=True))
            y2 = block2(x)
            self.assertTrue(torch.allclose(y1, y2, atol=1e-6))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_15_clamped_routing_and_gating_floor(self):
        """15. Clamped routing modes, gating floor, and channel entropy."""
        x = torch.randn(1, 128, 60, device=self.device)

        # CNN-only
        b_cnn = AdaptiveTriBranchBlock(d_model=128, router_type='cnn_only').to(self.device)
        y_cnn = b_cnn(x)
        self.assertEqual(y_cnn.shape, x.shape)
        self.assertTrue(torch.allclose(b_cnn.last_gates, torch.tensor([1.0, 0.0, 0.0], device=self.device)))

        # Pointwise-only
        b_pw = AdaptiveTriBranchBlock(d_model=128, router_type='pointwise_only').to(self.device)
        y_pw = b_pw(x)
        self.assertEqual(y_pw.shape, x.shape)
        self.assertTrue(torch.allclose(b_pw.last_gates, torch.tensor([0.0, 1.0, 0.0], device=self.device)))

        # Attention-only
        b_at = AdaptiveTriBranchBlock(d_model=128, router_type='attention_only').to(self.device)
        y_at = b_at(x)
        self.assertEqual(y_at.shape, x.shape)
        self.assertTrue(torch.allclose(b_at.last_gates, torch.tensor([0.0, 0.0, 1.0], device=self.device)))

        # Gating Floor
        b_fl = AdaptiveTriBranchBlock(d_model=128, router_type='cagc', gating_floor=0.15).to(self.device)
        y_fl = b_fl(x)
        self.assertGreaterEqual(b_fl.last_gates.min().item(), 0.1499)
        self.assertTrue(torch.allclose(b_fl.last_gates.sum(dim=-1), torch.ones(1, 128, device=self.device), atol=1e-5))

        # Channel entropy
        h = b_fl.compute_channel_entropy()
        self.assertGreater(h.item(), 0.0)
        self.assertLessEqual(h.item(), math.log(3.0) + 1e-4)


if __name__ == '__main__':
    unittest.main()
