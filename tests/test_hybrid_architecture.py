#!/usr/bin/env python
"""
Unit and Integration Tests for DeepTensor++ Hybrid Architecture
==============================================================
Validates model factory, forward passes, gradient flow, dynamic lengths,
checkpointing, and comparative evaluation infrastructure.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import deeptensor as dt
from deeptensor.models.factory import (
    build_spatial_net,
    build_spectral_net,
    build_deeptensor_model,
    build_deeptensor_cp_model,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
)
from deeptensor.models.attention import (
    HybridSpectralNet,
    SpectralSelfAttention,
    SpectralPointwiseControl,
    SpectralTransformer,
    ControlSpectralTransformer,
)
from deeptensor.utils.comparison import SideBySideComparator, compute_metrics, to_2d_slice


class TestHybridArchitecture(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_imports(self):
        """Verify all new classes and factory functions are accessible from deeptensor."""
        self.assertTrue(hasattr(dt, 'DeepTensor'))
        self.assertTrue(hasattr(dt, 'DeepTensorCP'))
        self.assertTrue(hasattr(dt, 'build_deeptensor_model'))
        self.assertTrue(hasattr(dt, 'build_deeptensor_cp_model'))
        self.assertTrue(hasattr(dt.utils, 'SideBySideComparator'))
        self.assertTrue(hasattr(dt.utils, 'compute_metrics'))

    def test_matrix_factor_forward_passes(self):
        """Verify 2-factor matrix DeepTensor forward pass output shapes for all backbones."""
        rank = 8
        H, W, C = 64, 64, 64
        variants = ['skip1d', 'hybrid', 'transformer', 'control', 'cnn_control', 'pointwise_control']

        im_inp = torch.randn(1, rank, H, W, device=self.device)
        spec_inp = torch.randn(1, rank, C, device=self.device)

        for v in variants:
            with self.subTest(variant=v):
                model = build_deeptensor_model(
                    rank=rank,
                    seq_len=C,
                    spatial_type='skip',
                    spectral_type=v,
                    skip_n33d=32,
                    skip_n33u=32,
                    d_model=32,
                    n_heads=2,
                    n_layers=2,
                    d_ffn=64,
                ).to(self.device)

                recon = model(im_inp, spec_inp)
                # Expected shape: (B, H*W, C) = (1, 4096, 64)
                self.assertEqual(recon.shape, (1, H * W, C))
                self.assertTrue(torch.isfinite(recon).all())

    def test_tensor_cp_forward_passes(self):
        """Verify 3-way PARAFAC DeepTensorCP forward pass output shapes for all backbones."""
        rank = 16
        H, W, T = 64, 64, 64
        w_variants = ['skip1d', 'hybrid', 'transformer', 'control', 'cnn_control', 'pointwise_control']

        u_inp = torch.randn(1, rank, H, device=self.device)
        v_inp = torch.randn(1, rank, W, device=self.device)
        w_inp = torch.randn(1, rank, T, device=self.device)

        for w_v in w_variants:
            with self.subTest(w_variant=w_v):
                model = build_deeptensor_cp_model(
                    rank=rank,
                    lengths=(H, W, T),
                    u_type='skip1d',
                    v_type='skip1d',
                    w_type=w_v,
                    skip_n33d=32,
                    skip_n33u=32,
                    d_model=32,
                    n_heads=2,
                    n_layers=2,
                    d_ffn=64,
                ).to(self.device)

                recon = model(u_inp, v_inp, w_inp)
                # Expected shape: (H, W, T) = (64, 64, 64)
                self.assertEqual(recon.shape, (H, W, T))
                self.assertTrue(torch.isfinite(recon).all())

    def test_gradient_flow_attention_and_transformer(self):
        """Verify backpropagation flows gradients into attention layers, projections, and FFNs."""
        rank = 8
        H, W, C = 64, 64, 64

        # Test Hybrid attention
        model_hybrid = build_deeptensor_model(
            rank=rank, seq_len=C, spatial_type='skip', spectral_type='hybrid',
            skip_n33d=32, skip_n33u=32, d_model=32, n_heads=2,
        ).to(self.device)

        im_inp = torch.randn(1, rank, H, W, device=self.device, requires_grad=True)
        spec_inp = torch.randn(1, rank, C, device=self.device, requires_grad=True)

        out_h = model_hybrid(im_inp, spec_inp)
        loss_h = out_h.sum()
        loss_h.backward()

        # Check gradients in MHA
        attn_block = model_hybrid.spectral_net.attention_block
        self.assertIsNotNone(attn_block.mha.in_proj_weight.grad)
        self.assertTrue((attn_block.mha.in_proj_weight.grad.abs() > 0).any())
        self.assertIsNotNone(attn_block.ffn[0].weight.grad)

        # Test SpectralTransformer
        model_tf = build_deeptensor_model(
            rank=rank, seq_len=C, spatial_type='skip', spectral_type='transformer',
            skip_n33d=32, skip_n33u=32, d_model=32, n_heads=2, n_layers=2, d_ffn=64,
        ).to(self.device)

        im_inp2 = torch.randn(1, rank, H, W, device=self.device, requires_grad=True)
        spec_inp2 = torch.randn(1, rank, C, device=self.device, requires_grad=True)

        out_tf = model_tf(im_inp2, spec_inp2)
        loss_tf = out_tf.sum()
        loss_tf.backward()

        self.assertIsNotNone(model_tf.spectral_net.embedding.weight.grad)
        self.assertTrue((model_tf.spectral_net.embedding.weight.grad.abs() > 0).any())
        self.assertIsNotNone(model_tf.spectral_net.layers[0]['attn'].in_proj_weight.grad)

    def test_dynamic_sequence_length_generalization(self):
        """Verify models can handle variable sequence lengths (L=56, 64, 160, 260) without errors."""
        rank = 8
        lengths = [56, 64, 160, 260]

        for L in lengths:
            with self.subTest(seq_len=L):
                # Hybrid
                spec_hybrid = build_spectral_net(rank, rank, seq_len=260, net_type='hybrid', skip_n33d=128, skip_n33u=128).to(self.device)
                inp = torch.randn(1, rank, L, device=self.device)
                out = spec_hybrid(inp)
                self.assertEqual(out.shape, (1, rank, L))

                # Transformer
                spec_tf = build_spectral_net(rank, rank, seq_len=260, net_type='transformer', d_model=64, n_heads=2, n_layers=2).to(self.device)
                out_tf = spec_tf(inp)
                self.assertEqual(out_tf.shape, (1, rank, L))

    def test_checkpoint_roundtrip(self):
        """Verify checkpoint save and load preserves state dict bit-exactly."""
        model = build_deeptensor_model(
            rank=8, seq_len=64, spatial_type='skip', spectral_type='hybrid',
            skip_n33d=32, skip_n33u=32, d_model=32, n_heads=2,
        ).to(self.device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_ckpt.pth")
            save_checkpoint(model, ckpt_path, epoch=5, metrics={'psnr': 30.5})

            model_loaded = build_deeptensor_model(
                rank=8, seq_len=64, spatial_type='skip', spectral_type='hybrid',
                skip_n33d=32, skip_n33u=32, d_model=32, n_heads=2,
            ).to(self.device)

            meta = load_checkpoint(model_loaded, ckpt_path)
            self.assertEqual(meta['epoch'], 5)
            self.assertEqual(meta['metrics']['psnr'], 30.5)

            # Check parameter equality
            for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_side_by_side_comparator(self):
        """Verify SideBySideComparator metric computation, CSV, JSON, and plotting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            comparator = SideBySideComparator("test_run", tmp_dir)
            gt = np.random.rand(32, 32, 10).astype(np.float32)
            recon_baseline = gt + np.random.normal(0, 0.05, gt.shape).astype(np.float32)
            recon_hybrid = gt + np.random.normal(0, 0.03, gt.shape).astype(np.float32)

            comparator.add_model_result("baseline", recon_baseline, params=1000, runtime_sec=1.5, peak_vram_mb=50.0)
            comparator.add_model_result("hybrid", recon_hybrid, params=1005, runtime_sec=1.7, peak_vram_mb=52.0)

            summary = comparator.generate_comparison(gt=gt, save_figures=True)
            self.assertIn('models', summary)
            self.assertIn('baseline', summary['models'])
            self.assertIn('hybrid', summary['models'])
            self.assertTrue(os.path.exists(summary['json_path']))
            self.assertTrue(os.path.exists(summary['csv_path']))
            self.assertTrue(os.path.exists(summary['figure_path']))


if __name__ == '__main__':
    unittest.main()
