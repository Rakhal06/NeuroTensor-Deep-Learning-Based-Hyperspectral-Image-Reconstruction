#!/usr/bin/env python
'''
Table 3: DeepTensor NMF & Semi-NMF on Speech Spectrograms
Paper: "DeepTensor: Low-Rank Tensor Decomposition with Deep Network Priors" (Section 4.3, Page 10)

Evaluates DeepTensor Nonnegative Matrix Factorization (NMF) and Semi-NMF on Speech Spectrograms
(Google Speech Commands STFT, 512 x 512) under Rician noise (input SNR ~ 4.6 dB) across:
- NMF Mode: Softplus, Abs. value, ReLU
- Semi-NMF Mode: Softplus, Abs. value, ReLU
Against:
- Baseline NMF (Lee & Seung 2001)
- Baseline Semi-NMF (Ding et al. 2010)
'''

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torchaudio
from sklearn.decomposition import NMF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure root directory and modules are on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import models, utils, losses, deep_prior, deep_decoder

def get_speech_spectrograms(data_root='./data', n_utterances=40):
    os.makedirs(data_root, exist_ok=True)
    import glob
    import scipy.io.wavfile as wf
    import warnings
    warnings.filterwarnings('ignore', category=wf.WavFileWarning)

    speech_dir = os.path.join(data_root, 'SpeechCommands', 'speech_commands_v0.02')
    wav_files = sorted(glob.glob(os.path.join(speech_dir, '*', '*.wav')))
    if len(wav_files) >= n_utterances:
        print(f"Loading {n_utterances} utterances from Google Speech Commands dataset...", flush=True)
        waveforms = []
        for f in wav_files[:n_utterances]:
            rate, d = wf.read(f)
            d_tensor = torch.tensor(d, dtype=torch.float32) / 32768.0
            if len(d_tensor) < 16000:
                d_tensor = torch.nn.functional.pad(d_tensor, (0, 16000 - len(d_tensor)))
            else:
                d_tensor = d_tensor[:16000]
            waveforms.append(d_tensor)
        waveforms = torch.stack(waveforms)
        spectro = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            win_length=512,
            hop_length=32,
            center=True,
            pad_mode="reflect",
            power=2.0
        )
        specs = spectro(waveforms)[:, None, :500, :500].numpy() # (40, 1, 500, 500)
    else:
        print("SpeechCommands audio files not found on disk, synthesizing matched harmonic spectrograms...", flush=True)
        np.random.seed(42)
        specs = np.zeros((n_utterances, 1, 500, 500), dtype=np.float32)
        t = np.linspace(0, 1, 500)
        for i in range(n_utterances):
            harmonic_spec = np.zeros((500, 500), dtype=np.float32)
            n_harmonics = np.random.randint(5, 15)
            for h in range(1, n_harmonics + 1):
                f0 = 20 * h + np.random.randn() * 2
                env = np.exp(-((t - 0.5) ** 2) / 0.05)
                harmonic_spec[int(f0) % 500, :] += env
            specs[i, 0] = harmonic_spec

    # Normalize per utterance to [0, 1] exactly as in run_table1.py line 421-422
    specs = specs.astype(np.float32)
    specs -= specs.min((1, 2, 3), keepdims=True)
    specs /= (specs.max((1, 2, 3), keepdims=True) + 1e-8)

    # Add Rician noise: n = 0.3 * z1 + 0.3 * z2^2 exactly as in run_table1.py line 424
    np.random.seed(42)
    rician_noise = (np.random.randn(*specs.shape).astype(np.float32) * 0.3 + 
                    (np.random.randn(*specs.shape).astype(np.float32) ** 2) * 0.3)
    noisy_specs = np.clip(specs + rician_noise, 0.0, 10.0)

    input_psnr = utils.psnr(specs, noisy_specs)
    print(f"Speech Spectrograms: shape={specs.shape} | Input Noisy PSNR: {input_psnr:.2f} dB (Paper: 4.6 dB)", flush=True)
    return specs, noisy_specs, input_psnr

def run_standard_nmf(data, noisy_data, n_components=10):
    N, C, H, W = data.shape
    X_noisy = noisy_data.reshape(N, H * W)

    # Standard Lee-Seung NMF
    nmf = NMF(n_components=n_components, init='nndsvda', max_iter=500, random_state=42)
    W_mat = nmf.fit_transform(X_noisy)
    H_mat = nmf.components_
    recon_nmf = np.dot(W_mat, H_mat).reshape(data.shape)
    psnr_nmf = utils.psnr(data, recon_nmf)

    # Semi-NMF approximation: unconstrained H, nonnegative W
    W_semi = np.maximum(W_mat, 0.0)
    H_semi = np.linalg.pinv(W_semi) @ X_noisy
    recon_snmf = np.dot(W_semi, H_semi).reshape(data.shape)
    psnr_snmf = utils.psnr(data, recon_snmf)

    return float(psnr_nmf), float(psnr_snmf), recon_nmf

def run_deeptensor_speech(data, noisy_data, act_u, act_v, mode_name, args):
    N, C, H, W = data.shape
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    n_clusters = args.n_clusters
    epochs = args.epochs
    lr = args.learning_rate

    data_ten = torch.tensor(data, device=device)
    noisy_ten = torch.tensor(noisy_data, device=device)

    if getattr(args, 'nettype', 'DD') == 'DD':
        u_inp = utils.get_inp([1, n_clusters, N // 4], device=device)
        v_inp = utils.get_inp([1, n_clusters, H // 4, W // 4], device=device)
        v_net = deep_decoder.decodernw(
            num_output_channels=n_clusters,
            num_channels_up=[n_clusters, 128]
        ).to(device)
        u_net = deep_decoder.decodernw1d(
            num_output_channels=n_clusters,
            num_channels_up=[n_clusters, 128]
        ).to(device)
    else:
        u_net = deep_prior.get_net(
            n_clusters, "skip1d", "reflection",
            upsample_mode="linear",
            skip_n33d=args.init_nconv,
            skip_n33u=args.init_nconv,
            num_scales=5,
            n_channels=n_clusters
        ).to(device)

        v_net = deep_prior.get_net(
            n_clusters, "skip", "reflection",
            upsample_mode="bilinear",
            skip_n33d=args.init_nconv,
            skip_n33u=args.init_nconv,
            num_scales=4,
            n_channels=n_clusters
        ).to(device)

        u_inp = utils.get_inp([1, n_clusters, N], device=device)
        v_inp = utils.get_inp([1, n_clusters, H, W], device=device)

    u_net.train()
    v_net.train()

    optimizer = torch.optim.Adam(list(u_net.parameters()) + list(v_net.parameters()) + [u_inp, v_inp], lr=lr)
    criterion = losses.L2Norm()

    best_loss = float('inf')
    best_recon = None
    loss_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        u_out = act_u(u_net(u_inp))       # (1, n_clusters, N)
        v_out = act_v(v_net(v_inp))       # (1, n_clusters, H, W)

        v_mat = v_out.reshape(1, n_clusters, H * W)
        u_mat = u_out.permute(0, 2, 1)    # (1, N, n_clusters)

        recon_flat = torch.bmm(u_mat, v_mat)
        recon = recon_flat.reshape(1, N, H, W).permute(1, 0, 2, 3)

        loss_data = criterion(recon - noisy_ten)
        loss_reg = 0.01 * (v_out.abs().mean() + u_out.abs().mean())
        loss = loss_data + loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_recon = recon.detach().clone()

        if epoch % args.log_interval == 0 or epoch == epochs:
            cur_psnr = utils.psnr(data, recon.detach().cpu().numpy())
            print(f"[{mode_name}] Epoch [{epoch:4d}/{epochs:4d}] | Loss: {loss_val:.4e} | PSNR: {cur_psnr:.2f} dB", flush=True)

    elapsed_time = time.time() - start_time
    recon_final = best_recon.cpu().numpy()
    final_psnr = float(utils.psnr(data, recon_final))
    rmse = float(np.sqrt(np.mean((data - recon_final) ** 2)))

    return {
        'psnr': final_psnr,
        'rmse': rmse,
        'runtime': elapsed_time,
        'loss_history': loss_history,
        'recon': recon_final
    }

def main():
    parser = argparse.ArgumentParser(description="Table 3: Speech Spectrogram NMF Reproduction")
    parser.add_argument('--n_utterances', type=int, default=40)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--nettype', type=str, default='DD', choices=['DD', 'DIP'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--init_nconv', type=int, default=128)
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--log_interval', type=int, default=250)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'results', 'baseline', 'table03_speech'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, 'images')
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("======================================================================")
    print("TABLE 3: SPEECH COMMANDS SPECTROGRAM NMF & SEMI-NMF BENCHMARK")
    print("======================================================================")

    data, noisy_data, in_psnr = get_speech_spectrograms(n_utterances=args.n_utterances)

    # Baselines: Standard NMF and Semi-NMF
    nmf_psnr, snmf_psnr, nmf_recon = run_standard_nmf(data, noisy_data, n_components=args.n_clusters)
    print(f"Standard NMF PSNR: {nmf_psnr:.2f} dB | Baseline Semi-NMF PSNR: {snmf_psnr:.2f} dB")

    results_table = {
        'input_psnr': float(in_psnr),
        'baseline_nmf': float(nmf_psnr),
        'baseline_snmf': float(snmf_psnr),
        'nmf': {},
        'semi_nmf': {}
    }

    # NMF Modes: Nonnegativity on U and V
    nmf_activations = {
        'softplus': (torch.nn.Softplus(), torch.nn.Softplus()),
        'abs': (torch.abs, torch.abs),
        'relu': (torch.relu, torch.relu)
    }

    print("\n--- RUNNING DEEPTENSOR NMF MODES ---", flush=True)
    last_recon = None
    for act_name, (act_u, act_v) in nmf_activations.items():
        run_psnrs = []
        for run_idx in range(args.n_runs):
            print(f"Running DT NMF [{act_name.upper()}] (Run {run_idx+1}/{args.n_runs})...", flush=True)
            res = run_deeptensor_speech(data, noisy_data, act_u, act_v, f"NMF_{act_name}_r{run_idx+1}", args)
            run_psnrs.append(res['psnr'])
            last_recon = res['recon']
        mean_p = float(np.mean(run_psnrs))
        std_p = float(np.std(run_psnrs))
        results_table['nmf'][act_name] = {'mean_psnr': mean_p, 'std_psnr': std_p, 'runs': run_psnrs}
        print(f"DT NMF {act_name.upper()}: {mean_p:.2f} ± {std_p:.2f} dB", flush=True)
        # Incremental save
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fp:
            json.dump(results_table, fp, indent=2)

    # Semi-NMF Modes: Nonnegativity on U only (identity on V)
    semi_activations = {
        'softplus': (torch.nn.Softplus(), lambda x: x),
        'abs': (torch.abs, lambda x: x),
        'relu': (torch.relu, lambda x: x)
    }

    print("\n--- RUNNING DEEPTENSOR SEMI-NMF MODES ---", flush=True)
    for act_name, (act_u, act_v) in semi_activations.items():
        run_psnrs = []
        for run_idx in range(args.n_runs):
            print(f"Running DT Semi-NMF [{act_name.upper()}] (Run {run_idx+1}/{args.n_runs})...", flush=True)
            res = run_deeptensor_speech(data, noisy_data, act_u, act_v, f"SemiNMF_{act_name}_r{run_idx+1}", args)
            run_psnrs.append(res['psnr'])
            last_recon = res['recon']
        mean_p = float(np.mean(run_psnrs))
        std_p = float(np.std(run_psnrs))
        results_table['semi_nmf'][act_name] = {'mean_psnr': mean_p, 'std_psnr': std_p, 'runs': run_psnrs}
        print(f"DT Semi-NMF {act_name.upper()}: {mean_p:.2f} ± {std_p:.2f} dB", flush=True)
        # Incremental save
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fp:
            json.dump(results_table, fp, indent=2)

    # Save visualization of sample spectrogram
    if last_recon is not None:
        idx = 0
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(data[idx, 0], aspect='auto', origin='lower', cmap='magma')
        plt.title('Clean STFT Spectrogram')
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.imshow(noisy_data[idx, 0], aspect='auto', origin='lower', cmap='magma')
        plt.title(f'Rician Noisy ({in_psnr:.1f} dB)')
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.imshow(last_recon[idx, 0], aspect='auto', origin='lower', cmap='magma')
        plt.title(f'DeepTensor Recon ({results_table["semi_nmf"]["relu"]["mean_psnr"]:.1f} dB)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'images', 'figure_speech_nmf_panel.png'), dpi=200)
        plt.close()

    # Save Results JSON and TXT
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fp:
        json.dump(results_table, fp, indent=2)

    summary_text = f"""======================================================================
TABLE 3: SPEECH NMF & SEMI-NMF REPRODUCTION RESULTS (Mean ± Std over {args.n_runs} runs)
======================================================================
MODE        ACTIVATION FUNCTION    REPRODUCED (OURS)       PAPER REPORTED
----------------------------------------------------------------------
NMF         Softplus               {results_table['nmf']['softplus']['mean_psnr']:.2f} ± {results_table['nmf']['softplus']['std_psnr']:.2f} dB       9.00 ± 0.03 dB
NMF         Abs. value             {results_table['nmf']['abs']['mean_psnr']:.2f} ± {results_table['nmf']['abs']['std_psnr']:.2f} dB       8.70 ± 0.01 dB
NMF         ReLU                   {results_table['nmf']['relu']['mean_psnr']:.2f} ± {results_table['nmf']['relu']['std_psnr']:.2f} dB       8.70 ± 0.03 dB
NMF         Baseline NMF           {results_table['baseline_nmf']:.2f} dB                 4.60 dB
----------------------------------------------------------------------
Semi-NMF    Softplus               {results_table['semi_nmf']['softplus']['mean_psnr']:.2f} ± {results_table['semi_nmf']['softplus']['std_psnr']:.2f} dB       8.70 ± 0.09 dB
Semi-NMF    Abs. value             {results_table['semi_nmf']['abs']['mean_psnr']:.2f} ± {results_table['semi_nmf']['abs']['std_psnr']:.2f} dB       8.80 ± 0.02 dB
Semi-NMF    ReLU                   {results_table['semi_nmf']['relu']['mean_psnr']:.2f} ± {results_table['semi_nmf']['relu']['std_psnr']:.2f} dB       8.80 ± 0.02 dB
Semi-NMF    Baseline Semi-NMF      {results_table['baseline_snmf']:.2f} dB                 4.70 dB
======================================================================
"""
    print(summary_text)
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as fp:
        fp.write(summary_text)

if __name__ == '__main__':
    main()

