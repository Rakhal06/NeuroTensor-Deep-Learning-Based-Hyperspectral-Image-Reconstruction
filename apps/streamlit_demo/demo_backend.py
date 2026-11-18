import os
import sys
import copy
import time
import json
import torch
import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add legacy modules and deeptensor package to path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(cur_dir, '..', '..'))
legacy_dir = os.path.join(root_dir, 'archive', 'legacy', 'modules_legacy')
for p in [root_dir, legacy_dir, os.path.join(root_dir, 'src')]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import deeptensor.models as models
    import deeptensor.utils as utils
    import deeptensor.losses as losses
    import deeptensor.models.spectral as spectral
    import deeptensor.models.deep_prior as deep_prior
except ImportError:
    import models
    import utils
    import losses
    import spectral
    import deep_prior

class DemoBackend:
    def __init__(self, expname='icvl', H=None, W=None, nwvl=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.expname = expname
        self.scaling = 1
        self.tau = 100
        self.noise_snr = 2
        
        self.load_data()
        
    def load_data(self):
        data_file = os.path.join(root_dir, 'data', f'{self.expname}.mat')
        if not os.path.exists(data_file):
            data_file = f'data/{self.expname}.mat'
        data = io.loadmat(data_file)
        self.cube = data['hypercube'].astype(np.float32)
        self.wavelengths = data['wavelengths'].astype(np.float32).ravel()
        self.cube = utils.resize(self.cube / self.cube.max(), self.scaling)
        
        self.H, self.W, self.nwvl = self.cube.shape
        self.hsmat = self.cube.reshape(self.H * self.W, self.nwvl)
        
        self.cube_noisy = utils.measure(self.cube, self.noise_snr, self.tau)
        self.hsten = torch.tensor(self.hsmat)[None, ...].to(self.device)
        self.hsten_noisy = torch.tensor(self.cube_noisy.reshape(self.H * self.W, self.nwvl))[None, ...].to(self.device)
        
        self.noisy_psnr = utils.psnr(self.cube, self.cube_noisy)

    def print_preprocessing(self):
        print(f"Dataset: {self.expname}")
        print(f"Original shape: ({self.H}, {self.W}, {self.nwvl})")
        print(f"dtype: {self.cube.dtype}")
        print(f"scaling: {self.scaling}")
        print(f"noise SNR: {self.noise_snr}")
        print(f"Device: {self.device}")
        print(f"Unfolded shape: ({self.H * self.W}, {self.nwvl})")

    def _init_networks(self, n_inputs, n_channels):
        im_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                    upsample_mode='bilinear',
                                    skip_n33d=128, skip_n33u=128,
                                    num_scales=5, n_channels=n_channels).to(self.device)
        spec_net = deep_prior.get_net(n_inputs, 'skip1d', 'reflection',
                                      upsample_mode='linear',
                                      skip_n33d=128, skip_n33u=128,
                                      num_scales=5, n_channels=n_channels).to(self.device)
        im_inp = utils.get_inp([1, n_inputs, self.H, self.W]).to(self.device)
        spec_inp = utils.get_inp([1, n_inputs, self.nwvl]).to(self.device)
        return im_net, spec_net, im_inp, spec_inp

    def run_baseline(self, mode='demo', epochs=5000):
        rank = 20
        n_inputs = rank
        im_net, spec_net, im_inp, spec_inp = self._init_networks(n_inputs, rank)
        
        chkpt_dir = os.path.join(root_dir, 'results', 'baseline', 'checkpoint')
        os.makedirs(chkpt_dir, exist_ok=True)
        chkpt_path = os.path.join(chkpt_dir, 'checkpoint.pt')
        
        if mode == 'demo':
            if not os.path.exists(chkpt_path):
                raise FileNotFoundError(f"[NOT AVAILABLE] Checkpoint not found at {chkpt_path}. Please run --mode full first.")
            ckpt = torch.load(chkpt_path, map_location=self.device)
            if ckpt.get('checkpoint_status') != 'FINAL':
                print("\n[WARNING]\nThis checkpoint is TEST/VALIDATION and cannot be used\nfor the official presentation.\n")
                raise RuntimeError("Checkpoint is not FINAL.")
            im_net.load_state_dict(ckpt['im_net'])
            spec_net.load_state_dict(ckpt['spec_net'])
            im_inp = ckpt['im_inp'].to(self.device)
            spec_inp = ckpt['spec_inp'].to(self.device)
            
            im_net.eval()
            spec_net.eval()
            with torch.no_grad():
                U_img = im_net(im_inp)
                V = spec_net(spec_inp)
                U = U_img.reshape(-1, rank, self.H * self.W).permute(0, 2, 1)
                
            if 'best_hs_estim' in ckpt:
                hs_estim = ckpt['best_hs_estim']
            else:
                hs_estim = torch.bmm(U, V)
            
            cube_estim = hs_estim.squeeze().cpu().numpy().reshape(self.H, self.W, self.nwvl)
            
            return {
                'method': 'Baseline',
                'U': U, 'V': V, 'U_img': U_img,
                'reconstruction': cube_estim,
                'hs_estim': hs_estim,
                'rank': rank,
                'loss': ckpt.get('final_loss', 0.0),
                'metrics': self._compute_metrics(cube_estim),
                'optimizer_info': ckpt.get('optimizer_info', 'Adam, lr=1e-3')
            }
        
        elif mode == 'full':
            import copy
            # Run the actual baseline training
            criterion_l1 = losses.L1Norm()
            im_inp_per = im_inp.detach().clone()
            spec_inp_per = spec_inp.detach().clone()
            
            params = list(im_net.parameters()) + list(spec_net.parameters()) + [im_inp, spec_inp]
            optimizer = torch.optim.Adam(lr=1e-3, params=params)
            
            best_loss = float('inf')
            best_hs_estim = None
            best_im_net_state = None
            best_spec_net_state = None
            best_im_inp = None
            best_spec_inp = None
            
            reg_noise_std = 1.0/30.0
            
            for idx in range(epochs):
                im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
                spec_inp_perturbed = spec_inp + spec_inp_per.normal_() * reg_noise_std
                
                U_img = im_net(im_inp_perturbed)
                V = spec_net(spec_inp_perturbed)
                U = U_img.reshape(-1, rank, self.H*self.W).permute(0, 2, 1)
                hs_estim = torch.bmm(U, V)
                
                loss = criterion_l1(self.hsten_noisy - hs_estim)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_hs_estim = hs_estim.detach().clone()
                    best_im_net_state = copy.deepcopy(im_net.state_dict())
                    best_spec_net_state = copy.deepcopy(spec_net.state_dict())
                    best_im_inp = im_inp.detach().clone()
                    best_spec_inp = spec_inp.detach().clone()
                    
            # Save checkpoint
            torch.save({
                'im_net': best_im_net_state,
                'spec_net': best_spec_net_state,
                'im_inp': best_im_inp,
                'spec_inp': best_spec_inp,
                'best_hs_estim': best_hs_estim,
                'final_loss': best_loss,
                'optimizer_info': 'Adam, lr=1e-3, epochs=' + str(epochs),
                'checkpoint_status': 'FINAL'
            }, chkpt_path)
            
            cube_estim = best_hs_estim.squeeze().cpu().numpy().reshape(self.H, self.W, self.nwvl)
            return {
                'method': 'Baseline',
                'U': U, 'V': V, 'U_img': U_img,
                'reconstruction': cube_estim,
                'hs_estim': best_hs_estim,
                'rank': rank,
                'loss': best_loss,
                'metrics': self._compute_metrics(cube_estim),
                'optimizer_info': 'Adam, lr=1e-3'
            }

    def run_adaptive(self, mode='demo', epochs=5000):
        R_MAX = 30
        n_inputs = 20
        im_net, spec_net, im_inp, spec_inp = self._init_networks(n_inputs, R_MAX)
        
        chkpt_dir = os.path.join(root_dir, 'results', 'extensions', 'adaptive_rank', 'runs', 'adaptive', 'checkpoint')
        if not os.path.exists(os.path.join(chkpt_dir, 'checkpoint.pt')):
            chkpt_dir = os.path.join(root_dir, 'results', 'adaptive', 'checkpoint')
        os.makedirs(chkpt_dir, exist_ok=True)
        chkpt_path = os.path.join(chkpt_dir, 'checkpoint.pt')
        
        if mode == 'demo':
            if not os.path.exists(chkpt_path):
                raise FileNotFoundError(f"[NOT AVAILABLE] Checkpoint not found at {chkpt_path}. Please run --mode full first.")
            ckpt = torch.load(chkpt_path, map_location=self.device)
            if ckpt.get('checkpoint_status') != 'FINAL':
                print("\n[WARNING]\nThis checkpoint is TEST/VALIDATION and cannot be used\nfor the official presentation.\n")
                raise RuntimeError("Checkpoint is not FINAL.")
            im_net.load_state_dict(ckpt['im_net'])
            spec_net.load_state_dict(ckpt['spec_net'])
            im_inp = ckpt['im_inp'].to(self.device)
            spec_inp = ckpt['spec_inp'].to(self.device)
            current_rank = ckpt['current_rank']
            
            im_net.eval()
            spec_net.eval()
            with torch.no_grad():
                U_img = im_net(im_inp)
                V = spec_net(spec_inp)
                U = U_img.reshape(-1, R_MAX, self.H * self.W).permute(0, 2, 1)
                
                U_active = U[:, :, :current_rank]
                V_active = V[:, :current_rank, :]
                
            if 'best_hs_estim' in ckpt:
                hs_estim = ckpt['best_hs_estim'].to(self.device)
            else:
                hs_estim = torch.bmm(U_active, V_active)
            
            cube_estim = hs_estim.squeeze().cpu().numpy().reshape(self.H, self.W, self.nwvl)
            
            return {
                'method': 'Adaptive',
                'U': U_active, 'V': V_active, 'U_img': U_img,
                'reconstruction': cube_estim,
                'hs_estim': hs_estim,
                'rank': current_rank,
                'loss': ckpt.get('final_loss', 0.0),
                'metrics': self._compute_metrics(cube_estim),
                'optimizer_info': ckpt.get('optimizer_info', 'Adam, lr=1e-3'),
                'rank_history': ckpt.get('rank_history', [])
            }
        
        elif mode == 'full':
            import copy
            # Implement training based on run_figure9_adaptive.py
            R_START = 4
            R_STEP = 2
            current_rank = R_START
            
            criterion_l1 = losses.L1Norm()
            im_inp_per = im_inp.detach().clone()
            spec_inp_per = spec_inp.detach().clone()
            
            params = list(im_net.parameters()) + list(spec_net.parameters()) + [im_inp, spec_inp]
            optimizer = torch.optim.Adam(lr=1e-3, params=params)
            
            np.random.seed(42)
            mask_np = np.random.rand(*self.cube_noisy.shape) > 0.05
            mask_train = torch.tensor(mask_np.reshape(self.H * self.W, self.nwvl))[None, ...].to(self.device)
            mask_val = ~mask_train
            np.random.seed(None)
            
            best_val_loss = float('inf')
            best_hs_estim = None
            best_im_net_state = None
            best_spec_net_state = None
            best_im_inp = None
            best_spec_inp = None
            best_rank = R_START
            reg_noise_std = 1.0/30.0
            
            LOSS_EMA_ALPHA = 0.99
            smoothed_train_loss = None
            smoothed_val_loss = None
            val_ema_at_last_plateau = None
            last_increase_epoch = 0
            rank_history = []
            train_loss_ema_history = []
            
            for idx in range(epochs):
                im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
                spec_inp_perturbed = spec_inp + spec_inp_per.normal_() * reg_noise_std
                
                U_img = im_net(im_inp_perturbed)
                V = spec_net(spec_inp_perturbed)
                U = U_img.reshape(-1, R_MAX, self.H*self.W).permute(0, 2, 1)
                
                U_active = U[:, :, :current_rank]
                V_active = V[:, :current_rank, :]
                hs_estim = torch.bmm(U_active, V_active)
                
                diff = self.hsten_noisy - hs_estim
                train_diff = diff[mask_train]
                train_loss = train_diff.abs().mean()
                val_diff = diff[mask_val]
                val_loss = val_diff.abs().mean()
                
                train_loss_val = train_loss.item()
                val_loss_val = val_loss.item()
                
                if smoothed_train_loss is None:
                    smoothed_train_loss = train_loss_val
                    smoothed_val_loss = val_loss_val
                    val_ema_at_last_plateau = val_loss_val
                else:
                    smoothed_train_loss = LOSS_EMA_ALPHA * smoothed_train_loss + (1 - LOSS_EMA_ALPHA) * train_loss_val
                    smoothed_val_loss = LOSS_EMA_ALPHA * smoothed_val_loss + (1 - LOSS_EMA_ALPHA) * val_loss_val
                train_loss_ema_history.append(smoothed_train_loss)
                
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    best_hs_estim = hs_estim.detach().clone()
                    best_im_net_state = copy.deepcopy(im_net.state_dict())
                    best_spec_net_state = copy.deepcopy(spec_net.state_dict())
                    best_im_inp = im_inp.detach().clone()
                    best_spec_inp = spec_inp.detach().clone()
                    best_rank = current_rank
                
                # Plateau detection
                epochs_since_increase = idx - last_increase_epoch
                if current_rank < R_MAX and epochs_since_increase > 300 and idx >= 200:
                    past_train_ema = train_loss_ema_history[idx - 200]
                    rel_train_improvement = (past_train_ema - smoothed_train_loss) / past_train_ema if past_train_ema > 0 else 0
                    if 0 <= rel_train_improvement < 1e-3:
                        if smoothed_val_loss < val_ema_at_last_plateau:
                            old_rank = current_rank
                            current_rank = min(current_rank + R_STEP, R_MAX)
                            last_increase_epoch = idx
                            val_ema_at_last_plateau = smoothed_val_loss
                            rank_history.append((idx, old_rank, current_rank))
            
            # Save checkpoint
            torch.save({
                'im_net': best_im_net_state,
                'spec_net': best_spec_net_state,
                'im_inp': best_im_inp,
                'spec_inp': best_spec_inp,
                'best_hs_estim': best_hs_estim,
                'current_rank': best_rank,
                'final_loss': best_val_loss,
                'optimizer_info': 'Adam, lr=1e-3, epochs=' + str(epochs),
                'rank_history': rank_history,
                'checkpoint_status': 'FINAL'
            }, chkpt_path)
            
            cube_estim = best_hs_estim.squeeze().cpu().numpy().reshape(self.H, self.W, self.nwvl)
            
            # Get U and V for best_hs_estim manually by forward pass of best
            with torch.no_grad():
                im_net.eval()
                spec_net.eval()
                U_img = im_net(im_inp)
                V = spec_net(spec_inp)
                U = U_img.reshape(-1, R_MAX, self.H*self.W).permute(0, 2, 1)
                U_active = U[:, :, :best_rank]
                V_active = V[:, :best_rank, :]
                
            return {
                'method': 'Adaptive',
                'U': U_active, 'V': V_active, 'U_img': U_img,
                'reconstruction': cube_estim,
                'hs_estim': best_hs_estim,
                'rank': best_rank,
                'loss': best_val_loss,
                'metrics': self._compute_metrics(cube_estim),
                'optimizer_info': 'Adam, lr=1e-3',
                'rank_history': rank_history
            }

    def run_sparsity(self, mode='demo', epochs=5000):
        R_MAX = 30
        n_inputs = 20
        
        chkpt_dir = os.path.join(root_dir, 'results', 'extensions', 'sparsity', 'runs', 'sparsity', 'checkpoint')
        if not os.path.exists(os.path.join(chkpt_dir, 'checkpoint.pt')):
            chkpt_dir = os.path.join(root_dir, 'results', 'sparsity', 'checkpoint')
        os.makedirs(chkpt_dir, exist_ok=True)
        chkpt_path = os.path.join(chkpt_dir, 'checkpoint.pt')
        
        def compute_group_regularizer(U, V):
            norm_U_sq = (U[0] ** 2).sum(dim=0)
            norm_V_sq = (V[0] ** 2).sum(dim=1)
            g_r = torch.sqrt(norm_U_sq + norm_V_sq + 1e-12)
            return g_r.sum(), g_r, norm_U_sq, norm_V_sq

        def compute_effective_ranks(norm_U_sq, norm_V_sq):
            u_norms = torch.sqrt(norm_U_sq + 1e-12)
            v_norms = torch.sqrt(norm_V_sq + 1e-12)
            powers = u_norms * v_norms
            powers_np = powers.detach().cpu().numpy()
            max_power = powers_np.max()
            primary_rank = int((powers_np > 0.01 * max_power).sum()) if max_power >= 1e-12 else 0
            return powers_np, primary_rank

        if mode == 'demo':
            im_net, spec_net, im_inp, spec_inp = self._init_networks(n_inputs, R_MAX)
            if not os.path.exists(chkpt_path):
                raise FileNotFoundError(f"[NOT AVAILABLE] Checkpoint not found at {chkpt_path}. Please run --mode full first.")
            ckpt = torch.load(chkpt_path, map_location=self.device)
            if ckpt.get('checkpoint_status') != 'FINAL':
                print("\n[WARNING]\nThis checkpoint is TEST/VALIDATION and cannot be used\nfor the official presentation.\n")
                raise RuntimeError("Checkpoint is not FINAL.")
            im_net.load_state_dict(ckpt['im_net'])
            spec_net.load_state_dict(ckpt['spec_net'])
            im_inp = ckpt['im_inp'].to(self.device)
            spec_inp = ckpt['spec_inp'].to(self.device)
            lambda_val = ckpt.get('lambda_val', 0.0)
            
            im_net.eval()
            spec_net.eval()
            with torch.no_grad():
                U_img = im_net(im_inp)
                V = spec_net(spec_inp)
                U = U_img.reshape(-1, R_MAX, self.H * self.W).permute(0, 2, 1)
                hs_estim = torch.bmm(U, V)
                reg_loss, g_r, norm_U_sq, norm_V_sq = compute_group_regularizer(U, V)
                powers_np, primary_rank = compute_effective_ranks(norm_U_sq, norm_V_sq)
            
            cube_estim = hs_estim.squeeze().cpu().numpy().reshape(self.H, self.W, self.nwvl)
            
            return {
                'method': 'Group Sparsity',
                'U': U, 'V': V, 'U_img': U_img,
                'reconstruction': cube_estim,
                'hs_estim': hs_estim,
                'rank': primary_rank,
                'loss': ckpt.get('final_loss', 0.0),
                'metrics': self._compute_metrics(cube_estim),
                'optimizer_info': ckpt.get('optimizer_info', 'Adam, lr=1e-3'),
                'g_r': g_r.detach().cpu().numpy(),
                'powers': powers_np,
                'lambda_val': lambda_val
            }
        
        elif mode == 'full':
            lambdas = [0, 1e-4, 1e-3, 1e-2]
            best_overall_val_loss = float('inf')
            best_lambda = None
            best_overall_im_net_state = None
            best_overall_spec_net_state = None
            best_overall_im_inp = None
            best_overall_spec_inp = None
            
            print("\n" + "="*70)
            print("STARTING LAMBDA GRID SEARCH (Group Sparsity)")
            print("="*70)
            print(f"{'Lambda':<10} | {'Val L1':<12} | {'Train L1':<12} | {'Penalty':<12} | {'Eff. Rank':<10}")
            print("-" * 70)
            
            for l_val in lambdas:
                torch.manual_seed(42)
                np.random.seed(42)
                im_net, spec_net, im_inp, spec_inp = self._init_networks(n_inputs, R_MAX)
                
                im_inp_per = im_inp.detach().clone()
                spec_inp_per = spec_inp.detach().clone()
                
                params = list(im_net.parameters()) + list(spec_net.parameters()) + [im_inp, spec_inp]
                optimizer = torch.optim.Adam(lr=1e-3, params=params)
                
                mask_np = np.random.rand(*self.cube_noisy.shape) > 0.05
                mask_train = torch.tensor(mask_np.reshape(self.H * self.W, self.nwvl))[None, ...].to(self.device)
                mask_val = ~mask_train
                
                best_val_loss = float('inf')
                best_im_net_state = None
                best_spec_net_state = None
                best_im_inp = None
                best_spec_inp = None
                reg_noise_std = 1.0/30.0
                
                for idx in range(epochs):
                    im_inp_perturbed = im_inp + im_inp_per.normal_() * reg_noise_std
                    spec_inp_perturbed = spec_inp + spec_inp_per.normal_() * reg_noise_std
                    
                    U_img = im_net(im_inp_perturbed)
                    V = spec_net(spec_inp_perturbed)
                    U = U_img.reshape(-1, R_MAX, self.H*self.W).permute(0, 2, 1)
                    hs_estim = torch.bmm(U, V)
                    
                    diff = self.hsten_noisy - hs_estim
                    train_diff = diff[mask_train]
                    train_l1_loss = train_diff.abs().mean()
                    val_diff = diff[mask_val]
                    val_l1_loss = val_diff.abs().mean()
                    
                    reg_loss, _, _, _ = compute_group_regularizer(U, V)
                    total_loss = train_l1_loss + l_val * reg_loss
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    if val_l1_loss.item() < best_val_loss:
                        best_val_loss = val_l1_loss.item()
                        best_im_net_state = copy.deepcopy(im_net.state_dict())
                        best_spec_net_state = copy.deepcopy(spec_net.state_dict())
                        best_im_inp = im_inp.detach().clone()
                        best_spec_inp = spec_inp.detach().clone()
                
                # Evaluate effective rank and final metrics on this lambda's best model
                im_net.load_state_dict(best_im_net_state)
                spec_net.load_state_dict(best_spec_net_state)
                with torch.no_grad():
                    im_net.eval()
                    spec_net.eval()
                    U_img = im_net(best_im_inp)
                    V = spec_net(best_spec_inp)
                    U = U_img.reshape(-1, R_MAX, self.H*self.W).permute(0, 2, 1)
                    reg_loss, _, norm_U_sq, norm_V_sq = compute_group_regularizer(U, V)
                    _, primary_rank = compute_effective_ranks(norm_U_sq, norm_V_sq)
                    hs_estim = torch.bmm(U, V)
                    diff = self.hsten_noisy - hs_estim
                    train_diff = diff[mask_train]
                    train_l1_loss = train_diff.abs().mean()
                
                final_train_l1 = train_l1_loss.item()
                final_penalty = reg_loss.item()
                print(f"{l_val:<10} | {best_val_loss:<12.6f} | {final_train_l1:<12.6f} | {final_penalty:<12.6f} | {primary_rank:<10}")
                
                if best_val_loss < best_overall_val_loss:
                    best_overall_val_loss = best_val_loss
                    best_lambda = l_val
                    best_overall_im_net_state = copy.deepcopy(best_im_net_state)
                    best_overall_spec_net_state = copy.deepcopy(best_spec_net_state)
                    best_overall_im_inp = best_im_inp.clone()
                    best_overall_spec_inp = best_spec_inp.clone()

            print("-" * 70)
            print(f"Grid search complete. Selected Lambda: {best_lambda} with Val L1: {best_overall_val_loss:.6f}")
            print("="*70 + "\n")
            
            # Save checkpoint for the selected lambda
            torch.save({
                'im_net': best_overall_im_net_state,
                'spec_net': best_overall_spec_net_state,
                'im_inp': best_overall_im_inp,
                'spec_inp': best_overall_spec_inp,
                'final_loss': best_overall_val_loss,
                'lambda_val': best_lambda,
                'optimizer_info': 'Adam, lr=1e-3, epochs=' + str(epochs) + ', lambda=' + str(best_lambda),
                'checkpoint_status': 'FINAL'
            }, chkpt_path)
            
            # Load best for final result construction
            im_net.load_state_dict(best_overall_im_net_state)
            spec_net.load_state_dict(best_overall_spec_net_state)
            with torch.no_grad():
                im_net.eval()
                spec_net.eval()
                U_img = im_net(best_overall_im_inp)
                V = spec_net(best_overall_spec_inp)
                U = U_img.reshape(-1, R_MAX, self.H*self.W).permute(0, 2, 1)
                hs_estim = torch.bmm(U, V)
                _, g_r, norm_U_sq, norm_V_sq = compute_group_regularizer(U, V)
                powers_np, primary_rank = compute_effective_ranks(norm_U_sq, norm_V_sq)
                
            cube_estim = hs_estim.squeeze().cpu().numpy().reshape(self.H, self.W, self.nwvl)
            
            return {
                'method': 'Group Sparsity',
                'U': U, 'V': V, 'U_img': U_img,
                'reconstruction': cube_estim,
                'hs_estim': hs_estim,
                'rank': primary_rank,
                'loss': best_overall_val_loss,
                'metrics': self._compute_metrics(cube_estim),
                'optimizer_info': 'Adam, lr=1e-3',
                'g_r': g_r.detach().cpu().numpy(),
                'powers': powers_np,
                'lambda_val': best_lambda
            }

    def _compute_metrics(self, cube_estim):
        data_range = self.cube.max() - self.cube.min()
        psnr = utils.psnr(self.cube, cube_estim)
        ssim = ssim_func(self.cube, cube_estim, channel_axis=-1, data_range=data_range)
        rmse = np.sqrt(((self.cube - cube_estim) ** 2).mean())
        return {'PSNR': float(psnr), 'SSIM': float(ssim), 'RMSE': float(rmse)}
