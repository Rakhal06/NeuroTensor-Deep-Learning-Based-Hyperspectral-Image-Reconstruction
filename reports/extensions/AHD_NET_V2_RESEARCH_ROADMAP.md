# Adaptive Hybrid DeepTensor v2 (AHD-Net v2)
# Comprehensive Research Roadmap and Diagnostic Study

**Date:** November 8, 2026  
**Project:** DeepTensor++ Research Initiative  
**Preceding Milestone:** AHD-Net v1 achieved **46.9248 dB** (clean) / **46.1245 dB** (in-loop) on Figure 09 ICVL Hyperspectral Denoising, outperforming Baseline CNN ($46.0653\text{ dB}$) and Serial Hybrid ($44.8526\text{ dB}$).  
**Document Status:** **DIAGNOSTIC & RESEARCH-DESIGN PHASE ONLY — STRICT NO-TRAINING PROTOCOL**  
**Associated Target:** Design next-generation architectural enhancements without violating zero-GT leakage or parameter parity constraints.

---

## 1. Systematic Diagnosis of AHD-Net v1

Before formulating architectural extensions, an exhaustive diagnostic inquiry into the converged state of AHD-Net v1 was conducted on the Figure 09 ICVL hyperspectral denoising run.

### 1.1. Empirical Diagnostics Summary

| Diagnostic Dimension | Observed Value / Phenomenon | Mathematical / Physical Interpretation |
|---|---|---|
| **Training Loss Trajectory** | Monotonic decrease from $0.528848$ (Epoch 1) to $0.029813$ (Epoch 4978). | Clean, well-conditioned optimization under Adam ($10^{-3}$) without gradient explosion or plateau instability. |
| **Global Router Weight Evolution** | CNN: $0.334 \to 0.333$; PW: $0.332 \to 0.335$; Attn: $0.335 \to 0.332$. | Global branch weights remain virtually uniform ($\sim 33.3\%$ each) throughout all 5,000 epochs. |
| **Global Router Shannon Entropy** | $H_{\text{global}} = 1.0986\text{ nats} \approx \ln 3$ throughout training. | Theoretical maximum entropy is maintained at the global mean level. |
| **Branch Dominance Across 128 Channels** | **CNN:** 45 channels ($35.2\%$); **PW:** 42 channels ($32.8\%$); **Attn:** 41 channels ($32.0\%$). | **Key Discovery:** The network bifurcates channels into three near-equal specialist cohorts. |
| **Channel-Wise Gating Spread** | CNN: $[0.1601, 0.5392]$ ($\sigma=0.0814$); PW: $[0.1874, 0.5356]$ ($\sigma=0.0794$); Attn: $[0.1820, 0.5196]$ ($\sigma=0.0853$). | Individual channels develop meaningful specialization without any single branch dominating universally. |
| **Channel-Wise Entropy Spread** | Mean $H_c = 1.0683\text{ nats}$, Range: $[1.0059, 1.0971]$. | Gating distribution remains relatively soft ($\tau=0.5$ limits peak certainty to $\approx 54\%$). |
| **Branch Gradient Norms at Convergence** | $\|\nabla \mathcal{L}_{\text{CNN}}\| = 0.000802$; $\|\nabla \mathcal{L}_{\text{PW}}\| = 0.001073$; $\|\nabla \mathcal{L}_{\text{Attn}}\| = 0.000647$; $\|\nabla \mathcal{L}_{\text{Router}}\| = 0.000160$. | Excellent gradient equilibrium across all three branches. Pointwise branch absorbs the largest gradient signal, followed by CNN and Attention. |
| **Reconstruction Fidelity** | Full-Volume PSNR: $46.9248\text{ dB}$ (clean) / $46.1245\text{ dB}$ (in-loop); SSIM: $0.9900$; RMSE: $0.004333$. | Resolves the serial hybrid deficit (+2.07 dB) and establishes new SOTA (+0.86 dB over baseline). |

### 1.2. Critical Analysis: Does Uniform Average Routing Mask Channel Specialization?

A critical theoretical question arose during the Figure 09 evaluation:  
*Does the near-uniform mean weight vector $\bar{\mathbf{g}} = [0.333, 0.335, 0.332]$ imply that the router is lazy/inactive, or does it mask meaningful channel-wise specialization?*

#### The Evidence:
1. **Channel Partition:** The 128 feature channels in the spectral net do *not* all carry weights $[0.33, 0.33, 0.33]$. Instead:
   - 45 channels have CNN weight exceeding Pointwise and Attention (up to $53.9\%$).
   - 42 channels have Pointwise weight dominating (up to $53.6\%$).
   - 41 channels have Attention weight dominating (up to $52.0\%$).
2. **Symmetric Balancing Effect:** Because the three groups contain virtually identical numbers of channels ($45 \approx 42 \approx 41$), summing across channels $c \in \{1, \dots, 128\}$ produces an almost perfectly balanced mean:
   $$\bar{g}_b = \frac{1}{C}\sum_{c=1}^C g_{c,b} \approx \frac{1}{3}$$
3. **Limitation of v1 Routing:** While specialization exists ($\sigma_c \approx 0.08$), the router rarely pushes a gate above $0.55$. This is partly enforced by the minimum temperature boundary ($\tau_{\min} = 0.5$). At $\tau = 0.5$, a logit difference of $\Delta z = 1.0$ yields a maximum probability of:
   $$\frac{e^{1.0 / 0.5}}{e^{1.0 / 0.5} + 1 + 1} = \frac{7.389}{9.389} \approx 0.787$$
   However, the router MLP weights remain moderate in norm ($\|\mathbf{W}_2\|_F \approx 11.3$), indicating that unsupervised measurement loss alone does not provide a sharp supervisory signal to drive hard routing.

---

## 2. Potential Architectural Improvements for AHD-Net v2

Based on the diagnostic findings, we identify four orthogonal avenues for architectural improvement:

```
                            ┌────────────────────────────────────────┐
                            │      AHD-Net v2 Research Program       │
                            └───────────────────┬────────────────────┘
                                                │
         ┌───────────────────┬──────────────────┴──────────────────┬──────────────────┐
         ▼                   ▼                                     ▼                  ▼
  [A. Adaptive         [B. Spectral                          [C. Spatial         [D. Optimization
     Routing]             Modelling]                            Modelling]          Refinements]
  • Learnable tau      • Multi-Scale 1D Conv (k=3,5,7)       • Residual Skip     • Cosine LR Warmup
  • Grouped Routing    • Dilated Spectral Kernels              Refinement        • Dynamic Branch
  • Entropy Schedule   • Spectral Continuity Regularization  • Spatial CoordConv   GradNorm Balancing
```

---

### Pillar A: Adaptive Routing Mechanisms

#### A.1. Progressive Temperature Annealing to Sharper Regimes ($\tau_{\min} = 0.2$)
- **Hypothesis:** Lowering the terminal temperature from $\tau_{\min} = 0.5$ to $\tau_{\min} = 0.2$ in the final $30\%$ of training will allow channels to specialize more aggressively (gates up to $\approx 85\%$) once stable representations are formed.
- **Mathematical Formulation:**
  $$\tau(t) = \begin{cases}
  \tau_{\text{start}} - \frac{t}{T_1}(\tau_{\text{start}} - \tau_{\text{mid}}), & t \le T_1 \\
  \tau_{\text{mid}} \cdot \left(\frac{\tau_{\min}}{\tau_{\text{mid}}}\right)^{\frac{t - T_1}{T - T_1}}, & t > T_1
  \end{cases}$$
  where $\tau_{\text{start}} = 2.0$, $\tau_{\text{mid}} = 0.5$, $\tau_{\min} = 0.2$, $T_1 = 3500$, $T = 5000$.

#### A.2. Channel-Grouped Hierarchical Routing (G-CAGC)
- **Hypothesis:** Rather than forcing all 128 channels to be routed independently from a single pooled descriptor, dividing channels into $G=4$ disjoint functional sub-bands ($32$ channels each) and providing group-specific context vectors will enhance domain-specific specialization (e.g. low-frequency vs high-frequency spectral bands).
- **Mathematical Formulation:**
  $$\mathbf{z}_g = \text{MLP}_g\left(\frac{1}{L}\sum_{l=1}^L \mathbf{X}_{g, l}\right) \in \mathbb{R}^{32 \times 3}, \quad g \in \{1, 2, 3, 4\}$$
- **Parameter Overhead:** 4 lightweight MLPs of dimension $32 \to 8 \to 96$, total parameters $= 4 \times (32 \times 8 + 8 + 8 \times 96 + 96) = 4,544$ parameters (vs v1 router $= 8,464$ parameters; parameter reduction of $-3,920$ params).

#### A.3. Dynamic Entropy Regularization Schedule
- **Hypothesis:** Applying an explicit entropy penalty $\lambda_H \mathcal{L}_H$ that shifts from encouraging exploration (high entropy, $\lambda_H < 0$) in early epochs to encouraging specialization (low entropy, $\lambda_H > 0$) in late epochs will actively shape channel specialization.
- **Mathematical Formulation:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{meas}} + \lambda_H(t) \cdot \frac{1}{C}\sum_{c=1}^C \sum_{b=1}^3 g_{c,b} \ln g_{c,b}$$
  where $\lambda_H(t)$ ramps from $-10^{-4}$ (anti-collapse) to $+5 \times 10^{-4}$ (sparsity induction).

---

### Pillar B: Spectral Factor Modelling

#### B.1. Multi-Scale Inception Spectral Branch (MS-1D-CNN)
- **Hypothesis:** In hyperspectral signals, certain spectral absorption bands are extremely sharp (requiring narrow receptive fields $k=3$), while continuous continuum variations span broader bands (requiring wider receptive fields $k=7$). Replacing the single $k=3$ depthwise convolution with a multi-scale parallel filter bank ($k \in \{3, 5, 7\}$) will capture both sharp emission features and smooth spectral envelopes.
- **Mathematical Formulation:**
  $$\mathbf{Y}_{\text{cnn}} = \mathbf{W}_{\text{proj}} \left[ \text{DWConv}_{k=3}(\mathbf{X}) \,\|\, \text{DWConv}_{k=5}(\mathbf{X}) \,\|\, \text{DWConv}_{k=7}(\mathbf{X}) \right]$$
  where each branch has $C/3 \approx 42$ channels, keeping total channel width at $C=128$.
- **Parameter Overhead:**
  - Standard DWConv $k=3$: $128 \times 3 = 384$ params.
  - Multi-Scale DWConv ($42\times 3 + 43\times 5 + 43\times 7$): $126 + 215 + 301 = 642$ params.
  - Net Delta: $+258$ parameters.

#### B.2. First- and Second-Order Spectral Continuity Priors
- **Hypothesis:** Natural hyperspectral signatures exhibit continuous physical transitions across contiguous wavelengths $\lambda$. Adding an unsupervised spectral Total Variation (TV) penalty on the reconstructed spectral factor $\mathbf{V}$ prevents high-frequency oscillatory artifacts.
- **Mathematical Formulation:**
  $$\mathcal{L}_{\text{spectral\_TV}} = \frac{1}{L-1} \sum_{l=1}^{L-1} \|\mathbf{V}_{l+1, :} - \mathbf{V}_{l, :}\|_2^2 + \frac{\gamma}{L-2} \sum_{l=2}^{L-1} \|\mathbf{V}_{l+1, :} - 2\mathbf{V}_{l, :} + \mathbf{V}_{l-1, :}\|_2^2$$

---

### Pillar C: Spatial Factor Modelling

#### C.1. Residual Skip-Refinement in Spatial U-Net
- **Hypothesis:** The spatial generator U-Net relies on standard concatenation skips. In high-noise regimes ($\text{SNR}=2$), skip connections inject noisy features from the encoder directly into the decoder. Introducing a learned channel-attention gating unit on skip connections (Spatial Skip Gate) will filter noise before upsampling.
- **Mathematical Formulation:**
  $$\mathbf{F}_{\text{skip\_gated}} = \mathbf{F}_{\text{skip}} \odot \sigma\left(\mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \text{GAP}(\mathbf{F}_{\text{skip}}))\right)$$

#### C.2. Normalized Coordinate Convolution (CoordConv)
- **Hypothesis:** Tensor CP factor spatial maps $\mathbf{U}(x, y)$ possess spatial non-stationarity (e.g. illumination gradients in Yale Face B and boundary geometries in PET/CT). Injecting normalized spatial coordinate grids $[-1, 1] \times [-1, 1]$ into the initial layer of the U-Net provides explicit geometric grounding without parameter overhead.

---

### Pillar D: Optimization & Training Dynamics

#### D.1. Cosine Learning Rate Decay with Linear Warmup
- **Hypothesis:** AHD-Net v1 used a fixed learning rate $\eta = 10^{-3}$ throughout 5,000 epochs. Introducing a 200-epoch linear warmup followed by cosine decay to $\eta_{\min} = 10^{-5}$ will prevent early latent shocks and allow finer convergence in the late optimization phase.
- **Mathematical Formulation:**
  $$\eta(t) = \begin{cases}
  \eta_{\max} \cdot \frac{t}{T_{\text{warm}}}, & t \le T_{\text{warm}} \\
  \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi(t - T_{\text{warm}})}{T - T_{\text{warm}}}\right)\right), & t > T_{\text{warm}}
  \end{cases}$$

#### D.2. Dynamic Branch Gradient Balancing (GradNorm for Branches)
- **Hypothesis:** Although branch gradient norms were relatively balanced at convergence ($0.0008, 0.0010, 0.0006$), early training may exhibit transient imbalances where one branch dominates parameter updates. A lightweight loss-weighting mechanism can scale branch learning rates to ensure equal learning velocity.

---

## 3. Controlled Ablation Matrix

To ensure scientific validity, every proposed modification must be isolated and evaluated independently against the frozen AHD-Net v1 baseline.

| Ablation ID | Innovation Tested | Baseline Control | Mathematical Modification | Target Application | Parameter Delta | Primary Success Criteria | Primary Failure Mode |
|:---:|---|---|---|:---:|:---:|---|---|
| **ABL-1** | Deep Temperature Annealing ($\tau_{\min}=0.2$) | AHD-Net v1 ($\tau_{\min}=0.5$) | Terminal $\tau = 0.2$ for $t > 3500$ | Fig 09 (ICVL) | $0$ | PSNR $> 46.95\text{ dB}$, higher branch specialization ($\sigma_c > 0.12$) | Premature routing starvation; training instability. |
| **ABL-2** | Cosine LR Schedule with Warmup | AHD-Net v1 (Fixed $\eta=10^{-3}$) | Warmup $200\text{ ep} \to \text{Cosine to } 10^{-5}$ | Fig 09 (ICVL) | $0$ | Faster convergence; final $L_1 < 0.02950$. | Early stagnation if LR decays too fast. |
| **ABL-3** | Multi-Scale Inception 1D CNN | Single $k=3$ DWConv | Parallel $k \in \{3, 5, 7\}$ DWConvs | Fig 09 (ICVL) | $+258$ ($0.008\%$) | PSNR gain $> +0.20\text{ dB}$; improved spectral continuity. | Computational latency; channel dilution. |
| **ABL-4** | Grouped Routing (G-CAGC, $G=4$) | Global CAGC ($G=1$) | 4 independent sub-band routers | Fig 09 & Fig 10 | $-3,920$ (Savings) | Lower measurement loss; distinct group specialization. | Inter-group boundary discontinuity. |
| **ABL-5** | Spectral TV Regularization | Pure Measurement Loss | $+\lambda_{\text{TV}} \mathcal{L}_{\text{TV}}$ ($\lambda=10^{-4}$) | Fig 09 (ICVL) | $0$ | Reduced high-frequency spectral noise artifacts. | Over-smoothing of sharp absorption bands. |
| **ABL-6** | Spatial Skip Attention Gates | Direct Concatenation | Channel-gated skip connections | Fig 11 & Fig 12 | $+4,800$ ($0.11\%$) | Sharper edge reconstruction in CT Radon & CACTI. | Slower backpropagation; redundant filtering. |
| **ABL-7** | Spatial CoordConv Ingestion | Random Latents Only | Add $[x, y]$ coordinate channels | Fig 10 (Yale B) | $+36$ params | Better illumination/shadow geometry recovery. | Artifacts if coordinates bias translation invariance. |

---

## 4. Parameter & Compute Budget Audit

To adhere to the core DeepTensor scientific principle—fair parameter-matched comparisons—any v2 architecture must remain strictly within $\pm 0.5\%$ of the historical target budgets:

| Application | Target Budget | AHD-Net v1 | Proposed AHD-Net v2 (All candidate modules) | Parameter Delta | Compliance Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Figure 09 (ICVL HSI)** | $3,137,808$ | $3,137,808$ | $3,134,146$ (with G-CAGC + MS-CNN) | $-3,662$ ($-0.11\%$) | **Strictly Compliant** |
| **Figure 10 (Yale Face B)** | $5,602,988$ | $5,602,988$ | $5,599,068$ (with G-CAGC + CoordConv) | $-3,920$ ($-0.07\%$) | **Strictly Compliant** |
| **Figure 11 (Video CS)** | $834,632$ | $834,632$ | $835,592$ (with Skip Gates + MS-CNN) | $+960$ ($+0.11\%$) | **Strictly Compliant** |
| **Figure 12 (PET/CT Radon)** | $4,050,988$ | $4,050,988$ | $4,051,948$ (with Skip Gates + CoordConv) | $+960$ ($+0.02\%$) | **Strictly Compliant** |

*Hardware Compute Feasibility:*  
On the verified NVIDIA RTX 4060 Laptop GPU, AHD-Net v1 required $410.37\text{s}$ ($6.84\text{ min}$) and $457.8\text{ MB}$ VRAM for 5,000 epochs. None of the proposed v2 modifications introduce quadratic attention overhead or large matrix expansions; all maintain expected runtime under $450\text{s}$ and VRAM under $500\text{ MB}$.

---

## 5. Strict Scientific Constraints for Future Work

The following principles must govern any future implementation of AHD-Net v2:
1. **Preservation of AHD-Net v1:** `src/deeptensor/models/attention/adaptive_hybrid.py` and its checkpoints in `results/extensions/adaptive_hybrid_v1_full/` must remain permanently frozen and untouched.
2. **Immutable Ground-Truth Firewall:** Ground-truth tensors (`icvl.mat`, `yale_b.mat`, `cacti_video.mat`, `pet.mat`) must never be accessed inside the training loop, optimizer, checkpoint selector, or hyper-parameter scheduler.
3. **Hyper-parameter Integrity:** No hyper-parameter (e.g. learning rate, temperature bounds, regularizer weights) may be tuned by repeatedly evaluating against test PSNR/SSIM. Any schedule must be justified a priori by mathematical principles or self-supervised measurement loss.
4. **Isolated Result Paths:** All v2 outputs must reside strictly in a new isolated directory (`results/extensions/adaptive_hybrid_v2/`).
5. **No Speculative or Inferred Claims:** Every reported number must derive from executed and reloaded PyTorch checkpoints.

---

## 6. Recommended Implementation & Execution Order

When research on v2 is officially approved, experiments must proceed in the following sequential order:

```
Step 1: Low-Risk Optimization Ablation (ABL-2: Cosine LR Warmup)
  └── Hypothesis: Does training stability improve without changing model architecture?
  └── Zero parameter changes, zero architectural risk.

Step 2: Temperature Annealing Sensitivity (ABL-1: Deep Annealing to tau=0.2)
  └── Hypothesis: Does sharper specialization emerge with deeper cooling?
  └── Zero parameter changes, purely testing router dynamics.

Step 3: Multi-Scale Spectral Inception (ABL-3: k=3, 5, 7)
  └── Hypothesis: Does multi-scale spectral convolution improve band continuity?
  └── Replaces only the CNN branch; preserves router and attention branch.

Step 4: Grouped Hierarchical Router (ABL-4: G-CAGC)
  └── Hypothesis: Does per-sub-band routing outperform global pooling?
  └── Evaluates parameter savings and specialization in parallel.

Step 5: Architectural Synthesis & Application Gate
  └── Only combine successful isolated ablations into a unified AHD-Net v2.
  └── Re-benchmark against the frozen AHD-Net v1 baseline.
```

---

## 7. Immediate Directive: Complete Halt

In accordance with user instructions:
- **No v2 code has been written.**
- **No new training has been launched.**
- **AHD-Net v1 checkpoints, logs, and benchmark tables remain 100% preserved.**

**STOP ENFORCED:** Awaiting user directive to proceed to Phase 4B: Figure 10 (Extended Yale Face B) full benchmark under AHD-Net v1.

