# AHD-Net v2 Controlled Research Ablation Plan
## Formal Specification of Controlled Ablation Experiments for Next-Generation Adaptive Hybrid Architectures

**Date:** November 8, 2026  
**Project:** DeepTensor++ Research Initiative  
**Author:** DeepTensor++ Autonomous Research Agent  
**Status:** **RESEARCH PROPOSAL ONLY — STRICT STOP ENFORCED (NO EXECUTION WITHOUT APPROVAL).**

---

## 1. Executive Summary & Motivation

The full-budget benchmarks of **Adaptive Hybrid DeepTensor v1 (AHD-Net)** established that parallel tri-branch processing with adaptive gating successfully cures the serial hybrid pathology (+2.07 dB on HSI, +0.61 dB on Video CS) and universally improves upon the Pure Baseline CNN across all four modalities.

However, the final validation identified two crucial architectural bottlenecks:
1. **The Myopic Greedy Router Trap (Yale Face B):** The unsupervised CAGC router reinforced Pointwise mixing ($49\%$) to achieve rapid early $L_1$ loss reduction on salt-and-pepper noise, prematurely starving the Attention branch down to $14\%$ and causing AHD-Net to lag the Pure Transformer by $-0.71\text{ dB}$.
2. **Parameter Dilution (PET/CT Radon CS):** Splitting a fixed parameter budget equally across three branches gave the Pointwise branch only $\sim 1/3$ the dedicated channel capacity of the pure Pointwise Control model, causing AHD-Net to lag Pointwise Control by $-0.37\text{ dB}$.

To decouple the effects of **branch capacity**, **adaptive gating vs. static ensembling**, **temperature dynamics**, and **router regularization**, this document specifies a controlled 7-condition ablation matrix for future research.

---

## 2. Controlled 7-Condition Ablation Matrix

All ablation experiments are designed to run under strictly identical experimental protocols (same seeds, sensing operators, noise realizations, loss functions, and budgets) on the two primary diagnostic testbeds:
* **Testbed 1: Figure 10 (Yale Face B, $R=2000$, 1,200 epochs)** — tests resolution of attention starvation under discontinuous lighting and impulsive noise.
* **Testbed 2: Figure 11 (Video CS, $R=32$, 3,000 epochs)** — tests resolution of temporal motion prioritization and translational equivariance.

### 2.1. Master Ablation Design Table

| Condition ID | Ablation Name | Routing Mechanism | Active Branches | Router Parameters | Temperature Schedule | Loss Objective | Research Hypothesis / Target Finding |
|---|---|---|---|:---:|:---:|---|---|
| **ABL-1** | **CNN-Only AHD-Net** | Fixed Clamped | CNN only ($w = [1, 0, 0]$) | Frozen / None | N/A | Standard Unsupervised $\mathcal{L}_{\text{meas}}$ | Measures baseline capacity of the tri-branch block when restricted exclusively to local 1D convolutions. |
| **ABL-2** | **Pointwise-Only AHD-Net** | Fixed Clamped | Pointwise only ($w = [0, 1, 0]$) | Frozen / None | N/A | Standard Unsupervised $\mathcal{L}_{\text{meas}}$ | Measures baseline capacity when restricted exclusively to inter-channel CP rank-mixing MLPs. |
| **ABL-3** | **Attention-Only AHD-Net** | Fixed Clamped | Attention only ($w = [0, 0, 1]$) | Frozen / None | N/A | Standard Unsupervised $\mathcal{L}_{\text{meas}}$ | Directly tests whether attention within the AHD-Net block can match the Pure Transformer when not starved by competing branches. |
| **ABL-4** | **Static Uniform Ensemble** | Fixed Equal | All three ($w = [1/3, 1/3, 1/3]$) | None ($0$ params) | N/A | Standard Unsupervised $\mathcal{L}_{\text{meas}}$ | **Crucial Baseline:** Isolates the quantitative gain of *adaptive learned gating* versus a *static parallel ensemble* of all three branches. |
| **ABL-5** | **Standard AHD-Net v1 (Reference)** | CAGC Softmax | All three (learned dynamic) | Fully active ($4,160$ params) | Cosine Anneal: $\tau: 2.0 \to 0.5$ | Standard Unsupervised $\mathcal{L}_{\text{meas}}$ | Benchmark reference from Phase 4 full-budget runs. |
| **ABL-6A** | **Temperature Sharpening** | CAGC Softmax | All three (learned dynamic) | Fully active ($4,160$ params) | Aggressive Anneal: $\tau: 2.0 \to 0.1$ | Standard Unsupervised $\mathcal{L}_{\text{meas}}$ | Tests whether forcing near-hard single-branch selection per channel improves performance by preventing feature dilution. |
| **ABL-6B** | **Conservative Temperature** | CAGC Softmax | All three (learned dynamic) | Fully active ($4,160$ params) | Soft Anneal: $\tau: 2.0 \to 1.0$ | Standard Unsupervised $\mathcal{L}_{\text{meas}}$ | Tests whether preventing extreme gate divergence prevents branch starvation. |
| **ABL-6C** | **Constant Temperature** | CAGC Softmax | All three (learned dynamic) | Fully active ($4,160$ params) | Fixed Constant: $\tau = 1.0$ (no anneal) | Standard Unsupervised $\mathcal{L}_{\text{meas}}$ | Evaluates whether dynamic temperature annealing is necessary or if standard unit temperature suffices. |
| **ABL-7A** | **Gating Floor Constraint** | CAGC Softmax + Floor | All three (learned dynamic) | Fully active ($4,160$ params) | Cosine Anneal: $\tau: 2.0 \to 0.5$ | $\mathcal{L}_{\text{meas}}$ with $w_b \ge \epsilon$ ($\epsilon = 0.15$) | Directly tests whether preventing Attention from falling below $15\%$ solves the Yale Face B performance deficit. |
| **ABL-7B** | **Entropy Regularization** | CAGC Softmax + Entropy | All three (learned dynamic) | Fully active ($4,160$ params) | Cosine Anneal: $\tau: 2.0 \to 0.5$ | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{meas}} - \lambda_{\text{ent}} H(\mathbf{w})$ ($\lambda=0.01$) | Tests whether encouraging high router entropy maintains exploration and prevents premature lock-in. |

---

## 3. Detailed Experimental Specifications

### 3.1. Mathematical Formulations

#### 1. Static Uniform Baseline (ABL-4)
For feature input $\mathbf{X} \in \mathbb{R}^{B \times C \times L}$:
$$\mathbf{F}_{\text{fused}} = \frac{1}{3} \mathbf{F}_{\text{cnn}} + \frac{1}{3} \mathbf{F}_{\text{pw}} + \frac{1}{3} \mathbf{F}_{\text{attn}}$$
$$\mathbf{Y} = \mathbf{X} + \mathbf{F}_{\text{fused}}$$
*Parameter Difference:* Saves exactly $4,160$ parameters on Mode 3 (router is deleted). Provides an exact test of whether adaptive gating provides value over simple parallel addition.

#### 2. Gating Floor Constraint (ABL-7A)
To prevent the myopic collapse of any branch:
$$\tilde{\mathbf{w}} = \text{Softmax}\left(\frac{\mathbf{z}}{\tau}\right)$$
$$\mathbf{w}_{\text{floored}} = (1 - 3\epsilon) \tilde{\mathbf{w}} + \epsilon, \quad \epsilon = 0.15$$
This guarantees that for every channel $c$ and branch $b$, $w_{c, b} \ge 0.15$, ensuring backpropagated gradient norms $\|\nabla \mathcal{L}_b\| \ge 0.15 \|\nabla \mathcal{L}_{\text{fused}}\|$ throughout training.

#### 3. Entropy Regularization Objective (ABL-7B)
The total unsupervised optimization loss is modified to:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{meas}}(\mathbf{Y}_{\text{meas}}, \mathcal{A}(\hat{\mathbf{X}})) - \lambda_{\text{ent}} \cdot \frac{1}{C} \sum_{c=1}^C H(\mathbf{w}_c)$$
where $H(\mathbf{w}_c) = -\sum_{b=1}^3 w_{c, b} \ln(w_{c, b})$.
*Hyperparameter Sweep:* $\lambda_{\text{ent}} \in \{0.001, 0.01, 0.05\}$.

---

## 4. Hardware Budgets & Execution Feasibility

| Experiment Suite | Testbed Application | Budget (Epochs) | Estimated Runtime / Run | Number of Conditions | Total GPU Hours |
|---|---|:---:|:---:|:---:|:---:|
| **Suite 1: Branch Isolation & Static Ensemble** | Figure 10 (Yale Face B) | 1,200 | $\approx 2.5\text{ min}$ | 4 (ABL-1, 2, 3, 4) | $0.17\text{ hours}$ |
| **Suite 2: Temperature & Regularization** | Figure 10 (Yale Face B) | 1,200 | $\approx 2.6\text{ min}$ | 5 (ABL-6A, 6B, 6C, 7A, 7B) | $0.22\text{ hours}$ |
| **Suite 3: Video CS Inductive Validation** | Figure 11 (Video CS) | 3,000 | $\approx 4.1\text{ min}$ | 4 (ABL-1, 3, 4, 7A) | $0.27\text{ hours}$ |
| **Total Ablation Matrix** | Figures 10 & 11 | — | — | **13 runs** | **$\approx 0.66\text{ GPU hours}$** |

All 13 proposed ablation runs can be executed on the NVIDIA RTX 4060 Laptop GPU in under **40 total minutes**.

---

## 5. Success Criteria & Architectural Decision Tree for AHD-Net v2

The results of this ablation matrix will dictate the exact architectural changes for **AHD-Net v2**:

```
                       [ ABLATION MATRIX RESULTS ]
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
[ ABL-4 vs ABL-5: Static vs Adaptive ]     [ ABL-7A / 7B: Floor & Entropy on Yale ]
  If Static (ABL-4) matches Adaptive:        If Floor (ABL-7A) restores >28.3 dB:
    ==> Router is unnecessary overhead;        ==> Adopt Gating Floor in AHD-Net v2.
        transition to fixed tri-branch.      If Attention (ABL-3) cannot match Transformer:
  If Adaptive (ABL-5) outperforms Static:      ==> Parameter dilution is the root cause;
    ==> Confirm value of adaptive gating;          transition to Dynamic MoE routing.
        proceed to router refinement.
```

---

## 6. Safety Mandate & Gate Enforcement

**STRICT GATE STATUS: STOP ENFORCED.**  
This document is a design specification only. No code modifications to `src/deeptensor/models/attention/adaptive_hybrid.py` or training executions will be launched until formal user approval is granted.

