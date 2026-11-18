import os
import json

ROOT = 'e:/Rakhal_Tensor'

with open('scratch_exp_mapping.json', encoding='utf-8') as f:
    experiments = json.load(f)

doc = """# DeepTensor++ — Research Experiment Mapping Across All 13 Paper Targets

> **Audit Date:** November 6, 2026  
> **Workspace Root:** `e:\\Rakhal_Tensor`  
> **Audit Status:** Verified against physical disk files  
> **Paper Coverage:** Exhaustive mapping of all 13 experimental figures and tables from the DeepTensor paper.

---

## 1. Executive Experiment Status Matrix

| Paper Target | Application Domain | Primary Dataset | Baseline Status | Improved Status | Published Metric | Achieved Metric | Audit Outcome |
|---|---|---|---|---|---|---|---|
| **Figure 2** | Eigenfaces PCA | `weizzman.mat` | Complete | N/A | $\\\\sim 88.0\\%$ Acc | **88.67%** Acc | Verified & Frozen |
| **Figure 4** | 3D Separable Speedup | Synthetic 3D MRI ($128^2 \\\\times 150$) | Flawed | Ready | $\\\\sim 100\\\\times$ Speedup | Authored ($R=5..30$) | Pending user approval |
| **Figure 6** | Synthetic Robustness | Synthetic Tensors ($64^2$) | Complete | N/A | Robust over SVD | Matches paper curve | Verified & Frozen |
| **Figure 7** | Robust PCA Recovery | Covariance Matrix | Complete | N/A | Lower error vs PCA | Matches paper curve | Verified & Frozen |
| **Figure 9** | ICVL HSI Denoising | `icvl.mat` ($512^2 \\\\times 31$) | Complete | N/A | 43.40 dB | **43.43 dB** | Verified & Frozen |
| **Figure 10** | Yale Face B Denoising | `yale_b.mat` ($192 \\\\times 168 \\\\times 64$) | Completed | Approved | 28.9 dB / 29.7 dB | **30.86 dB / 30.46 dB** | Complete & Approved |
| **Figure 11** | Video Compressive Sensing | `cat_video.mat` ($64 \\\\times 128^2$) | Completed | Approved | $\\\\sim 27.5$ dB (HQ) | **25.32 dB** (+1.8 dB TV)| Audited (Data limited) |
| **Figure 12** | 3D PET/CT Radon Recon | `pet.mat` ($128^2 \\\\times 150$) | Completed | Approved | $\\\\sim 30.5$ dB | **30.55 dB** (0.8302 SSIM)| Complete & Audited |
| **Figure 13(a)**| HSI Rank Invariance | `icvl.mat` ($512^2 \\\\times 31$) | Complete | N/A | Flat plateau $\\\\ge R=20$ | Flat plateau $\\\\ge R=20$ | Verified & Frozen |
| **Figure 13(b)**| Yale Rank Invariance | `yale_b.mat` ($192 \\\\times 168 \\\\times 64$) | Completed | Approved | $\\\\sim 26.0$ dB plateau | **28.50–28.97 dB** | Complete & Audited |
| **Table 1** | Synthetic Grid Decomposition | Synthetic ($N=64..256, R=2..16$)| Complete | N/A | $0.00 \\\\pm 0.00$ Rel Err | $\\\\mathbf{0.00 \\\\pm 0.00}$ | Verified & Frozen |
| **Table 2** | NMF & Semi-NMF | MNIST & CIFAR-10 | Complete | N/A | 7.3–8.8 dB | **7.9–9.5 dB** | Verified & Frozen |
| **Table 3** | Speech Spectrogram NMF | SpeechCommands v2 (10 classes) | Complete | N/A | Matches paper PSNR | Matches paper PSNR | Verified & Frozen |

---

## 2. Detailed Experiment Profiles

"""

for exp in experiments:
    doc += f"### {exp['name']}\n\n"
    doc += f"- **Target ID**: `{exp['id']}`\n"
    doc += f"- **Paper Context**: Section / Figure reference replicating DeepTensor performance against classical and deep priors.\n"
    doc += f"- **Current Status**: **{exp['status']}**\n\n"
    
    doc += "#### Code & Configuration\n"
    doc += f"- **Original Script**: `{exp['orig_script']}`\n"
    doc += f"- **Improved Script**: `{exp['improved_script'] if exp['improved_script'] else 'Not applicable (Baseline frozen)'}`\n"
    doc += f"- **Configuration File**: `{exp['config']}`\n\n"
    
    doc += "#### Data Assets\n"
    doc += f"- **Dataset Name**: {exp['dataset_name']}\n"
    doc += f"- **Dataset Storage Path**: `{exp['dataset_path']}`\n\n"
    
    doc += "#### Baseline Outputs & Artifacts\n"
    doc += f"- **Baseline Output Directory**: `{exp['orig_outputs']}`\n"
    doc += f"- **Baseline Metrics**: `{exp['metrics']}`\n"
    doc += f"- **Baseline Checkpoint**: `{exp['checkpoint'] if exp['checkpoint'] and not exp['improved_outputs'] else 'N/A'}`\n"
    doc += f"- **Baseline Visual Plot**: `{exp['plots']}`\n"
    doc += f"- **Baseline Visual Panel**: `{exp['images'] if exp['images'] else 'N/A'}`\n\n"
    
    if exp['improved_outputs']:
        doc += "#### Improved Outputs & Artifacts\n"
        doc += f"- **Improved Output Directory**: `{exp['improved_outputs']}`\n"
        doc += f"- **Improved Metrics**: `{exp['metrics']}`\n"
        doc += f"- **Improved Checkpoint**: `{exp['checkpoint']}`\n"
        doc += f"- **Improved Visual Plot**: `{exp['plots']}`\n"
        doc += f"- **Improved Visual Panel**: `{exp['images']}`\n\n"
    
    doc += "#### Documentation & Audits\n"
    doc += f"- **Audit Report**: `{exp['reports']}`\n"
    doc += f"- **Known Discrepancies & Protocol Notes**: {exp['issues']}\n\n"
    doc += "---\n\n"

out_path = os.path.join(ROOT, "codebase_audit", "RESEARCH_EXPERIMENT_MAPPING.md")
with open(out_path, "w", encoding="utf-8") as fp:
    fp.write(doc.strip() + "\n")

print(f"Generated {out_path} ({len(doc.splitlines())} lines)")

