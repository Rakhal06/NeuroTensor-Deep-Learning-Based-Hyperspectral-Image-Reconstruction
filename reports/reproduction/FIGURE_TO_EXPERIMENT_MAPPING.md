# DeepTensor Figure to Experiment Mapping

This document provides an explicit mapping between every Figure/Table in the paper and our 9 application reproduction folders.

---

## Mapping Matrix

| Paper Figure / Table | Paper Title / Description | Section in Paper | Target Application Folder | Primary Script / Runner |
|---|---|---|---|---|
| **Figure 1** | Conceptual SVD vs DeepTensor | Sec 1 | Conceptual Architecture | `modules/deep_prior.py` |
| **Figure 2** | DeepTensor Eigenfaces & SVM Accuracy | Sec 1 & 4.2 | `04_eigenfaces_classification/` | `run_figure2.py` |
| **Figure 3** | Inductive Bias Rejects Noise | Sec 3.1 | `01_synthetic_decomposition/` | `run_figure3_inductive_bias.py` |
| **Figure 4** | Separable Convolutions Speedup | Sec 3.2 | `01_synthetic_decomposition/` | `run_figure4_separable_speedup.py` |
| **Figure 5** | DeepTensor DIP Architectures | Sec 3.3 | Architectural Reference | `modules/deep_prior.py` |
| **Figure 6** | Signal & Noise Distribution Robustness | Sec 3.2 | `01_synthetic_decomposition/` | `run_figure6.py` |
| **Figure 7** | Robust PCA under Low SNR & Sample Size | Sec 3.2 | `02_pca/` | `run_figure7.py` |
| **Figure 8** | Stopping Criterion: DIP vs Deep Decoder | Sec 3.4 | `01_synthetic_decomposition/` | `run_figure8_stopping_criterion.py` |
| **Figure 9** | Hyperspectral Image Denoising (ICVL) | Sec 4.1 | `05_hyperspectral_denoising/` | `run_figure9.py` |
| **Figure 10** | Tensor Denoising (Yale Face B) | Sec 4.1 | `06_yale_face_tensor_denoising/` | `run_figure10_yale_tensor.py` |
| **Figure 11** | Video Compressive Sensing (Cat Video) | Sec 4.3 | `07_video_compressive_sensing/` | `run_figure11.py` |
| **Figure 12** | 3D CT Reconstruction (PET Volume) | Sec 4.1 | `08_ct_reconstruction/` | `run_figure12.py` |
| **Figure 13a**| Effect of Rank Choice: Matrix (ICVL) | Sec 4.4 | `05_hyperspectral_denoising/` | `run_figure13_rank_sweep.py` |
| **Figure 13b**| Effect of Rank Choice: Tensor (Yale-B) | Sec 4.4 | `06_yale_face_tensor_denoising/` | `run_figure13b_yale_rank_sweep.py` |
| **Table 1** | Learning Rate Schedulers on Toy Matrix | Sec 3.4 | `01_synthetic_decomposition/` | `run_table1_lr_schedulers.py` |
| **Table 2** | NMF on MNIST & CIFAR-10 | Sec 3.3 | `03_nmf/` | `run_table2_nmf_mnist_cifar.py` |
| **Table 3** | NMF on Speech Spectrograms | Sec 4.3 | `09_speech_time_frequency/` | `run_table3_speech_nmf.py` |

