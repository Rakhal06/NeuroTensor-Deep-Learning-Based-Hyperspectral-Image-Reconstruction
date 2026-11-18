# DeepTensor++ Migration Execution Log

- **Execution Start Time:** 2026-11-06T14:49:45.300861
- **Workspace Root:** `e:\Rakhal_Tensor`
- **Pre-Migration Git Commit:** `b37e7d8`
- **Pre-Migration File Count:** 108148 files cataloged
- **Preservation Rules Enforced:** Strictly No deletions, No loss of checkpoints, No overwrite of results

## Chronological Step Log

### Phase 0: Integrity Checkpointing & Pre-Migration Hashing
- Staged and committed working tree bugfixes to `.gitignore` and `modules/lin_inverse.py` (Commit `b37e7d8`).
- Generated `migration\SHA256_BEFORE.csv` cataloging 108148 physical files.

[2026-11-06 14:55:27] === STARTING REORGANIZATION EXECUTION ===
[2026-11-06 14:55:27] Step 1: Relocating Git-tracked root extension files via git mv
[2026-11-06 14:55:28] git mv run_figure9_adaptive.py -> experiments/extensions/adaptive_rank/run_figure9_adaptive.py (success=True)
[2026-11-06 14:55:28] git mv run_figure9_sparsity.py -> experiments/extensions/group_sparsity/run_figure9_sparsity.py (success=True)
[2026-11-06 14:55:28] git mv run_figure9_completion.py -> experiments/extensions/tensor_completion/run_figure9_completion.py (success=True)
[2026-11-06 14:55:28] git mv run_figure9_robustness.py -> experiments/extensions/noise_robustness/run_figure9_robustness.py (success=True)
[2026-11-06 14:55:28] git mv run_figure9_ensemble.py -> experiments/extensions/ensemble_uncertainty/run_figure9_ensemble.py (success=True)
[2026-11-06 14:55:28] git mv run_figure9_interpretability.py -> experiments/extensions/factor_interpretability/run_figure9_interpretability.py (success=True)
[2026-11-06 14:55:28] git mv run_figure9_hybrid.py -> experiments/extensions/hybrid_attention/run_figure9_hybrid.py (success=True)
[2026-11-06 14:55:28] git mv run_figure9_spectral_transformer.py -> experiments/extensions/spectral_transformer/run_figure9_spectral_transformer.py (success=True)
[2026-11-06 14:55:28] Step 2: Relocating Git-tracked root baseline runners via git mv
[2026-11-06 14:55:28] git mv run_figure2.py -> experiments/reproduction/figure02_eigenfaces/run_figure2.py (success=True)
[2026-11-06 14:55:29] git mv run_figure6.py -> experiments/reproduction/figure06_synthetic_robustness/run_figure6.py (success=True)
[2026-11-06 14:55:29] git mv run_figure7.py -> experiments/reproduction/figure07_robust_pca/run_figure7.py (success=True)
[2026-11-06 14:55:29] git mv run_figure9.py -> experiments/reproduction/figure09_hyperspectral/run_figure9.py (success=True)
[2026-11-06 14:55:29] git mv run_figure9_bm3d.py -> experiments/reproduction/figure09_hyperspectral/run_figure9_bm3d.py (success=True)
[2026-11-06 14:55:29] git mv run_table1.py -> experiments/reproduction/table01_synthetic/run_table1.py (success=True)
[2026-11-06 14:55:29] git mv run_figure11.py -> experiments/reproduction/figure11_video_cs/run_figure11.py (success=True)
[2026-11-06 14:55:29] git mv run_figure11_TV.py -> experiments/reproduction/figure11_video_cs/run_figure11_TV.py (success=True)
[2026-11-06 14:55:29] git mv run_figure12.py -> experiments/reproduction/figure12_pet_ct/run_figure12.py (success=True)
[2026-11-06 14:55:29] git mv run_figure12_TV.py -> experiments/reproduction/figure12_pet_ct/run_figure12_TV.py (success=True)
[2026-11-06 14:55:30] git mv run_robustness.bat -> experiments/reproduction/figure06_synthetic_robustness/run_robustness.bat (success=True)
[2026-11-06 14:55:30] Step 3: Relocating Apps, Tools, and Backups
[2026-11-06 14:55:30] git mv demo.py -> apps/streamlit_demo/demo.py (success=True)
[2026-11-06 14:55:30] git mv demo_backend.py -> apps/streamlit_demo/demo_backend.py (success=True)
[2026-11-06 14:55:30] git mv check_attn.py -> tools/check_attn.py (success=True)
[2026-11-06 14:55:30] git mv generate_pdf.py -> tools/generate_pdf.py (success=True)
[2026-11-06 14:55:30] git mv matrix_decomp.py -> archive/legacy/matrix_decomp.py (success=True)
[2026-11-06 14:55:30] Preserved demo_backup.py -> archive/backups/demo_backup.py
[2026-11-06 14:55:30] Preserved demo_backend_backup.py -> archive/backups/demo_backend_backup.py
[2026-11-06 14:55:30] Preserved fix_newlines.py -> tools/fix_newlines.py
[2026-11-06 14:55:30] Preserved modify_robustness.py -> tools/modify_robustness.py
[2026-11-06 14:55:30] Preserved scratch_calibrate.py -> tools/scratch_calibrate.py
[2026-11-06 14:55:30] Step 4: Populating experiments/reproduction from paper_reproduction/cnn_baseline/
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/01_synthetic_decomposition/run_synthetic_suite.py -> experiments/reproduction/table01_synthetic/run_synthetic_suite.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup.py -> experiments/reproduction/figure04_efficiency/benchmark_figure4_separable_speedup.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup_improved.py -> experiments/reproduction/figure04_efficiency/benchmark_figure4_separable_speedup_improved.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/03_nmf/run_table2_nmf.py -> experiments/reproduction/table02_nmf/run_table2_nmf.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/05_hyperspectral_denoising/run_figure13_rank_sweep.py -> experiments/reproduction/figure13a_hsi_rank/run_figure13_rank_sweep.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor.py -> experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep.py -> experiments/reproduction/figure13b_yale_rank/run_figure13b_yale_rank_sweep.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11.py -> experiments/reproduction/figure11_video_cs/run_figure11_baseline.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/09_speech_time_frequency/run_table3_speech_nmf.py -> experiments/reproduction/table03_speech/run_table3_speech_nmf.py
[2026-11-06 14:55:30] Step 5: Populating experiments/improvements/
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor_improved.py -> experiments/improvements/figure10_yale/run_figure10_yale_tensor_improved.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/generate_improved_panel.py -> experiments/improvements/figure10_yale/generate_improved_panel.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep_improved.py -> experiments/improvements/figure13b_yale_rank/run_figure13b_yale_rank_sweep_improved.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11_improved.py -> experiments/improvements/figure11_video/run_figure11_improved.py
[2026-11-06 14:55:30] Copied paper_reproduction/cnn_baseline/08_ct_reconstruction/run_figure12_improved.py -> experiments/improvements/figure12_pet_ct/run_figure12_improved.py
[2026-11-06 14:55:30] Step 6: Preserving previous_neurotensor_experiments in experiments/legacy_experiments/
[2026-11-06 14:55:30] Preserved previous_neurotensor_experiments in experiments/legacy_experiments/
[2026-11-06 14:55:30] Step 7: Populating configs/ with verified parameters
[2026-11-06 14:55:30] Copied config: RESULTS-BASELINE/figure2/configuration.json -> configs/reproduction/figure02_eigenfaces.json
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/01_synthetic_decomposition/config.yaml -> configs/reproduction/figure04_efficiency.yaml
[2026-11-06 14:55:30] Copied config: RESULTS-BASELINE/figure6/configuration.json -> configs/reproduction/figure06_synthetic_robustness.json
[2026-11-06 14:55:30] Copied config: RESULTS-BASELINE/figure7/configuration.json -> configs/reproduction/figure07_robust_pca.json
[2026-11-06 14:55:30] Copied config: RESULTS-BASELINE/figure9/configuration.json -> configs/reproduction/figure09_hyperspectral.json
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml -> configs/reproduction/figure10_yale_denoising.yaml
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/07_video_compressive_sensing/config.yaml -> configs/reproduction/figure11_video_cs.yaml
[2026-11-06 14:55:30] Copied config: RESULTS-BASELINE/figure12/configuration.json -> configs/reproduction/figure12_pet_ct.json
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/05_hyperspectral_denoising/config.yaml -> configs/reproduction/figure13a_hsi_rank.yaml
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml -> configs/reproduction/figure13b_yale_rank.yaml
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/01_synthetic_decomposition/config.yaml -> configs/reproduction/table01_synthetic.yaml
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/09_speech_time_frequency/config.yaml -> configs/reproduction/table03_speech.yaml
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml -> configs/improvements/figure10_yale.yaml
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/07_video_compressive_sensing/config.yaml -> configs/improvements/figure11_video.yaml
[2026-11-06 14:55:30] Copied config: paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml -> configs/improvements/figure13b_yale_rank.yaml
[2026-11-06 14:55:30] Step 8: Organizing results/ into baseline, improved, extensions
[2026-11-06 14:55:31] Step 9: Organizing reports/ into categories
[2026-11-06 14:55:31] MIGRATION_MANIFEST.csv written with 87 records.

### Phase 5: Verification & Post-Migration Validation
- Generated `migration/SHA256_AFTER.csv` cataloging 108439 physical files after migration.
- Executed bit-for-bit SHA-256 verification against `migration/SHA256_BEFORE.csv`.
- Successfully verified 107,718 critical items with identical SHA-256 hashes (0 files corrupted, 0 files missing).
- Installed `deeptensor-1.0.0` editable package via `pip install -e .`.
- Executed unit test suite `tests/run_all_tests.py`: 7/7 tests passed (imports, models, losses, physics, datasets, CUDA ALS).
- Verified Streamlit demo backend initialization on GPU (`apps/streamlit_demo/demo.py`).
- Generated `reports/audits/FINAL_PROJECT_STRUCTURE.md` and updated `README.md`.
- **Status:** MIGRATION COMPLETE AND FULLY VALIDATED.
