import os
import shutil
import subprocess
import csv
import datetime

ROOT = os.path.abspath('e:/Rakhal_Tensor')
MIGRATION_DIR = os.path.join(ROOT, 'migration')
LOG_FILE = os.path.join(MIGRATION_DIR, 'MIGRATION_LOG.md')
MANIFEST_FILE = os.path.join(MIGRATION_DIR, 'MIGRATION_MANIFEST.csv')

def log_step(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")

manifest_entries = []

def record_manifest(orig_path, new_path, category, action, notes=""):
    manifest_entries.append({
        'original_path': orig_path.replace('\\', '/'),
        'new_path': new_path.replace('\\', '/') if new_path else 'N/A',
        'category': category,
        'action': action,
        'notes': notes
    })

def git_mv(src, dst):
    src_abs = os.path.join(ROOT, src)
    dst_abs = os.path.join(ROOT, dst)
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    res = subprocess.run(['git', 'mv', src, dst], cwd=ROOT, capture_output=True, text=True)
    if res.returncode == 0:
        record_manifest(src, dst, 'Git-Tracked File', 'git mv', 'Relocated with git history preserved')
        return True
    else:
        # Fallback to copy/move
        shutil.copy2(src_abs, dst_abs)
        record_manifest(src, dst, 'Git-Tracked File (Fallback)', 'shutil.copy2', res.stderr.strip())
        return False

def copy_or_move(src, dst, category, action="copy"):
    src_abs = os.path.join(ROOT, src)
    dst_abs = os.path.join(ROOT, dst)
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    if os.path.isdir(src_abs):
        if action == "move":
            shutil.move(src_abs, dst_abs)
        else:
            shutil.copytree(src_abs, dst_abs, dirs_exist_ok=True)
    elif os.path.isfile(src_abs):
        if action == "move":
            shutil.move(src_abs, dst_abs)
        else:
            shutil.copy2(src_abs, dst_abs)
    record_manifest(src, dst, category, action, "Preserved without modification")

print("Starting Project Reorganization Execution...")
log_step("=== STARTING REORGANIZATION EXECUTION ===")

# --- Step 1: Relocate Git-Tracked Root Extension Files via git mv ---
log_step("Step 1: Relocating Git-tracked root extension files via git mv")
extensions_map = [
    ('run_figure9_adaptive.py', 'experiments/extensions/adaptive_rank/run_figure9_adaptive.py'),
    ('run_figure9_sparsity.py', 'experiments/extensions/group_sparsity/run_figure9_sparsity.py'),
    ('run_figure9_completion.py', 'experiments/extensions/tensor_completion/run_figure9_completion.py'),
    ('run_figure9_robustness.py', 'experiments/extensions/noise_robustness/run_figure9_robustness.py'),
    ('run_figure9_ensemble.py', 'experiments/extensions/ensemble_uncertainty/run_figure9_ensemble.py'),
    ('run_figure9_interpretability.py', 'experiments/extensions/factor_interpretability/run_figure9_interpretability.py'),
    ('run_figure9_hybrid.py', 'experiments/extensions/hybrid_attention/run_figure9_hybrid.py'),
    ('run_figure9_spectral_transformer.py', 'experiments/extensions/spectral_transformer/run_figure9_spectral_transformer.py'),
]
for src, dst in extensions_map:
    if os.path.exists(os.path.join(ROOT, src)):
        ok = git_mv(src, dst)
        log_step(f"git mv {src} -> {dst} (success={ok})")

# --- Step 2: Relocate Git-Tracked Root Baseline Runners ---
log_step("Step 2: Relocating Git-tracked root baseline runners via git mv")
root_runners_map = [
    ('run_figure2.py', 'experiments/reproduction/figure02_eigenfaces/run_figure2.py'),
    ('run_figure6.py', 'experiments/reproduction/figure06_synthetic_robustness/run_figure6.py'),
    ('run_figure7.py', 'experiments/reproduction/figure07_robust_pca/run_figure7.py'),
    ('run_figure9.py', 'experiments/reproduction/figure09_hyperspectral/run_figure9.py'),
    ('run_figure9_bm3d.py', 'experiments/reproduction/figure09_hyperspectral/run_figure9_bm3d.py'),
    ('run_table1.py', 'experiments/reproduction/table01_synthetic/run_table1.py'),
    ('run_figure11.py', 'experiments/reproduction/figure11_video_cs/run_figure11.py'),
    ('run_figure11_TV.py', 'experiments/reproduction/figure11_video_cs/run_figure11_TV.py'),
    ('run_figure12.py', 'experiments/reproduction/figure12_pet_ct/run_figure12.py'),
    ('run_figure12_TV.py', 'experiments/reproduction/figure12_pet_ct/run_figure12_TV.py'),
    ('run_robustness.bat', 'experiments/reproduction/figure06_synthetic_robustness/run_robustness.bat'),
]
for src, dst in root_runners_map:
    if os.path.exists(os.path.join(ROOT, src)):
        ok = git_mv(src, dst)
        log_step(f"git mv {src} -> {dst} (success={ok})")

# --- Step 3: Relocate Demos, Tools, Backups via git mv / copy ---
log_step("Step 3: Relocating Apps, Tools, and Backups")
apps_and_tools = [
    ('demo.py', 'apps/streamlit_demo/demo.py'),
    ('demo_backend.py', 'apps/streamlit_demo/demo_backend.py'),
    ('check_attn.py', 'tools/check_attn.py'),
    ('generate_pdf.py', 'tools/generate_pdf.py'),
    ('matrix_decomp.py', 'archive/legacy/matrix_decomp.py'),
]
for src, dst in apps_and_tools:
    if os.path.exists(os.path.join(ROOT, src)):
        ok = git_mv(src, dst)
        log_step(f"git mv {src} -> {dst} (success={ok})")

# Untracked backups & diagnostics
untracked_tools = [
    ('demo_backup.py', 'archive/backups/demo_backup.py'),
    ('demo_backend_backup.py', 'archive/backups/demo_backend_backup.py'),
    ('fix_newlines.py', 'tools/fix_newlines.py'),
    ('modify_robustness.py', 'tools/modify_robustness.py'),
    ('scratch_calibrate.py', 'tools/scratch_calibrate.py'),
]
for src, dst in untracked_tools:
    if os.path.exists(os.path.join(ROOT, src)):
        copy_or_move(src, dst, 'Backup/Tool', action="copy")
        log_step(f"Preserved {src} -> {dst}")

# --- Step 4: Populate experiments/reproduction from paper_reproduction/cnn_baseline/ ---
log_step("Step 4: Populating experiments/reproduction from paper_reproduction/cnn_baseline/")
reproduction_map = [
    ('paper_reproduction/cnn_baseline/01_synthetic_decomposition/run_synthetic_suite.py', 'experiments/reproduction/table01_synthetic/run_synthetic_suite.py'),
    ('paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup.py', 'experiments/reproduction/figure04_efficiency/benchmark_figure4_separable_speedup.py'),
    ('paper_reproduction/cnn_baseline/01_synthetic_decomposition/benchmark_figure4_separable_speedup_improved.py', 'experiments/reproduction/figure04_efficiency/benchmark_figure4_separable_speedup_improved.py'),
    ('paper_reproduction/cnn_baseline/03_nmf/run_table2_nmf.py', 'experiments/reproduction/table02_nmf/run_table2_nmf.py'),
    ('paper_reproduction/cnn_baseline/05_hyperspectral_denoising/run_figure13_rank_sweep.py', 'experiments/reproduction/figure13a_hsi_rank/run_figure13_rank_sweep.py'),
    ('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor.py', 'experiments/reproduction/figure10_yale_denoising/run_figure10_yale_tensor.py'),
    ('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep.py', 'experiments/reproduction/figure13b_yale_rank/run_figure13b_yale_rank_sweep.py'),
    ('paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11.py', 'experiments/reproduction/figure11_video_cs/run_figure11_baseline.py'),
    ('paper_reproduction/cnn_baseline/09_speech_time_frequency/run_table3_speech_nmf.py', 'experiments/reproduction/table03_speech/run_table3_speech_nmf.py'),
]
for src, dst in reproduction_map:
    if os.path.exists(os.path.join(ROOT, src)):
        copy_or_move(src, dst, 'Reproduction Runner', action="copy")
        log_step(f"Copied {src} -> {dst}")

# --- Step 5: Populate experiments/improvements/ ---
log_step("Step 5: Populating experiments/improvements/")
improvements_map = [
    ('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure10_yale_tensor_improved.py', 'experiments/improvements/figure10_yale/run_figure10_yale_tensor_improved.py'),
    ('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/generate_improved_panel.py', 'experiments/improvements/figure10_yale/generate_improved_panel.py'),
    ('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/run_figure13b_yale_rank_sweep_improved.py', 'experiments/improvements/figure13b_yale_rank/run_figure13b_yale_rank_sweep_improved.py'),
    ('paper_reproduction/cnn_baseline/07_video_compressive_sensing/run_figure11_improved.py', 'experiments/improvements/figure11_video/run_figure11_improved.py'),
    ('paper_reproduction/cnn_baseline/08_ct_reconstruction/run_figure12_improved.py', 'experiments/improvements/figure12_pet_ct/run_figure12_improved.py'),
]
for src, dst in improvements_map:
    if os.path.exists(os.path.join(ROOT, src)):
        copy_or_move(src, dst, 'Improved Runner', action="copy")
        log_step(f"Copied {src} -> {dst}")

# --- Step 6: Preserve previous_neurotensor_experiments ---
log_step("Step 6: Preserving previous_neurotensor_experiments in experiments/legacy_experiments/")
copy_or_move('previous_neurotensor_experiments', 'experiments/legacy_experiments/previous_neurotensor_experiments', 'Legacy Extensions Archive', action="copy")
log_step("Preserved previous_neurotensor_experiments in experiments/legacy_experiments/")

# --- Step 7: Populate configs/ ---
log_step("Step 7: Populating configs/ with verified parameters")
configs_map = [
    ('RESULTS-BASELINE/figure2/configuration.json', 'configs/reproduction/figure02_eigenfaces.json'),
    ('paper_reproduction/cnn_baseline/01_synthetic_decomposition/config.yaml', 'configs/reproduction/figure04_efficiency.yaml'),
    ('RESULTS-BASELINE/figure6/configuration.json', 'configs/reproduction/figure06_synthetic_robustness.json'),
    ('RESULTS-BASELINE/figure7/configuration.json', 'configs/reproduction/figure07_robust_pca.json'),
    ('RESULTS-BASELINE/figure9/configuration.json', 'configs/reproduction/figure09_hyperspectral.json'),
    ('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml', 'configs/reproduction/figure10_yale_denoising.yaml'),
    ('paper_reproduction/cnn_baseline/07_video_compressive_sensing/config.yaml', 'configs/reproduction/figure11_video_cs.yaml'),
    ('RESULTS-BASELINE/figure12/configuration.json', 'configs/reproduction/figure12_pet_ct.json'),
    ('paper_reproduction/cnn_baseline/05_hyperspectral_denoising/config.yaml', 'configs/reproduction/figure13a_hsi_rank.yaml'),
    ('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml', 'configs/reproduction/figure13b_yale_rank.yaml'),
    ('paper_reproduction/cnn_baseline/01_synthetic_decomposition/config.yaml', 'configs/reproduction/table01_synthetic.yaml'),
    ('paper_reproduction/cnn_baseline/03_nmf/config.yaml', 'configs/reproduction/table02_nmf.yaml'),
    ('paper_reproduction/cnn_baseline/09_speech_time_frequency/config.yaml', 'configs/reproduction/table03_speech.yaml'),
    ('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml', 'configs/improvements/figure10_yale.yaml'),
    ('paper_reproduction/cnn_baseline/07_video_compressive_sensing/config.yaml', 'configs/improvements/figure11_video.yaml'),
    ('paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run/configuration.json', 'configs/improvements/figure12_pet_ct.json'),
    ('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/config.yaml', 'configs/improvements/figure13b_yale_rank.yaml'),
]
for src, dst in configs_map:
    if os.path.exists(os.path.join(ROOT, src)):
        copy_or_move(src, dst, 'Configuration', action="copy")
        log_step(f"Copied config: {src} -> {dst}")

# --- Step 8: Organize results/ into baseline, improved, extensions ---
log_step("Step 8: Organizing results/ into baseline, improved, extensions")

# Baseline results
copy_or_move('RESULTS-BASELINE/figure2', 'results/baseline/figure02_eigenfaces', 'Baseline Results', action="copy")
copy_or_move('RESULTS-BASELINE/figure6', 'results/baseline/figure06_synthetic_robustness', 'Baseline Results', action="copy")
copy_or_move('RESULTS-BASELINE/figure7', 'results/baseline/figure07_robust_pca', 'Baseline Results', action="copy")
copy_or_move('RESULTS-BASELINE/figure9', 'results/baseline/figure09_hyperspectral', 'Baseline Results', action="copy")
copy_or_move('RESULTS-BASELINE/figure12', 'results/baseline/figure12_pet_ct', 'Baseline Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/01_synthetic_decomposition/synthetic_suite_metrics.json', 'results/baseline/table01_synthetic/synthetic_suite_metrics.json', 'Baseline Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/03_nmf/metrics.json', 'results/baseline/table02_nmf/metrics.json', 'Baseline Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/05_hyperspectral_denoising/figure13a_metrics.json', 'results/baseline/figure13a_hsi_rank/figure13a_metrics.json', 'Baseline Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/original_baseline', 'results/baseline/figure10_yale_denoising', 'Baseline Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/original_baseline/figure13b_metrics.json', 'results/baseline/figure13b_yale_rank/figure13b_metrics.json', 'Baseline Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/07_video_compressive_sensing/original_baseline', 'results/baseline/figure11_video_cs', 'Baseline Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/09_speech_time_frequency/metrics.json', 'results/baseline/table03_speech/metrics.json', 'Baseline Results', action="copy")

# Improved results
copy_or_move('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run', 'results/improved/figure10_yale_denoising', 'Improved Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/improved_run_fig13b', 'results/improved/figure13b_yale_rank', 'Improved Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/07_video_compressive_sensing/improved_run', 'results/improved/figure11_video_cs', 'Improved Results', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/08_ct_reconstruction/improved_run', 'results/improved/figure12_pet_ct', 'Improved Results', action="copy")

# --- Step 9: Organize reports/ into reproduction, improvements, extensions, audits ---
log_step("Step 9: Organizing reports/ into categories")

# Audits
copy_or_move('codebase_audit', 'reports/audits/codebase_audit', 'Audit Reports', action="copy")
copy_or_move('DATA_DIMENSION_TRACE.md', 'reports/audits/DATA_DIMENSION_TRACE.md', 'Audit Reports', action="copy")
copy_or_move('PROJECT_CONTEXT.md', 'reports/audits/PROJECT_CONTEXT.md', 'Audit Reports', action="copy")
copy_or_move('REPRODUCIBILITY.md', 'reports/audits/REPRODUCIBILITY.md', 'Audit Reports', action="copy")

# Improvements
copy_or_move('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/FIGURE10_IMPROVEMENT_REPORT.md', 'reports/improvements/FIGURE10_IMPROVEMENT_REPORT.md', 'Improvement Reports', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/06_yale_face_tensor_denoising/FIGURE13B_IMPROVEMENT_REPORT.md', 'reports/improvements/FIGURE13B_IMPROVEMENT_REPORT.md', 'Improvement Reports', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/07_video_compressive_sensing/FIGURE11_IMPROVEMENT_REPORT.md', 'reports/improvements/FIGURE11_IMPROVEMENT_REPORT.md', 'Improvement Reports', action="copy")
copy_or_move('paper_reproduction/cnn_baseline/08_ct_reconstruction/FIGURE12_IMPROVEMENT_REPORT.md', 'reports/improvements/FIGURE12_IMPROVEMENT_REPORT.md', 'Improvement Reports', action="copy")
copy_or_move('PHASE1_PERFORMANCE_GAP_AUDIT.md', 'reports/improvements/PHASE1_PERFORMANCE_GAP_AUDIT.md', 'Improvement Reports', action="copy")

# Reproduction
copy_or_move('paper_reproduction/consolidated_results/CONSOLIDATED_CNN_BASELINE_REPORT.md', 'reports/reproduction/CONSOLIDATED_CNN_BASELINE_REPORT.md', 'Reproduction Reports', action="copy")
copy_or_move('PAPER_RESULTS', 'reports/reproduction/gallery_paper_results', 'Publication Gallery', action="copy")

# Extensions
copy_or_move('documentation', 'reports/extensions/documentation', 'Extension Documentation', action="copy")

# Write MIGRATION_MANIFEST.csv
with open(MANIFEST_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['original_path', 'new_path', 'category', 'action', 'notes'])
    writer.writeheader()
    writer.writerows(manifest_entries)
log_step(f"MIGRATION_MANIFEST.csv written with {len(manifest_entries)} records.")

print("Reorganization file relocation completed successfully.")

