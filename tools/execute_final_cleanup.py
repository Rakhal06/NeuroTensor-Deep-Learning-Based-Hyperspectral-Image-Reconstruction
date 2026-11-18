"""
DeepTensor++ Final Root Cleanup & Consolidation Script
Exhaustively consolidates all legacy and root files into canonical architecture.
Outputs migration/CLEANUP_MANIFEST.csv
"""

import os
import sys
import shutil
import hashlib
import subprocess
import csv

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.chdir(WORKSPACE_ROOT)

MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, 'migration', 'CLEANUP_MANIFEST.csv')
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

def hash_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def is_git_tracked(p):
    p_norm = p.replace('\\', '/')
    res = subprocess.run(['git', 'ls-files', '--error-unmatch', p_norm],
                         capture_output=True, text=True, cwd=WORKSPACE_ROOT)
    return res.returncode == 0

manifest_records = []

def record_action(orig_path, new_path, category, action, sha256):
    manifest_records.append({
        'original_path': orig_path.replace('\\', '/'),
        'new_path': new_path.replace('\\', '/'),
        'category': category,
        'action': action,
        'sha256': sha256
    })

def safe_file_move(src, dst, category):
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    
    if not os.path.exists(src_abs):
        print(f"[SKIP] Source does not exist: {src}")
        return
        
    sha = hash_file(src_abs)
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    
    src_tracked = is_git_tracked(src)
    
    if os.path.exists(dst_abs):
        dst_sha = hash_file(dst_abs)
        if sha == dst_sha:
            if src_tracked:
                # Remove untracked dst and git mv src dst
                os.remove(dst_abs)
                cmd = ['git', 'mv', '-f', src.replace('\\', '/'), dst.replace('\\', '/')]
                res = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_ROOT)
                if res.returncode != 0:
                    raise RuntimeError(f"git mv failed: {cmd} -> {res.stderr}")
                record_action(src, dst, category, 'git_mv_dedup', sha)
            else:
                # Untracked src is identical to existing dst
                os.remove(src_abs)
                record_action(src, dst, category, 'verified_identical_deduplicated', sha)
            return
        else:
            raise RuntimeError(f"Collision with differing hash: {src} vs {dst} ({sha} vs {dst_sha})")
            
    # Destination does not exist
    if src_tracked:
        cmd = ['git', 'mv', '-f', src.replace('\\', '/'), dst.replace('\\', '/')]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_ROOT)
        if res.returncode != 0:
            raise RuntimeError(f"git mv failed: {cmd} -> {res.stderr}")
        record_action(src, dst, category, 'git_mv', sha)
    else:
        shutil.move(src_abs, dst_abs)
        record_action(src, dst, category, 'moved', sha)
        
    # Verify dst
    assert os.path.exists(dst_abs), f"Dst missing after move: {dst}"
    assert hash_file(dst_abs) == sha, f"Hash mismatch after move: {dst}"

def move_tree_contents(src_dir, dst_dir, category):
    src_dir_abs = os.path.abspath(src_dir)
    dst_dir_abs = os.path.abspath(dst_dir)
    if not os.path.exists(src_dir_abs):
        return
    for root, dirs, files in os.walk(src_dir_abs):
        for f in files:
            src_file = os.path.join(root, f)
            rel = os.path.relpath(src_file, src_dir_abs)
            dst_file = os.path.join(dst_dir_abs, rel)
            safe_file_move(os.path.relpath(src_file, WORKSPACE_ROOT),
                           os.path.relpath(dst_file, WORKSPACE_ROOT),
                           category)
    # Remove empty src_dir tree
    try:
        shutil.rmtree(src_dir_abs)
    except Exception:
        pass

def main():
    print("Starting Final Root Cleanup and Consolidation...")
    
    # 0. Clean pycache before moves
    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        for d in dirs:
            if d == '__pycache__':
                pyc_dir = os.path.join(root, d)
                try:
                    shutil.rmtree(pyc_dir)
                except Exception:
                    pass

    # 1. Root Research Documentation (Markdown, CSV, PDF, PPTX)
    root_docs_mapping = {
        'DATA_DIMENSION_TRACE.md': ('reports/audits/DATA_DIMENSION_TRACE.md', 'reports_audits'),
        'DEMO_IMPLEMENTATION_PLAN.md': ('reports/audits/DEMO_IMPLEMENTATION_PLAN.md', 'reports_audits'),
        'IMPLEMENTATION_DOCUMENTATION.md': ('reports/audits/IMPLEMENTATION_DOCUMENTATION.md', 'reports_audits'),
        'PROJECT_CONTEXT.md': ('reports/audits/PROJECT_CONTEXT.md', 'reports_audits'),
        'DATASET_REGISTER.md': ('reports/audits/DATASET_REGISTER.md', 'reports_audits'),
        'EXPERIMENT_TRACKER.md': ('reports/audits/EXPERIMENT_TRACKER.md', 'reports_audits'),
        'FIGURE_TRACKER.md': ('reports/audits/FIGURE_TRACKER.md', 'reports_audits'),
        'IMPLEMENTATION_STATUS.md': ('reports/audits/IMPLEMENTATION_STATUS.md', 'reports_audits'),
        'PHASE1_EXPERIMENT_CONFIG_AUDIT.csv': ('reports/audits/PHASE1_EXPERIMENT_CONFIG_AUDIT.csv', 'reports_audits'),
        'PROJECT_ROADMAP.md': ('reports/audits/PROJECT_ROADMAP.md', 'reports_audits'),
        'RESEARCH_NOTES.md': ('reports/audits/RESEARCH_NOTES.md', 'reports_audits'),
        'DeepTensor_Implementation_Documentation.pdf': ('reports/reproduction/papers_and_slides/DeepTensor_Implementation_Documentation.pdf', 'papers_and_slides'),
        'DeepTensor_Paper_2204.03145.pdf': ('reports/reproduction/papers_and_slides/DeepTensor_Paper_2204.03145.pdf', 'papers_and_slides'),
        'Enhanced_DeepTensor_Phase1_Detailed_Presentation-1.pptx': ('reports/reproduction/papers_and_slides/Enhanced_DeepTensor_Phase1_Detailed_Presentation-1.pptx', 'papers_and_slides'),
        'PAPER_EXPERIMENT_REGISTER.md': ('reports/reproduction/PAPER_EXPERIMENT_REGISTER.md', 'reports_reproduction'),
        'REPRODUCIBILITY.md': ('reports/reproduction/REPRODUCIBILITY.md', 'reports_reproduction'),
        'PHASE1_IMPROVEMENT_PLAN.md': ('reports/improvements/PHASE1_IMPROVEMENT_PLAN.md', 'reports_improvements'),
        'PHASE1_PERFORMANCE_GAP_AUDIT.md': ('reports/improvements/PHASE1_PERFORMANCE_GAP_AUDIT.md', 'reports_improvements'),
        'PHASE1_RERUN_PRIORITY.csv': ('reports/improvements/PHASE1_RERUN_PRIORITY.csv', 'reports_improvements'),
    }
    for src, (dst, cat) in root_docs_mapping.items():
        if os.path.exists(src):
            safe_file_move(src, dst, cat)

    # 2. Root Backup & Maintenance Files
    root_misc_mapping = {
        'demo.py.bak': ('archive/backups/demo.py.bak', 'archive_backups'),
        'demo_backend.py.bak': ('archive/backups/demo_backend.py.bak', 'archive_backups'),
        'demo_backend_backup.py': ('archive/backups/demo_backend_backup.py', 'archive_backups'),
        'demo_backup.py': ('archive/backups/demo_backup.py', 'archive_backups'),
        'fix_newlines.py': ('tools/fix_newlines.py', 'tools'),
        'modify_robustness.py': ('archive/legacy/modify_robustness.py', 'archive_legacy'),
        'scratch_calibrate.py': ('archive/legacy/scratch_calibrate.py', 'archive_legacy'),
        'scratch_exp_mapping.json': ('archive/legacy/scratch_exp_mapping.json', 'archive_legacy'),
        'scratch_py_inventory.json': ('archive/legacy/scratch_py_inventory.json', 'archive_legacy'),
        'paper_extracted_text.txt': ('archive/legacy/paper_extracted_text.txt', 'archive_legacy'),
        'user_prompt.txt': ('archive/legacy/user_prompt.txt', 'archive_legacy'),
    }
    for src, (dst, cat) in root_misc_mapping.items():
        if os.path.exists(src):
            safe_file_move(src, dst, cat)

    # 3. Root Lambda Logs
    for lambda_file in ['results_lambda_0.txt', 'results_lambda_0.0001.txt', 'results_lambda_0.001.txt', 'results_lambda_0.01.txt']:
        if os.path.exists(lambda_file):
            dst = os.path.join('results/extensions/sparsity/logs', lambda_file)
            safe_file_move(lambda_file, dst, 'results_extensions_sparsity')

    # 4. scratch/ directory -> archive/legacy/scratch_audit/
    if os.path.exists('scratch'):
        move_tree_contents('scratch', 'archive/legacy/scratch_audit', 'archive_legacy_scratch')

    # 5. codebase_audit/ -> deduplicate against reports/audits/codebase_audit/
    if os.path.exists('codebase_audit'):
        move_tree_contents('codebase_audit', 'reports/audits/codebase_audit', 'reports_audits_codebase')

    # 6. previous_neurotensor_experiments/ -> deduplicate against experiments/legacy_experiments/previous_neurotensor_experiments/
    if os.path.exists('previous_neurotensor_experiments'):
        move_tree_contents('previous_neurotensor_experiments',
                           'experiments/legacy_experiments/previous_neurotensor_experiments',
                           'legacy_neurotensor')

    # 7. PAPER_RESULTS/ -> deduplicate against reports/reproduction/gallery_paper_results/
    if os.path.exists('PAPER_RESULTS'):
        move_tree_contents('PAPER_RESULTS', 'reports/reproduction/gallery_paper_results', 'gallery_paper_results')

    # 8. paper_reproduction/ -> copy key reports then archive to archive/legacy/paper_reproduction_legacy/
    if os.path.exists('paper_reproduction'):
        key_reports = {
            'paper_reproduction/consolidated_results/CNN_BASELINE_CONSOLIDATED_REPORT.md': 'reports/reproduction/CNN_BASELINE_CONSOLIDATED_REPORT.md',
            'paper_reproduction/datasets/DATASET_AUDIT.md': 'reports/audits/DATASET_AUDIT.md',
            'paper_reproduction/original_paper/EXPERIMENT_SETTINGS.md': 'reports/reproduction/EXPERIMENT_SETTINGS.md',
            'paper_reproduction/original_paper/FIGURE_TO_EXPERIMENT_MAPPING.md': 'reports/reproduction/FIGURE_TO_EXPERIMENT_MAPPING.md',
            'paper_reproduction/original_paper/REPORTED_METRICS.md': 'reports/reproduction/REPORTED_METRICS.md',
        }
        for s_rep, d_rep in key_reports.items():
            if os.path.exists(s_rep):
                os.makedirs(os.path.dirname(d_rep), exist_ok=True)
                shutil.copy2(s_rep, d_rep)
                record_action(s_rep, d_rep, 'reports_reproduction', 'copied_report', hash_file(d_rep))
        move_tree_contents('paper_reproduction', 'archive/legacy/paper_reproduction_legacy', 'paper_reproduction_legacy')

    # 9. documentation/ (tracked in git) -> reports/extensions/documentation/
    if os.path.exists('documentation'):
        # If reports/extensions/documentation exists untracked, remove it first so git mv documentation works cleanly
        untracked_doc_copy = os.path.join(WORKSPACE_ROOT, 'reports', 'extensions', 'documentation')
        if os.path.exists(untracked_doc_copy):
            shutil.rmtree(untracked_doc_copy)
        cmd = ['git', 'mv', 'documentation', 'reports/extensions/documentation']
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_ROOT)
        if res.returncode != 0:
            raise RuntimeError(f"git mv documentation failed: {res.stderr}")
        for r, _, fs in os.walk('reports/extensions/documentation'):
            for f in fs:
                p = os.path.join(r, f)
                orig = p.replace('reports/extensions/documentation', 'documentation')
                record_action(orig, p, 'documentation_extensions', 'git_mv', hash_file(p))

    # 10. modules/ (tracked in git) -> archive/legacy/modules_legacy/
    if os.path.exists('modules'):
        cmd = ['git', 'mv', 'modules', 'archive/legacy/modules_legacy']
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_ROOT)
        if res.returncode != 0:
            raise RuntimeError(f"git mv modules failed: {res.stderr}")
        for r, _, fs in os.walk('archive/legacy/modules_legacy'):
            for f in fs:
                p = os.path.join(r, f)
                orig = p.replace('archive/legacy/modules_legacy', 'modules')
                record_action(orig, p, 'modules_legacy', 'git_mv', hash_file(p))

    # 11. RESULTS-BASELINE/ (tracked in git) -> results/baseline/
    if os.path.exists('RESULTS-BASELINE'):
        baseline_fig_map = {
            'figure2': 'figure02_eigenfaces',
            'figure6': 'figure06_synthetic_robustness',
            'figure7': 'figure07_robust_pca',
            'figure9': 'figure09_hyperspectral',
            'figure12': 'figure12_pet_ct',
        }
        for sub in os.listdir('RESULTS-BASELINE'):
            sub_p = os.path.join('RESULTS-BASELINE', sub)
            if os.path.isdir(sub_p):
                target_sub = baseline_fig_map.get(sub, sub)
                move_tree_contents(sub_p, os.path.join('results/baseline', target_sub), 'results_baseline')
            else:
                safe_file_move(sub_p, os.path.join('results/baseline', sub), 'results_baseline')
        try:
            shutil.rmtree('RESULTS-BASELINE')
        except Exception:
            pass

    # 12. 14 results_* directories -> results/extensions/ and results/hybrid/
    results_dir_map = {
        'results_completion': ('results/extensions/completion/standard', 'results_completion'),
        'results_completion_debug': ('results/extensions/completion/debug', 'results_completion'),
        'results_completion_sanity': ('results/extensions/completion/sanity', 'results_completion'),
        'results_completion_smoke': ('results/extensions/completion/smoke', 'results_completion'),
        'results_ensemble': ('results/extensions/ensemble/standard', 'results_ensemble'),
        'results_ensemble_smoke': ('results/extensions/ensemble/smoke', 'results_ensemble'),
        'results_interpretability': ('results/extensions/interpretability/standard', 'results_interpretability'),
        'results_interpretability_smoke': ('results/extensions/interpretability/smoke', 'results_interpretability'),
        'results_robustness': ('results/extensions/robustness/standard', 'results_robustness'),
        'results_sparsity': ('results/extensions/sparsity/standard', 'results_sparsity'),
        'results_sparsity_corrected': ('results/extensions/sparsity/corrected', 'results_sparsity'),
        'results_sparsity_diagnostic': ('results/extensions/sparsity/diagnostic', 'results_sparsity'),
        'results_hybrid': ('results/hybrid/hybrid_attention', 'results_hybrid'),
        'results_spectral_transformer': ('results/hybrid/spectral_transformer', 'results_hybrid'),
    }
    for r_dir, (t_dir, cat) in results_dir_map.items():
        if os.path.exists(r_dir):
            move_tree_contents(r_dir, t_dir, cat)

    # 13. Loose items directly inside results/
    results_internal_moves = [
        ('results/adaptive', 'results/extensions/adaptive_rank/runs/adaptive', 'results_extensions_adaptive'),
        ('results/adaptive_fixed', 'results/extensions/adaptive_rank/runs/adaptive_fixed', 'results_extensions_adaptive'),
        ('results/adaptive_rank_results.png', 'results/extensions/adaptive_rank/adaptive_rank_results.png', 'results_extensions_adaptive'),
        ('results/adaptive_rank_selection', 'results/extensions/adaptive_rank/standard', 'results_extensions_adaptive'),
        ('results/comparison', 'results/extensions/comparison', 'results_extensions_comparison'),
        ('results/factor_group-sparsity', 'results/extensions/sparsity/factor_group_sparsity', 'results_extensions_sparsity'),
        ('results/sparsity', 'results/extensions/sparsity/runs/sparsity', 'results_extensions_sparsity'),
        ('results/sparsity_rank_results.png', 'results/extensions/sparsity/sparsity_rank_results.png', 'results_extensions_sparsity'),
        ('results/_archived_diagnostic_l0.001', 'results/extensions/sparsity/archived_diagnostic_l0.001', 'results_extensions_sparsity'),
        ('results/_validation_10_epoch', 'results/extensions/historical/_validation_10_epoch', 'results_extensions_historical'),
        ('results/baseline_fixed', 'results/baseline/baseline_fixed', 'results_baseline'),
        ('results/baseline_figure9_metrics.md', 'results/baseline/baseline_figure9_metrics.md', 'results_baseline'),
        ('results/baseline_figure9_report.md', 'results/baseline/baseline_figure9_report.md', 'results_baseline'),
    ]
    for src, dst, cat in results_internal_moves:
        if os.path.exists(src):
            if os.path.isdir(src):
                move_tree_contents(src, dst, cat)
            else:
                safe_file_move(src, dst, cat)

    # Write manifest CSV
    with open(MANIFEST_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['original_path', 'new_path', 'category', 'action', 'sha256'])
        writer.writeheader()
        writer.writerows(manifest_records)

    print(f"Cleanup completed successfully! Logged {len(manifest_records)} entries to {MANIFEST_PATH}")

if __name__ == '__main__':
    main()

