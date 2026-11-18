import os

ROOT = os.path.abspath('e:/Rakhal_Tensor')

dirs_to_create = [
    'configs/reproduction',
    'configs/improvements',
    'configs/hybrid',
    'src/deeptensor/models/backbones',
    'src/deeptensor/models/attention',
    'src/deeptensor/physics',
    'src/deeptensor/losses',
    'src/deeptensor/baselines',
    'src/deeptensor/training',
    'src/deeptensor/utils',
    'experiments/reproduction/figure02_eigenfaces',
    'experiments/reproduction/figure04_efficiency',
    'experiments/reproduction/figure06_synthetic_robustness',
    'experiments/reproduction/figure07_robust_pca',
    'experiments/reproduction/figure09_hyperspectral',
    'experiments/reproduction/figure10_yale_denoising',
    'experiments/reproduction/figure11_video_cs',
    'experiments/reproduction/figure12_pet_ct',
    'experiments/reproduction/figure13a_hsi_rank',
    'experiments/reproduction/figure13b_yale_rank',
    'experiments/reproduction/table01_synthetic',
    'experiments/reproduction/table02_nmf',
    'experiments/reproduction/table03_speech',
    'experiments/improvements/figure10_yale',
    'experiments/improvements/figure11_video',
    'experiments/improvements/figure12_pet_ct',
    'experiments/improvements/figure13b_yale_rank',
    'experiments/extensions/adaptive_rank',
    'experiments/extensions/group_sparsity',
    'experiments/extensions/tensor_completion',
    'experiments/extensions/noise_robustness',
    'experiments/extensions/ensemble_uncertainty',
    'experiments/extensions/factor_interpretability',
    'experiments/extensions/hybrid_attention',
    'experiments/extensions/spectral_transformer',
    'experiments/legacy_experiments',
    'experiments/hybrid_prototypes',
    'results/baseline',
    'results/improved',
    'results/extensions',
    'results/hybrid',
    'reports/reproduction',
    'reports/improvements',
    'reports/extensions',
    'reports/audits',
    'apps',
    'tests',
    'tools',
    'archive/backups',
    'archive/legacy',
    'migration'
]

print("Creating target directory scaffold...")
for d in dirs_to_create:
    p = os.path.join(ROOT, d)
    os.makedirs(p, exist_ok=True)
print(f"Scaffold created: {len(dirs_to_create)} directories ready.")

