import json
import csv
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
base_dir = os.path.join(project_root, 'results', 'extensions', 'ahd_net_v2_ablations', 'figure12_pet_ct')

with open(os.path.join(base_dir, 'figure12_routing_validation_audit.json'), 'r') as f:
    audit_data = json.load(f)

hist = audit_data['historical_reference']
records = {r['cond_id']: r for r in audit_data['audited_records']}

models_comp = {
    'Baseline CNN': {
        'inloop_psnr': hist['Baseline CNN']['inloop_psnr'],
        'dip_psnr': hist['Baseline CNN']['dip_psnr'],
        'ssim': hist['Baseline CNN']['ssim'],
        'params': hist['Baseline CNN']['params'],
        'source': 'historical'
    },
    'Serial Hybrid': {
        'inloop_psnr': hist['Serial Hybrid']['inloop_psnr'],
        'dip_psnr': hist['Serial Hybrid']['dip_psnr'],
        'ssim': hist['Serial Hybrid']['ssim'],
        'params': hist['Serial Hybrid']['params'],
        'source': 'historical'
    },
    'CNN Control': {
        'inloop_psnr': hist['CNN Control']['inloop_psnr'],
        'dip_psnr': hist['CNN Control']['dip_psnr'],
        'ssim': hist['CNN Control']['ssim'],
        'params': hist['CNN Control']['params'],
        'source': 'historical'
    },
    'Pointwise Control (Winner)': {
        'inloop_psnr': hist['Pointwise Control (Winner)']['inloop_psnr'],
        'dip_psnr': hist['Pointwise Control (Winner)']['dip_psnr'],
        'ssim': hist['Pointwise Control (Winner)']['ssim'],
        'params': hist['Pointwise Control (Winner)']['params'],
        'source': 'historical'
    },
    'Pure Transformer': {
        'inloop_psnr': hist['Pure Transformer']['inloop_psnr'],
        'dip_psnr': hist['Pure Transformer']['dip_psnr'],
        'ssim': hist['Pure Transformer']['ssim'],
        'params': hist['Pure Transformer']['params'],
        'source': 'historical'
    },
    'Standard AHD-Net v1': {
        'inloop_psnr': hist['Standard AHD-Net v1']['inloop_psnr'],
        'dip_psnr': hist['Standard AHD-Net v1']['dip_psnr'],
        'ssim': hist['Standard AHD-Net v1']['ssim'],
        'params': hist['Standard AHD-Net v1']['params'],
        'source': 'historical'
    },
    'Condition A: Static Equal Fusion': {
        'inloop_psnr': records['cond_a_static_equal']['inloop_psnr'],
        'dip_psnr': records['cond_a_static_equal']['dip_psnr'],
        'ssim': records['cond_a_static_equal']['dip_ssim'],
        'params': records['cond_a_static_equal']['params'],
        'source': 'stage3_validation'
    },
    'Condition B: Constant Temp (tau=1.0) CAGC': {
        'inloop_psnr': records['cond_b_constant_tau1']['inloop_psnr'],
        'dip_psnr': records['cond_b_constant_tau1']['dip_psnr'],
        'ssim': records['cond_b_constant_tau1']['dip_ssim'],
        'params': records['cond_b_constant_tau1']['params'],
        'source': 'stage3_validation'
    }
}

base_psnr = hist['Baseline CNN']['dip_psnr']
for k, v in models_comp.items():
    v['delta_vs_baseline_dip'] = v['dip_psnr'] - base_psnr

# Save JSON
json_path = os.path.join(base_dir, 'figure12_routing_validation_comparison.json')
with open(json_path, 'w') as f:
    json.dump({'experiment': 'figure12_pet_ct_routing_validation', 'models': models_comp}, f, indent=2)

# Save CSV
csv_path = os.path.join(base_dir, 'figure12_routing_validation_comparison.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Source', 'In-Loop PSNR (dB)', 'Clean DIP PSNR (dB)', 'SSIM', 'Parameters', 'Delta vs Baseline (dB)'])
    for name, data in models_comp.items():
        writer.writerow([name, data['source'], f"{data['inloop_psnr']:.4f}", f"{data['dip_psnr']:.4f}", f"{data['ssim']:.4f}", data['params'], f"{data['delta_vs_baseline_dip']:+.4f}"])

print('Successfully generated comparison JSON and CSV!')

