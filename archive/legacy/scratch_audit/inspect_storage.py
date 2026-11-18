import json
import os

with open('scratch/comprehensive_audit_data.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Checkpoints ({len(data['checkpoints'])}):")
for cp in sorted(data['checkpoints'], key=lambda x: x['size'], reverse=True):
    mb = cp['size'] / (1024 * 1024)
    print(f"  {cp['path']:<80s} | {mb:6.2f} MB")

print(f"\nDatasets (.mat) ({len(data['datasets_mat'])}):")
for ds in sorted(data['datasets_mat'], key=lambda x: x['size'], reverse=True):
    mb = ds['size'] / (1024 * 1024)
    print(f"  {ds['path']:<50s} | {mb:6.2f} MB")

print(f"\nMetrics JSON count: {len(data['metrics_json'])}")
print(f"Metrics CSV count: {len(data['metrics_csv'])}")
print(f"Plots PNG count: {len(data['plots_png'])}")
print(f"Reports MD count: {len(data['reports_md'])}")

