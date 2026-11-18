import numpy as np

attn_weights = np.load('results_spectral_transformer/attention_weights_smoke.npy')
print("Shape:", attn_weights.shape)
print("Finite:", np.isfinite(attn_weights).all())
row_sums = attn_weights.sum(axis=-1)
print("Row sums mean:", row_sums.mean())
print("Row sums min:", row_sums.min())
print("Row sums max:", row_sums.max())
