def calc_tri_branch_params(d_model, c_cnn, c_pw, d_attn, d_route):
    # Router
    router_p = (d_model * d_route + d_route) + (d_route * 3 * d_model + 3 * d_model)
    
    # Branch 1 (CNN):
    # GroupNorm: 2 * d_model
    # dw1 (k=3): d_model * 3 + d_model = 4 * d_model
    # pw1 (1x1): d_model * c_cnn + c_cnn
    # dw2 (k=5): c_cnn * 5 + c_cnn = 6 * c_cnn
    # pw2 (1x1): c_cnn * d_model + d_model
    b1_p = (2 * d_model) + (4 * d_model) + (d_model * c_cnn + c_cnn) + (6 * c_cnn) + (c_cnn * d_model + d_model)
    
    # Branch 2 (Pointwise):
    # LayerNorm: 2 * d_model
    # Linear 1: d_model * c_pw + c_pw
    # Linear 2: c_pw * d_model + d_model
    b2_p = (2 * d_model) + (d_model * c_pw + c_pw) + (c_pw * d_model + d_model)
    
    # Branch 3 (Attention):
    # LayerNorm1: 2 * d_model
    # Proj In: d_model * d_attn + d_attn
    # MHA(d_attn, heads=2):
    # in_proj: 3 * d_attn * d_attn + 3 * d_attn
    # out_proj: d_attn * d_attn + d_attn
    # Proj Out: d_attn * d_model + d_model
    # LayerNorm2: 2 * d_model
    # FFN:
    # Linear 1: d_model * d_attn + d_attn
    # Linear 2: d_attn * d_model + d_model
    b3_p = (2 * d_model) + (d_model * d_attn + d_attn) + (4 * d_attn * d_attn + 4 * d_attn) + (d_attn * d_model + d_model) + (2 * d_model) + (2 * d_model * d_attn + d_attn + d_model)
    
    total = router_p + b1_p + b2_p + b3_p
    return total, router_p, b1_p, b2_p, b3_p

# Solve for d_model=128, target = 132,480
target_128 = 132480
best_128 = None
min_diff_128 = 999999

for c_cnn in range(64, 256):
    for c_pw in range(64, 256):
        for d_attn in [48, 56, 64, 72, 80]:
            for d_route in [8, 16, 24, 32]:
                total, r_p, b1_p, b2_p, b3_p = calc_tri_branch_params(128, c_cnn, c_pw, d_attn, d_route)
                diff = abs(total - target_128)
                if diff < min_diff_128:
                    min_diff_128 = diff
                    best_128 = (c_cnn, c_pw, d_attn, d_route, total, r_p, b1_p, b2_p, b3_p, diff)

print(f"Target 128: {target_128}")
print(f"Best 128: c_cnn={best_128[0]}, c_pw={best_128[1]}, d_attn={best_128[2]}, d_route={best_128[3]}")
print(f"Total: {best_128[4]:,} (Diff: {best_128[9]:+d})")
print(f"  Router:   {best_128[5]:,} params ({best_128[5]/best_128[4]*100:.2f}%)")
print(f"  Branch 1: {best_128[6]:,} params ({best_128[6]/best_128[4]*100:.2f}%)")
print(f"  Branch 2: {best_128[7]:,} params ({best_128[7]/best_128[4]*100:.2f}%)")
print(f"  Branch 3: {best_128[8]:,} params ({best_128[8]/best_128[4]*100:.2f}%)")

# Solve for d_model=64, target = 49,984
target_64 = 49984
best_64 = None
min_diff_64 = 999999

for c_cnn in range(32, 128):
    for c_pw in range(32, 128):
        for d_attn in [24, 32, 40, 48]:
            for d_route in [4, 8, 12, 16]:
                total, r_p, b1_p, b2_p, b3_p = calc_tri_branch_params(64, c_cnn, c_pw, d_attn, d_route)
                diff = abs(total - target_64)
                if diff < min_diff_64:
                    min_diff_64 = diff
                    best_64 = (c_cnn, c_pw, d_attn, d_route, total, r_p, b1_p, b2_p, b3_p, diff)

print(f"\nTarget 64: {target_64}")
print(f"Best 64: c_cnn={best_64[0]}, c_pw={best_64[1]}, d_attn={best_64[2]}, d_route={best_64[3]}")
print(f"Total: {best_64[4]:,} (Diff: {best_64[9]:+d})")
print(f"  Router:   {best_64[5]:,} params ({best_64[5]/best_64[4]*100:.2f}%)")
print(f"  Branch 1: {best_64[6]:,} params ({best_64[6]/best_64[4]*100:.2f}%)")
print(f"  Branch 2: {best_64[7]:,} params ({best_64[7]/best_64[4]*100:.2f}%)")
print(f"  Branch 3: {best_64[8]:,} params ({best_64[8]/best_64[4]*100:.2f}%)")
