from find_exact_tri_branch import calc_tri_branch_params

target_64 = 49984
candidates_64 = []
for c_cnn in range(30, 120):
    for c_pw in range(30, 120):
        for d_attn in [24, 28, 32, 36, 40, 48]:
            for d_route in [4, 6, 8, 10, 12]:
                total, r_p, b1_p, b2_p, b3_p = calc_tri_branch_params(64, c_cnn, c_pw, d_attn, d_route)
                diff = abs(total - target_64)
                if diff <= 5:
                    candidates_64.append((diff, c_cnn, c_pw, d_attn, d_route, total, r_p, b1_p, b2_p, b3_p))

candidates_64.sort()
print(f"Top candidates for d_model=64 (target={target_64}):")
for c in candidates_64[:8]:
    print(f"Diff: {c[0]:+d} | c_cnn={c[1]}, c_pw={c[2]}, d_attn={c[3]}, d_route={c[4]} | Total={c[5]:,} (Router={c[6]:,}, B1_CNN={c[7]:,}, B2_PW={c[8]:,}, B3_Attn={c[9]:,})")
