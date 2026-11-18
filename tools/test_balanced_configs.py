from find_exact_tri_branch import calc_tri_branch_params

target_128 = 132480
candidates_128 = []
for c_cnn in range(100, 200):
    for c_pw in range(100, 200):
        for d_attn in [56, 64, 72]:
            for d_route in [16, 24]:
                total, r_p, b1_p, b2_p, b3_p = calc_tri_branch_params(128, c_cnn, c_pw, d_attn, d_route)
                diff = abs(total - target_128)
                if diff <= 20:
                    candidates_128.append((diff, c_cnn, c_pw, d_attn, d_route, total, r_p, b1_p, b2_p, b3_p))

candidates_128.sort()
print(f"Top balanced candidates for d_model=128 (target={target_128}):")
for c in candidates_128[:5]:
    print(f"Diff: {c[0]:+d} | c_cnn={c[1]}, c_pw={c[2]}, d_attn={c[3]}, d_route={c[4]} | Total={c[5]:,} (Router={c[6]:,}, B1_CNN={c[7]:,}, B2_PW={c[8]:,}, B3_Attn={c[9]:,})")

