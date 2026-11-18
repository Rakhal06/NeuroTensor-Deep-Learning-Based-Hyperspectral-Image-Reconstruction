import torch

def parafac_als_cuda(tensor_target, rank, num_iters=30, verbose=False):
    """
    Exact, vectorized PyTorch CUDA Alternating Least Squares (ALS) for 3-way PARAFAC decomposition.
    tensor_target: (H, W, T) on GPU
    rank: int
    """
    device = tensor_target.device
    H, W, T = tensor_target.shape

    # Unfoldings
    # Mode-0 (H, W*T)
    X0 = tensor_target.reshape(H, W * T)
    # Mode-1 (W, H*T)
    X1 = tensor_target.permute(1, 0, 2).reshape(W, H * T)
    # Mode-2 (T, H*W)
    X2 = tensor_target.permute(2, 0, 1).reshape(T, H * W)

    # Initialization via random normal
    gen = torch.Generator(device=device).manual_seed(42)
    A = torch.randn(H, rank, device=device, generator=gen) * 0.1
    B = torch.randn(W, rank, device=device, generator=gen) * 0.1
    C = torch.randn(T, rank, device=device, generator=gen) * 0.1

    # Normalize initial factors
    A = A / (torch.norm(A, dim=0, keepdim=True) + 1e-12)
    B = B / (torch.norm(B, dim=0, keepdim=True) + 1e-12)
    C = C / (torch.norm(C, dim=0, keepdim=True) + 1e-12)

    def khatri_rao(Mat1, Mat2):
        # Mat1: (I, R), Mat2: (J, R) -> (I*J, R)
        R = Mat1.shape[1]
        return (Mat1.unsqueeze(1) * Mat2.unsqueeze(0)).reshape(-1, R)

    reg = 1e-6 * torch.eye(rank, device=device)

    for it in range(num_iters):
        # Update A: X0 approx A (C (KR) B)^T
        KR_CB = khatri_rao(C, B)
        VtV = (C.T @ C) * (B.T @ B) + reg
        A = torch.linalg.solve(VtV, (X0 @ KR_CB).T).T

        # Update B: X1 approx B (C (KR) A)^T
        KR_CA = khatri_rao(C, A)
        VtV = (C.T @ C) * (A.T @ A) + reg
        B = torch.linalg.solve(VtV, (X1 @ KR_CA).T).T

        # Update C: X2 approx C (B (KR) A)^T
        KR_BA = khatri_rao(B, A)
        VtV = (B.T @ B) * (A.T @ A) + reg
        C = torch.linalg.solve(VtV, (X2 @ KR_BA).T).T

    # Reconstruct tensor: X_hat[h, w, t] = sum_r A[h, r] * B[w, r] * C[t, r]
    # Vectorized: einsum 'hr, wr, tr -> hwt'
    X_hat = torch.einsum('hr,wr,tr->hwt', A, B, C)
    return X_hat, (A, B, C)
