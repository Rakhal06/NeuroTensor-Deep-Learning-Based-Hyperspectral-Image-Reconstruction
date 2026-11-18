# DeepTensor — Code Dependency Diagrams

**Document ID:** EVAL-DIAG-01  
**Scope:** Architectural and Workflow Dependency Visualizations in Mermaid

---

## Workflow A: DeepTensor Baseline Workflow

```mermaid
flowchart TD
    subgraph DataPrep ["1. Data Preparation & Physics"]
        RawData["Raw Dataset (data/icvl.mat, yale_b.mat)"]
        NoiseInject["Forward Operator & Noise Injection (Poisson / Gaussian / S&P)"]
        RawData --> NoiseInject
        Y_obs["Degraded Observation Y"]
        NoiseInject --> Y_obs
    end

    subgraph DeepPrior ["2. Deep Prior Factor Generators"]
        ZU["Spatial Latent Z_U (Fixed Uniform Noise)"]
        ZV["Spectral Latent Z_V (Fixed Uniform Noise)"]
        NetU["Spatial Generator f_U (Skip U-Net 2D)"]
        NetV["Spectral Generator f_V (Skip U-Net 1D)"]
        ZU --> NetU
        ZV --> NetV
        U_fac["Spatial Factor U (H*W x R)"]
        V_fac["Spectral Factor V (R x C)"]
        NetU --> U_fac
        NetV --> V_fac
    end

    subgraph Contraction ["3. Factor Contraction & Reconstruction"]
        BMM["Matrix Contraction: torch.bmm(U, V)"]
        U_fac --> BMM
        V_fac --> BMM
        X_hat["Reconstructed Tensor X_hat"]
        BMM --> X_hat
    end

    subgraph Optimization ["4. Self-Supervised Optimization Loop"]
        Loss["Measurement Loss: L1 or L2(A(X_hat), Y)"]
        X_hat --> Loss
        Y_obs --> Loss
        Backprop["Autograd Backward: loss.backward()"]
        Loss --> Backprop
        Adam["Adam Optimizer (lr=1e-3)"]
        Backprop --> Adam
        Adam -.->|"Update Weights"| NetU
        Adam -.->|"Update Weights"| NetV
    end
```

---

## Workflow B: Five-Architecture Comparison Workflow

```mermaid
flowchart TD
    subgraph Dispatcher ["Model Factory (src/deeptensor/models/factory.py)"]
        Config["Model Selection Flag ('baseline', 'hybrid', 'cnn_control', 'pointwise_control', 'transformer')"]
        ParamMatch["Parameter Matching Engine (count_parameters, +/-0.05% tolerance)"]
        Config --> ParamMatch
    end

    subgraph Architectures ["Five Architectural Variants"]
        M1["1. Baseline CNN (skip1d)"]
        M2["2. Hybrid CNN-Transformer (SpectralSelfAttention + CNN)"]
        M3["3. CNN Control (SpectralConvControl)"]
        M4["4. Pointwise Control (SpectralPointwiseControl 1x1)"]
        M5["5. Pure Transformer (SpectralTransformer with ALiBi)"]
        ParamMatch --> M1
        ParamMatch --> M2
        ParamMatch --> M3
        ParamMatch --> M4
        ParamMatch --> M5
    end

    subgraph FactorGen ["Tensor Factor Output"]
        M1 --> Factors["Factor Matrices U, V, W"]
        M2 --> Factors
        M3 --> Factors
        M4 --> Factors
        M5 --> Factors
    end

    subgraph Benchmarks ["Empirical Applications"]
        Factors --> App1["Figure 09: ICVL HSI (Winner: Baseline CNN)"]
        Factors --> App2["Figure 10: Yale Face B (Winner: Transformer)"]
        Factors --> App3["Figure 11: Video CS (Winner: Baseline CNN)"]
        Factors --> App4["Figure 12: PET/CT (Winner: Pointwise Control)"]
    end
```

---

## Workflow C: Adaptive Rank Selection Workflow

```mermaid
flowchart TD
    StartRank["Initialize Model at Minimum Rank (R_START = 4)"] --> Train["Train Epoch (Adam, lr=1e-3)"]
    Train --> ComputeLoss["Compute Measurement L1 Loss"]
    ComputeLoss --> EMA["Update Exponential Moving Average (alpha = 0.99)"]
    EMA --> WindowCheck{"Epoch % 200 == 0 ?"}
    
    WindowCheck -- No --> CheckMax{"Epoch >= 5000 ?"}
    CheckMax -- No --> Train
    CheckMax -- Yes --> SelectBest["Evaluate Best Validation Checkpoint"]
    
    WindowCheck -- Yes --> PlateauCheck{"Plateau Detected? (dLoss < 0.001)"}
    PlateauCheck -- No --> CheckMax
    PlateauCheck -- Yes --> RankCap{"Current Rank < R_MAX (30)?"}
    
    RankCap -- Yes --> ExpandRank["Grow Rank: R = R + 2 (Allocate New Slices)"]
    ExpandRank --> ResetOpt["Re-initialize Adam Optimizer States"]
    ResetOpt --> Train
    RankCap -- No --> CheckMax
    
    SelectBest --> Landed["Final Selected Rank: R = 8 (46.1571 dB)"]
```

---

## Workflow D: Factor Group-Sparsity Workflow

```mermaid
flowchart TD
    OverParam["Over-Parameterized Model (R_MAX = 30)"] --> GenFactors["Generate Factor Matrices U and V"]
    GenFactors --> Recon["Reconstruct Tensor: Y_hat = U * V^T"]
    Recon --> ReconLoss["Data Fidelity Loss (95% Masked L1)"]
    
    subgraph GroupLasso ["Group Sparsity Regularizer"]
        GenFactors --> ColNormU["Compute ||U_{:,r}||_2^2 for all r"]
        GenFactors --> RowNormV["Compute ||V_{r,:}||_2^2 for all r"]
        ColNormU --> JointNorm["g_r = sqrt(||U_{:,r}||_2^2 + ||V_{r,:}||_2^2)"]
        RowNormV --> JointNorm
        JointNorm --> GroupSum["R(U,V) = sum_{r=1}^{R_MAX} g_r"]
    end
    
    ReconLoss --> TotalObj["Total Loss: L_recon + lambda * R(U,V)"]
    GroupSum --> TotalObj
    TotalObj --> Backprop["Backpropagation & Adam Step"]
    Backprop -.->|"Prune Inactive Components"| GenFactors
    
    subgraph GridSearch ["Hyperparameter Selection"]
        Sweep["Evaluate lambda in {0.0, 1e-4, 1e-3, 1e-2}"]
        Sweep --> HeldOutVal["5% Held-Out Blind-Spot Validation"]
        HeldOutVal --> BestModel["lambda = 1e-4 prunes R: 27 -> 3 components"]
    end
```

---

## Workflow E: PET/CT Radon Reconstruction Workflow

```mermaid
flowchart TD
    subgraph PhysicalCT ["1. Physical Measurement Model"]
        Volume["3D PET/CT Ground Truth (data/pet.mat, 128x128x56)"]
        RadonOp["Parallel-Beam Radon Transform (40 Angles, [0, pi))"]
        Volume --> RadonOp
        Noise["Poisson (tau=100) + Gaussian (SNR=2) Noise"]
        RadonOp --> Noise
        Sinogram["Measured Sinogram Y (40 x 128 x 56)"]
        Noise --> Sinogram
    end

    subgraph NeuralCP ["2. 3-Way CP Factor Generation"]
        Latents["3x 1D Latents (Z_U, Z_V, Z_W)"]
        FactorNets["3x 1D Factor Generators (skip1d)"]
        Latents --> FactorNets
        Factors["U in R^{128x1000}, V in R^{128x1000}, W in R^{56x1000}"]
        FactorNets --> Factors
        CPContraction["torch.einsum('r, ir, jr, kr -> ijk', scale, U, V, W)"]
        Factors --> CPContraction
        X_recon["Reconstructed Volume X_hat (128x128x56)"]
        CPContraction --> X_recon
    end

    subgraph ForwardProj ["3. In-Loop Projection & Backprop"]
        SimSinogram["Forward Radon: radon(X_recon, 40 angles)"]
        X_recon --> SimSinogram
        L2Loss["L2 Frobenius Loss: ||radon(X_recon) - Sinogram||_F^2"]
        SimSinogram --> L2Loss
        Sinogram --> L2Loss
        L2Loss --> Optimize["Adam Optimizer Updates Generator Weights"]
        Optimize -.-> FactorNets
    end
```

---

## Workflow F: Proposed Adaptive Hybrid Model Architecture (Future Work)

> [!NOTE]
> This diagram illustrates the proposed future architecture presented on Slide 27. It depicts how the implemented components will be integrated with an adaptive routing mechanism.

```mermaid
flowchart TD
    Input["Input Tensor / Corrupted Observation"] --> LatentGen["Latent Positional Features"]
    
    subgraph RoutingEngine ["Proposed Routing Mechanism (Feature Selector)"]
        LatentGen --> Router["Learned Routing Network (Gating G)"]
        Router --> Softmax["Dynamic Softmax / Gumbel-Softmax Router"]
        Softmax --> GateCNN["Gate alpha_cnn (Local Weight)"]
        Softmax --> GateTrans["Gate alpha_trans (Global Weight)"]
    end

    subgraph MultiBranch ["Parallel Branch Processing"]
        LatentGen --> CNNBranch["CNN Branch (skip1d: Local Details & Textures)"]
        LatentGen --> TransBranch["Transformer Branch (MHSA: Global Context & Symmetries)"]
        GateCNN --> WeightedCNN["Weighted CNN Features"]
        CNNBranch --> WeightedCNN
        GateTrans --> WeightedTrans["Weighted Transformer Features"]
        TransBranch --> WeightedTrans
    end

    subgraph Fusion ["Feature Fusion & Contraction"]
        WeightedCNN --> Fuse["Feature Fusion Layer (Concatenate / Linear Projection)"]
        WeightedTrans --> Fuse
        Fuse --> FactorGen["Generate Low-Rank CP Factors (U, V, W)"]
    end

    subgraph DynamicRegulation ["Integrated Regularization (Existing Implemented Modules)"]
        FactorGen --> AdaptiveRank["Adaptive Rank Selection (Plateau-Driven Growth)"]
        FactorGen --> GroupLasso["Factor Group-Sparsity (Group Lasso Pruning)"]
        AdaptiveRank --> CP["CP Multilinear Tensor Contraction"]
        GroupLasso --> CP
    end

    CP --> Output["Reconstructed High-Fidelity Tensor Output"]
```

