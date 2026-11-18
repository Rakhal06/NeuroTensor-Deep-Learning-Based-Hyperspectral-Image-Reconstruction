#!/usr/bin/env python

'''
This script implements models other than the one proposed by [Ulyanov et al. 2018],
and [Heckel et al. 2018].
'''

import torch


class SimpleForwardMLP(torch.nn.Module):
    """
    Simple feed forward MLP

    Inputs:
        n_inputs: Number of input dim
        ksizes: (nlayers,) Kernel sizes for each layer

    """

    def __init__(self, n_inputs, ksizes):
        super().__init__()

        self.relu = torch.nn.ReLU(inplace=True)

        modules = [torch.nn.Linear(n_inputs, ksizes[0]), self.relu]

        for idx in range(1, len(ksizes)):
            modules.append(torch.nn.Linear(ksizes[idx - 1], ksizes[idx]))
            if idx < len(ksizes) - 1:
                modules.append(self.relu)
        self.module_list = torch.nn.Sequential(*modules)

    def forward(self, X, **kwargs):
        return self.module_list(X)


class SimpleForward2D(torch.nn.Module):
    """
    Simple feed forward 2D convolutional neural network

    Inputs:
        n_inputs: Number of input channels
        n_outputs: Number of output channels
        nconvs: (nlayers,) filters for each layer

    """

    def __init__(self, n_inputs, n_outputs, nconvs):
        super().__init__()

        self.relu = torch.nn.ReLU(inplace=True)

        modules = [
            torch.nn.Conv2d(n_inputs, nconvs[0], (15, 15), padding=(7, 7)),
            self.relu,
            torch.nn.BatchNorm2d(nconvs[0]),
        ]

        for idx in range(len(nconvs) - 1):
            modules.append(
                torch.nn.Conv2d(nconvs[idx], nconvs[idx + 1], (3, 3), padding=(1, 1))
            )
            modules.append(self.relu)
            modules.append(torch.nn.BatchNorm2d(nconvs[idx + 1]))

        modules.append(torch.nn.Conv2d(nconvs[-1], n_outputs, (3, 3), padding=(1, 1)))
        modules.append(torch.nn.BatchNorm2d(n_outputs))

        self.module_list = torch.nn.Sequential(*modules)

    def forward(self, X, **kwargs):
        return self.module_list(X)

class SimpleForward1D(torch.nn.Module):
    """
    Simple feed forward 2D convolutional neural network

    Inputs:
        n_inputs: Number of input channels
        n_outputs: Number of output channels
        nlayers: Number of layers
        ksizes: (nlayers,) Kernel sizes for each layer
        nconvs: (nlayers,) filters for each layer

    """

    def __init__(self, n_inputs, n_outputs, nlayers, nconvs):
        super().__init__()

        self.relu = torch.nn.LeakyReLU(0.2, inplace=True)

        modules = [
            torch.nn.Conv1d(n_inputs, nconvs[0], 15, padding=7),
            self.relu,
            torch.nn.BatchNorm1d(nconvs[0]),
        ]

        for idx in range(nlayers - 1):
            modules.append(torch.nn.Conv1d(nconvs[idx], nconvs[idx + 1], 3, padding=1))
            modules.append(self.relu)
            modules.append(torch.nn.BatchNorm1d(nconvs[idx + 1]))

        modules.append(torch.nn.Conv1d(nconvs[-1], n_outputs, 3, padding=1))
        modules.append(torch.nn.BatchNorm1d(n_outputs))

        self.module_list = torch.nn.Sequential(*modules)

    def forward(self, X, **kwargs):
        return self.module_list(X)

class UNetND(torch.nn.Module):
    """
    A generalrized ND Unet that can be instantiated to 1D, 2D or 3D UNet

    Inputs:
        n_inputs: Number of input channels
        n_outputs: Number of output channels
        ndim: 1, 2, or 3 for 1d, 2d, 3d
        ndown: Number of downsampling layers
        init_nconv: Number of convolutions for first layer. Subsequent
            downsampled layers have twice the number of layers as the first
            one.

    """

    def __init__(self, n_inputs, n_outputs, ndim=2, init_nconv=32):
        super().__init__()

        self.relu = torch.nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

        # Create operatorsr
        if ndim == 1:
            self.conv = torch.nn.Conv1d
            self.bn = torch.nn.BatchNorm1d
            self.maxpool = torch.nn.MaxPool1d(2)
            self.upmode = "linear"
            ksize = 5
            stride = 2
            skip_ksize = 1
            pad = 2
        elif ndim == 2:
            self.conv = torch.nn.Conv2d
            self.bn = torch.nn.BatchNorm2d
            self.maxpool = torch.nn.MaxPool2d(2)
            self.upmode = "bilinear"
            ksize = (5, 5)
            stride = (2, 2)
            skip_ksize = (1, 1)
            pad = (2, 2)
        else:
            self.conv = torch.nn.Conv3d
            self.bn = torch.nn.BatchNorm3d
            self.maxpool = torch.nn.MaxPool3d(2)
            self.upmode = "trilinear"
            ksize = (5, 5, 5)
            stride = (2, 2, 2)
            skip_ksize = (1, 1, 1)
            pad = (2, 2, 2)

        # Create a static list of convolutions -- simpler
        nconvs = [init_nconv] * 5
        # nconvs = [16, 32, 64, 128, 128]

        # Downsampling
        self.do_conv1 = self.conv(n_inputs, nconvs[0], ksize, padding=pad)
        self.do_bn1 = self.bn(nconvs[0])
        self.do_conv2 = self.conv(
            nconvs[0], nconvs[1], ksize, stride=stride, padding=pad
        )
        self.do_bn2 = self.bn(nconvs[1])
        self.do_conv3 = self.conv(
            nconvs[1], nconvs[2], ksize, stride=stride, padding=pad
        )
        self.do_bn3 = self.bn(nconvs[2])
        self.do_conv4 = self.conv(
            nconvs[2], nconvs[3], ksize, stride=stride, padding=pad
        )
        self.do_bn4 = self.bn(nconvs[3])
        self.do_conv5 = self.conv(
            nconvs[3], nconvs[4], ksize, stride=stride, padding=pad
        )
        self.do_bn5 = self.bn(nconvs[4])

        # Skip connection convolutions
        self.skip1 = self.conv(nconvs[0], nconvs[1] // 2, skip_ksize)
        self.skip2 = self.conv(nconvs[1], nconvs[2] // 2, skip_ksize)
        self.skip3 = self.conv(nconvs[2], nconvs[3] // 2, skip_ksize)
        self.skip4 = self.conv(nconvs[3], nconvs[4] // 2, skip_ksize)

        # Upsampling
        self.up_conv1 = self.conv(nconvs[4], nconvs[4] // 2, ksize, padding=pad)
        self.up_bn1 = self.bn(nconvs[4] // 2)
        self.up_conv2 = self.conv(nconvs[4], nconvs[3] // 2, ksize, padding=pad)
        self.up_bn2 = self.bn(nconvs[3] // 2)
        self.up_conv3 = self.conv(nconvs[3], nconvs[2] // 2, ksize, padding=pad)
        self.up_bn3 = self.bn(nconvs[2] // 2)
        self.up_conv4 = self.conv(nconvs[2], nconvs[1] // 2, ksize, padding=pad)
        self.up_bn4 = self.bn(nconvs[1] // 2)
        self.up_conv5 = self.conv(nconvs[1], n_outputs, ksize, padding=pad)

    def forward(self, X):
        # Downsample
        X1 = self.relu(self.do_bn1(self.do_conv1(X)))
        X2 = self.relu(self.do_bn2(self.do_conv2(X1)))
        X3 = self.relu(self.do_bn3(self.do_conv3(X2)))
        X4 = self.relu(self.do_bn4(self.do_conv4(X3)))
        X5 = self.relu(self.do_bn5(self.do_conv5(X4)))

        # Upsample
        X6 = self.relu(self.up_bn1(self.up_conv1(X5)))
        X6 = torch.nn.Upsample(size=X4.shape[2:], mode=self.upmode)(X6)

        X7 = torch.cat((X6, self.relu(self.skip4(X4))), dim=1)
        X7 = self.relu(self.up_bn2(self.up_conv2(X7)))
        X7 = torch.nn.Upsample(size=X3.shape[2:], mode=self.upmode)(X7)

        X8 = torch.cat((X7, self.relu(self.skip3(X3))), dim=1)
        X8 = self.relu(self.up_bn3(self.up_conv3(X8)))
        X8 = torch.nn.Upsample(size=X2.shape[2:], mode=self.upmode)(X8)

        X9 = torch.cat((X8, self.relu(self.skip2(X2))), dim=1)
        X9 = self.relu(self.up_bn4(self.up_conv4(X9)))
        X9 = torch.nn.Upsample(size=X1.shape[2:], mode=self.upmode)(X9)

        X10 = torch.cat((X9, self.relu(self.skip1(X1))), dim=1)
        X10 = self.up_conv5(X10)

        return X10

class DeepTensor(torch.nn.Module):
    """
    DeepTensor: Low-Rank Matrix and Tensor Decomposition with Deep Network Priors.
    Combines spatial prior generator (e.g. 2D U-Net/DIP) and spectral/temporal prior
    generator (e.g. 1D U-Net/DIP) via bilinear or multi-linear factor contraction.
    """
    def __init__(self, spatial_net, spectral_net, rank):
        super().__init__()
        self.spatial_net = spatial_net
        self.spectral_net = spectral_net
        self.rank = rank

    def forward(self, spatial_inp, spectral_inp):
        # spatial factor U: (B, rank, H, W)
        U_img = self.spatial_net(spatial_inp)
        # spectral factor V: (B, rank, C)
        V = self.spectral_net(spectral_inp)
        B, R, H, W = U_img.shape
        U = U_img.reshape(B, R, H * W).permute(0, 2, 1)  # (B, H*W, R)
        reconstruction = torch.bmm(U, V)  # (B, H*W, C)
        return reconstruction

class DeepTensorCP(torch.nn.Module):
    """
    DeepTensorCP: 3-Way PARAFAC / CANDECOMP-PARAFAC (CP) Tensor Decomposition
    with Deep Network Priors.
    Combines three 1D factor generator networks (e.g., U_net, V_net, W_net) and an optional
    learnable rank core weight vector:
        X_{ijk} = sum_{r=1}^R (core_r / R) * U_{ir} * V_{jr} * W_{kr}
    """
    def __init__(self, u_net, v_net, w_net, rank, learnable_core=True, normalize_core=True):
        super().__init__()
        self.u_net = u_net
        self.v_net = v_net
        self.w_net = w_net
        self.rank = rank
        self.normalize_core = normalize_core
        if learnable_core:
            self.core = torch.nn.Parameter(torch.ones(rank))
        else:
            self.register_buffer('core', torch.ones(rank))

    def forward(self, u_inp, v_inp, w_inp):
        u_out = self.u_net(u_inp)
        v_out = self.v_net(v_inp)
        w_out = self.w_net(w_inp)

        # Factor permutation to (dim_len, rank)
        u_fac = u_out[0].permute(1, 0) if u_out.dim() == 3 else u_out
        v_fac = v_out[0].permute(1, 0) if v_out.dim() == 3 else v_out
        w_fac = w_out[0].permute(1, 0) if w_out.dim() == 3 else w_out

        scale = (self.core / self.rank) if self.normalize_core else self.core
        recon = torch.einsum('r, ir, jr, kr -> ijk', scale, u_fac, v_fac, w_fac)
        return recon
