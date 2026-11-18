#!/usr/bin/env python

'''
This script replicates results in Figure 2 of our main paper.
'''

import os
import sys
import tqdm
import importlib
import copy
import argparse

import numpy as np
from scipy import io
from scipy import linalg
from skimage.metrics import structural_similarity as ssim_func

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score as apfunc
from sklearn.decomposition import PCA, FastICA
from sklearn.svm import SVC

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt
plt.gray()
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deeptensor import models, utils, losses, deep_prior, deep_decoder

def do_pca_and_svm(X_train, y_train, X_test, n_components, param_grid):
    '''
        Just a combined wrapper for PCA + SVM
    '''
    
    X_train_centered = X_train - X_train.mean(1).reshape(-1, 1)
    X_train_centered /= X_train_centered.std(1).reshape(-1, 1)
    
    covmat = X_train_centered.T.dot(X_train_centered)
    N = covmat.shape[0]
    _, eigenfaces = linalg.eigh(covmat, subset_by_index=[N - n_components, N - 1])
    eigenfaces = eigenfaces.T   
    
    y_pred = do_svm(eigenfaces, X_train, y_train, X_test, param_grid)
    
    return eigenfaces, y_pred

def do_svm(eigenfaces, X_train, y_train, X_test, param_grid):
    '''
        Just do SVM
    '''
    
    X_train_centered = X_train - X_train.mean(1).reshape(-1, 1)
    
    X_test_centered = X_test - X_test.mean(1).reshape(-1, 1)
    
    X_train_proj = X_train_centered.dot(eigenfaces.T)
    X_test_proj = X_test_centered.dot(eigenfaces.T)
    
    # Train a SVM classification model for pca
    clf = GridSearchCV(
        SVC(kernel='linear', class_weight='balanced'), param_grid
    )
    clf = clf.fit(X_train_proj, y_train)

    # Quantitative evaluation of the model quality on the test set
    y_pred = clf.predict(X_test_proj)
    
    return y_pred

def do_ica_and_svm(X_train, y_train, X_test, n_components, param_grid):
    '''
        Independent component analysis
    '''
    ica = FastICA(n_components=n_components, random_state=0).fit(X_train)
    eigenfaces = ica.components_
    
    y_pred = do_svm(eigenfaces, X_train, y_train, X_test, param_grid)
    
    return eigenfaces, y_pred

def average_precision(y_test, y_pred, n_classes):
    return apfunc(np.eye(n_classes)[y_test], np.eye(n_classes)[y_pred])

if __name__ == '__main__':
    expname = 'weizzman'
    n_components = 84
    train_size = 0.25
    scale = 1
    
    # Noise constants
    tau = 20
    noise_snr = 2
    
    # Network constants
    n_inputs = n_components
    init_nconv = 128
    num_channels_up = 5
    nettype = 'dip'    
    
    # SVM constants
    param_grid = {'C': [1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0, 10.0], }
    
    # Learning constants
    sched_args = argparse.Namespace()
    scheduler_type = 'none'
    learning_rate = 1e-3
    epochs = 100
    reg_noise_std = 1.0/30.0
    exp_weight = 0.9 
    sched_args.step_size = 2000
    sched_args.gamma = pow(10, -2/epochs)
    sched_args.max_lr = learning_rate
    sched_args.min_lr = 1e-6
    sched_args.epochs = epochs 
    
    # Load data
    if expname == 'lfw':
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
        n_samples, h, w = lfw_people.images.shape
        
        X = lfw_people.data/255.0
        y = lfw_people.target
        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]
    elif expname == 'olivetti':
        X, y = fetch_olivetti_faces(return_X_y=True)
        h = 64
        w = 64
        n_classes = np.unique(y).size
        n_samples = X.shape[0]
        
    else:
        data = io.loadmat(os.path.join(project_root, 'data', 'weizzman.mat'))
        faces = data['faces'].astype(np.float32)/255
        y = data['labels'].ravel()
        n_classes = np.unique(y).size
        
        names = data['names'].ravel()
        target_names = [name[0] for name in names]
        
        # Resize to avoid flooding GPU
        data = faces
        n_samples, h, w = data.shape
        X = data.reshape(n_samples, h*w)
            
    # Split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42)
    
    # Corrupt data after splitting so that we can keep track of clean PCA
    X_train_noisy = utils.measure(X_train, noise_snr, tau)
    X_test_noisy = utils.measure(X_test, noise_snr, tau)
    
    # Generate networks
    if nettype == 'unet':
        eigen_net = models.UNetND(n_inputs, n_components, 2,
                                  init_nconv).to(device)            
        net_inp = utils.get_inp([1, n_inputs, h, w], device=device)
    elif nettype == 'dip':
        eigen_net = deep_prior.get_net(n_inputs, 'skip', 'reflection',
                                    upsample_mode='bilinear',
                                    skip_n33d=128,
                                    skip_n33u=128,
                                    num_scales=5,
                                    n_channels=n_components).to(device)
        net_inp = utils.get_inp([1, n_inputs, h, w], device=device)
    elif nettype == 'dd':
        nchans = [init_nconv]*num_channels_up
        eigen_net = deep_decoder.decodernw(n_components).to(device)
        
        H1 = h // pow(2, num_channels_up)
        W1 = w // pow(2, num_channels_up)
        net_inp = utils.get_inp([1, init_nconv, H1, W1], device=device)

    # Clean data
    print('Generating eigenfaces on clean data')
    eigenfaces_gt, y_gt = do_ica_and_svm(X_train, y_train, X_test, n_components,
                                         param_grid)
    
    print('Generating eigenfaces on noisy data')
    eigenfaces_ica, y_ica = do_ica_and_svm(X_train_noisy, y_train, X_test_noisy,
                                           n_components, param_grid)
    
    print('Computing ICA components')
    eigenfaces_pca, y_pca = do_pca_and_svm(X_train_noisy, y_train, X_test_noisy,
                                           n_components, param_grid)
    
    ## Part two DeepTensor
    # Compute covariance matrix
    print('Now starting DeepTensor')
    X_train_noisy_centered = X_train_noisy - X_train_noisy.mean(1).reshape(-1, 1)
    
    covmat = X_train_noisy_centered.T.dot(X_train_noisy_centered)
    covmat_ten = torch.tensor(covmat)[None, ...].to(device)
    
    net_params = list(eigen_net.parameters())
    inp_params = [net_inp]
    
    params = net_params + inp_params
    optimizer = torch.optim.Adam(lr=learning_rate, params=params)
    
    # Create a learning scheduler
    scheduler = utils.get_scheduler(scheduler_type, optimizer, sched_args)
    
    net_inp_delta = net_inp.detach().clone()
    
    criterion_l1 = losses.L2Norm()
    loss_array = np.zeros(epochs)
    best_loss = float('inf')
    best_epoch = 0
    
    eigenimg_array = np.zeros((h, w, n_components, epochs), dtype=np.float32)
    
    for idx in tqdm.tqdm(range(epochs)):
        # Perturb inputs
        net_inp_perturbed = net_inp + net_inp_delta.normal_()*reg_noise_std
            
        eigenimg = eigen_net(net_inp_perturbed)
        eigenvec = eigenimg.reshape(1, n_components, h*w)
                
        covmat_estim = torch.bmm(eigenvec.permute(0, 2, 1), eigenvec)
        
        loss = criterion_l1(covmat_ten - covmat_estim)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_array[idx] = loss.item()
        
        tmp = eigenimg[0, ...].permute(1, 2, 0).detach().cpu().numpy()
        eigenimg_array[:, :, :, idx] = tmp
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = idx

            best_eigenvec = copy.deepcopy(eigenvec.detach())
            best_covmat = copy.deepcopy(covmat_estim.detach())
            
        if sys.platform == 'win32':            
            img = utils.build_montage(eigenimg.squeeze().detach().cpu().numpy())
            
            # cv2.imshow('Eigenvector', utils.normalize(img, True))
            # cv2.waitKey(1)
    
    covmat_dlrf = best_covmat.squeeze().cpu().numpy()
    N = covmat_dlrf.shape[0]
    _, eigenface_dlrf = linalg.eigh(covmat_dlrf, subset_by_index=[N - n_components, N - 1])
    eigenface_dlrf = eigenface_dlrf.T
    
    # Now predict
    y_dlrf = do_svm(eigenface_dlrf, X_train_noisy, y_train, X_test_noisy,
                    param_grid)
        
    accuracy_gt = (y_test == y_gt).sum()/y_test.size
    accuracy_pca = (y_test == y_pca).sum()/y_test.size
    accuracy_ica = (y_test == y_ica).sum()/y_test.size
    accuracy_dlrf = (y_test == y_dlrf).sum()/y_test.size
    
    map_gt = average_precision(y_test, y_gt, n_classes)
    map_pca = average_precision(y_test, y_pca, n_classes)
    map_ica = average_precision(y_test, y_ica, n_classes)
    map_dlrf = average_precision(y_test, y_dlrf, n_classes)
        
    print('============================================================')
    print('FIGURE 2 EXPERIMENT RESULTS')
    print('============================================================')
    print('Noiseless PCA accuracy: %.4f | mAP: %.4f' % (accuracy_gt, map_gt))
    print('PCA accuracy:           %.4f | mAP: %.4f' % (accuracy_pca, map_pca))
    print('ICA accuracy:           %.4f | mAP: %.4f' % (accuracy_ica, map_ica))
    print('DeepTensor accuracy:    %.4f | mAP: %.4f' % (accuracy_dlrf, map_dlrf))
    print('============================================================')
    
    # Save outputs to RESULTS-BASELINE/figure2
    out_dir = os.path.join(project_root, 'results', 'baseline', 'figure02_eigenfaces')
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/plots', exist_ok=True)
    
    # 1. Save configuration
    import json
    config = {
        'experiment': 'Figure 2: DeepTensor Eigenfaces for Facial Classification',
        'dataset': 'weizzman.mat',
        'n_samples': int(n_samples),
        'image_height': int(h),
        'image_width': int(w),
        'n_classes': int(n_classes),
        'n_components': int(n_components),
        'train_size': float(train_size),
        'tau': float(tau),
        'noise_snr': float(noise_snr),
        'network_type': nettype,
        'learning_rate': float(learning_rate),
        'epochs': int(epochs),
        'reg_noise_std': float(reg_noise_std),
        'device': str(device)
    }
    with open(f'{out_dir}/configuration.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # 2. Save numerical metrics
    with open(f'{out_dir}/metrics.txt', 'w') as f:
        f.write('============================================================\n')
        f.write('DEEPTENSOR BASELINE: FIGURE 2 METRICS\n')
        f.write('============================================================\n')
        f.write('Experiment: Figure 2 (DeepTensor Eigenfaces)\n')
        f.write('Dataset: Weizmann Face Database (data/weizzman.mat)\n')
        f.write(f'Input shape: {n_samples} samples x {h}x{w}\n')
        f.write(f'Basis rank (components): {n_components}\n')
        f.write(f'Noise configuration: Poisson (tau={tau}) + Gaussian readout (snr={noise_snr})\n')
        f.write(f'Input Noisy PSNR (approx): {utils.psnr(X_train, X_train_noisy):.2f} dB\n')
        f.write(f'Training epochs: {epochs}\n')
        f.write(f'Device: {device}\n\n')
        f.write('METHOD                    ACCURACY       mAP\n')
        f.write('------------------------------------------------------------\n')
        f.write(f'Clean PCA (Oracle)        {accuracy_gt:.4f}         {map_gt:.4f}\n')
        f.write(f'Noisy PCA                 {accuracy_pca:.4f}         {map_pca:.4f}\n')
        f.write(f'Noisy ICA                 {accuracy_ica:.4f}         {map_ica:.4f}\n')
        f.write(f'DeepTensor                {accuracy_dlrf:.4f}         {map_dlrf:.4f}\n')
        f.write('------------------------------------------------------------\n\n')
        f.write(f'DeepTensor Accuracy Gain over Noisy PCA: +{(accuracy_dlrf - accuracy_pca)*100:.2f}%\n')
        f.write(f'DeepTensor mAP Gain over Noisy PCA:      +{(map_dlrf - map_pca)*100:.2f}%\n')
        f.write(f'Final Covariance Reconstruction Loss:   {best_loss:.6e}\n')

    # 3. Save basis images (top 12 eigenfaces comparison matching Fig. 2)
    top_k = 12
    fig, axes = plt.subplots(3, top_k, figsize=(20, 5))
    for k in range(top_k):
        # PCA
        axes[0, k].imshow(eigenfaces_pca[k].reshape(h, w), cmap='gray')
        axes[0, k].axis('off')
        if k == 0:
            axes[0, k].set_title('PCA', fontsize=12)
        # ICA
        axes[1, k].imshow(eigenfaces_ica[k].reshape(h, w), cmap='gray')
        axes[1, k].axis('off')
        if k == 0:
            axes[1, k].set_title('ICA', fontsize=12)
        # DeepTensor
        axes[2, k].imshow(eigenface_dlrf[k].reshape(h, w), cmap='gray')
        axes[2, k].axis('off')
        if k == 0:
            axes[2, k].set_title('DeepTensor', fontsize=12)
            
    plt.tight_layout()
    fig.savefig(f'{out_dir}/images/figure2_eigenfaces_top12.png', dpi=300)
    plt.close(fig)
    
    # Save individual basis montages
    plt.figure(figsize=(8, 8))
    plt.imshow(utils.build_montage(eigenface_dlrf[:64].reshape(-1, h, w)), cmap='gray')
    plt.title('DeepTensor Learned Eigenfaces Montage (Top 64)')
    plt.axis('off')
    plt.savefig(f'{out_dir}/images/deeptensor_eigenfaces_montage.png', dpi=300)
    plt.close()
    
    # 4. Save training loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(loss_array, label='Covariance Fitting L2 Loss', color='blue')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DeepTensor Figure 2 Covariance Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{out_dir}/plots/figure2_loss_curve.png', dpi=300)
    plt.close()

    # 5. Save classification performance comparison bar chart
    plt.figure(figsize=(6, 4))
    methods = ['Clean PCA', 'Noisy PCA', 'Noisy ICA', 'DeepTensor']
    accs = [accuracy_gt, accuracy_pca, accuracy_ica, accuracy_dlrf]
    colors = ['gray', 'crimson', 'darkorange', 'forestgreen']
    plt.bar(methods, accs, color=colors, width=0.55)
    plt.ylim(0, 1.0)
    plt.ylabel('Classification Accuracy')
    plt.title('Figure 2: Classification Accuracy under Poisson Noise')
    for i, v in enumerate(accs):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f'{out_dir}/plots/figure2_classification_accuracy.png', dpi=300)
    plt.close()
    
    # 6. Save array checkpoint
    np.savez_compressed(
        f'{out_dir}/figure2_results.npz',
        eigenfaces_pca=eigenfaces_pca,
        eigenfaces_ica=eigenfaces_ica,
        eigenface_dlrf=eigenface_dlrf,
        loss_array=loss_array,
        accuracies=[accuracy_gt, accuracy_pca, accuracy_ica, accuracy_dlrf]
    )
    print(f'All Figure 2 results successfully saved to: {out_dir}')
