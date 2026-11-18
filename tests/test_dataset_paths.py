import os
import scipy.io as io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')

def test_mat_datasets_exist():
    expected_mats = {
        'weizzman.mat': (840, 103, 71),
        'icvl.mat': (174, 164, 260),
        'yale_b.mat': (192, 168, 160),
        'pet.mat': (128, 128, 56),
        'cat_video.mat': (128, 128, 64)
    }
    for filename, expected_shape in expected_mats.items():
        fp = os.path.join(DATA_DIR, filename)
        assert os.path.exists(fp), f"Missing dataset: {filename}"
        mat = io.loadmat(fp)
        # Check primary array
        found_shape = None
        for k in ['faces', 'hypercube', 'cat_video', 'weizzman']:
            if k in mat:
                found_shape = mat[k].shape
                break
        if found_shape is None:
            # take first non-dunder key
            keys = [k for k in mat.keys() if not k.startswith('__')]
            found_shape = mat[keys[0]].shape
        assert found_shape == expected_shape, f"{filename} expected shape {expected_shape}, got {found_shape}"

def test_benchmark_dirs_exist():
    expected_dirs = ['MNIST', 'cifar-10-batches-py', 'SpeechCommands']
    for d in expected_dirs:
        dp = os.path.join(DATA_DIR, d)
        assert os.path.exists(dp) and os.path.isdir(dp), f"Missing benchmark directory: {d}"
