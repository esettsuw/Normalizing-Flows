'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# device settings
device = 'cpu' # or 'cuda'
import torch
# torch.cuda.set_device(0)

# data settings
dataset_path = "dataset"
class_name = "zipper" # "zipper", "MNIST"
modelname = "zipper" # "zipper", "MNIST"
train_pct = 100 # percentage of training set to enter the model

img_size = (448, 448)
img_dims = [3] + list(img_size)

# transformation settings
transf_rotations = True
transf_brightness = 0.0
transf_contrast = 0.0
transf_saturation = 0.0
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
n_scales = 3 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp_alpha = 3
n_coupling_blocks = 8
fc_internal = 2048 # number of neurons in hidden layers of s-t-networks
dropout = 0.0 # dropout in s-t-networks
lr_init = 2e-4
n_feat = 256 * n_scales

# dataloader parameters
if class_name == "MNIST":
    n_transforms = 1 # number of transformations per sample in training
    n_transforms_test = 1 # number of transformations per sample in testing
else:
    n_transforms = 4 # number of transformations per sample in training
    n_transforms_test = 64 # number of transformations per sample in testing 

batch_size = 24
batch_size_test = batch_size * n_transforms // n_transforms_test

# total epochs = meta_epochs * sub_epochs
meta_epochs = 5
sub_epochs = 8

# output settings
verbose = True
grad_map_viz = False
pro_score = False
hide_tqdm_bar = True
save_model = False

