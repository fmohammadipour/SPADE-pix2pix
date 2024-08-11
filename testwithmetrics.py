"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict
import csv
import torch
from skimage.metrics import structural_similarity as ssim
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision.utils import save_image
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from util.util import tensor2im
from PIL import Image
import data
# Import L1 Loss function from PyTorch
import torch.nn.functional as F

# Parse options
opt = TestOptions().parse()

# Load the dataset
dataloader = data.create_dataloader(opt)

# Create model
model = Pix2PixModel(opt)
model.eval()

# Determine device and move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create tool for visualization
visualizer = Visualizer(opt)

# Create a webpage that summarizes all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# Directories for saving images
real_images_dir = './test_real_images'
fake_images_dir = './test_fake_images'
os.makedirs(real_images_dir, exist_ok=True)
os.makedirs(fake_images_dir, exist_ok=True)

# CSV file to save metrics
test_metrics_file = 'test_metrics_log.csv'
fieldnames = ['image_id', 'ssim', 'psnr', 'fid', 'L1']
if not os.path.exists(test_metrics_file):
    with open(test_metrics_file, 'w', newline='') as csvfile:
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()

# Define metric functions
def compute_ssim(img1, img2):
    img1 = img1.permute(1, 2, 0).numpy()
    img2 = img2.permute(1, 2, 0).numpy()
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    ssim_value, _ = ssim(img1, img2, full=True, multichannel=True, win_size=win_size, channel_axis=-1, data_range=img1.max() - img1.min())
    return ssim_value

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def compute_fid(real_images_path, fake_images_path, device):
    fid_value = calculate_fid_given_paths([real_images_path, fake_images_path], batch_size=50, device=device, dims=2048)
    return fid_value

def compute_l1_loss(img1, img2):
    # Ensure both tensors are on the same device
    if img2.device != img1.device:
        img2 = img2.to(img1.device)

    # Compute the L1 loss
    return F.l1_loss(img1, img2).item()

# Test loop
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    # Move only tensor data to the appropriate device
    for key, value in data_i.items():
        if isinstance(value, torch.Tensor):
            data_i[key] = value.to(device)

    generated = model(data_i, mode='inference')
    synthesized_image = tensor2im(generated)
    img_path = data_i['path']
    
    for b in range(synthesized_image.shape[0]):
        print('Process image... %s' % img_path[b])
        
        # Save real and fake images for FID computation
        real_image = data_i['image'][b].to('cpu')
        fake_image = generated[b].to('cpu')
        save_image(real_image, os.path.join(real_images_dir, f'real_{i}_{b}.png'), normalize=True)
        save_image(fake_image, os.path.join(fake_images_dir, f'fake_{i}_{b}.png'), normalize=True)
        
        # Compute SSIM and PSNR
        ssim_value = compute_ssim(fake_image.detach(), real_image.detach())
        psnr_value = compute_psnr(fake_image.detach(), real_image.detach())
        #print(f"Image {i}_{b} - SSIM: {ssim_value}, PSNR: {psnr_value}")
        l1_value = compute_l1_loss(fake_image.detach(), real_image.detach())    
        
        # Save metrics to CSV file
        with open(test_metrics_file, 'a', newline='') as csvfile:
            writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer_csv.writerow({'image_id': f'{i}_{b}', 'ssim': ssim_value, 'psnr': psnr_value, 'fid': 'N/A','L1': l1_value})
        
        # Save synthesized image
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', fake_image)])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        save_image_path = os.path.join(opt.results_dir, os.path.basename(img_path[b]))
        Image.fromarray(synthesized_image[b]).save(save_image_path)

# Compute FID for the entire test set
fid_value = compute_fid(real_images_dir, fake_images_dir, device)
print(f"FID for test set: {fid_value}")

# Log FID to CSV file
with open(test_metrics_file, 'a', newline='') as csvfile:
    writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer_csv.writerow({'image_id': 'overall', 'ssim': 'N/A', 'psnr': 'N/A', 'fid': fid_value})

# Save webpage
webpage.save()

print('Testing was successfully finished.')
