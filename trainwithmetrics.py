"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import numpy as np
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from torch.utils.tensorboard import SummaryWriter
import torch
from skimage.metrics import structural_similarity as ssim
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
from torchvision.utils import save_image
import csv

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader) * opt.batchSize)

# create tool for visualization
visualizer = Visualizer(opt)

writer = SummaryWriter(log_dir=opt.log_dir)

# Directories for saving images
real_images_dir = './real_images'
fake_images_dir = './fake_images'
os.makedirs(real_images_dir, exist_ok=True)
os.makedirs(fake_images_dir, exist_ok=True)

# CSV file to save metrics
metrics_file = 'metrics_log.csv'
fieldnames = ['epoch', 'iteration', 'ssim', 'psnr', 'fid']  # Define fieldnames
if not os.path.exists(metrics_file):
    with open(metrics_file, 'w', newline='') as csvfile:
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()

def compute_ssim_old(img1, img2):
    # Convert img1 and img2 to NumPy arrays if they are PyTorch tensors
    img1_np = img1.numpy() if isinstance(img1, torch.Tensor) else np.array(img1)
    img2_np = img2.numpy() if isinstance(img2, torch.Tensor) else np.array(img2)
    
    # Ensure img1 and img2 are in shape (H, W, C) if they are not already
    if len(img1_np.shape) == 4:  # If shape is (N, H, W, C) for PyTorch tensors
        img1_np = np.transpose(img1_np, (0, 2, 3, 1))
        img2_np = np.transpose(img2_np, (0, 2, 3, 1))
    elif len(img1_np.shape) == 3:  # If shape is (H, W, C) for NumPy arrays
        img1_np = np.transpose(img1_np, (1, 2, 0))
        img2_np = np.transpose(img2_np, (1, 2, 0))
    
    # Calculate the smaller dimension of the image
    min_dim = min(img1_np.shape[0], img1_np.shape[1])
    
    # Ensure win_size is an odd value and less than or equal to the smaller side of the images
    win_size = min(min_dim // 10, min_dim)  # Adjust the fraction 
    
    if win_size % 2 == 0:
        win_size += 1  # Ensure win_size is odd
    
    # Calculate SSIM value with adjusted win_size and adaptive data_range
    if img2_np.min() == img2_np.max():
        data_range = 1.0  # If all pixels are the same, set a default range
    else:
        data_range = np.max(img2_np) - np.min(img2_np)  # Calculate range based on image content
    
    # Check if data_range is zero or close to zero
    if data_range == 0:
        data_range = 1.0  # Set a default data_range to avoid division by zero
    
    try:
        ssim_value, _ = ssim(img1_np, img2_np, win_size=win_size, full=True, multichannel=True, data_range=data_range)
    except ZeroDivisionError:
        ssim_value = 0.0  # Handle ZeroDivisionError gracefully
    
    return ssim_value


def compute_ssim(img1, img2):
    # Check if the images are 4D tensors (batch, channels, height, width)
    if img1.dim() == 4:
        # Iterate over the batch dimension
        ssim_values = []
        for i in range(img1.size(0)):
            img1_np = img1[i].permute(1, 2, 0).numpy()
            img2_np = img2[i].permute(1, 2, 0).numpy()
            ssim_value = calculate_ssim(img1_np, img2_np)
            ssim_values.append(ssim_value)
        return np.mean(ssim_values)  # Return the average SSIM over the batch

    # If images are 3D tensors (channels, height, width)
    elif img1.dim() == 3:
        img1_np = img1.permute(1, 2, 0).numpy()
        img2_np = img2.permute(1, 2, 0).numpy()
        return calculate_ssim(img1_np, img2_np)
    
    else:
        raise ValueError("Input images must be 3D or 4D tensors")

def calculate_ssim(img1_np, img2_np):
    min_dim = min(img1_np.shape[0], img1_np.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    return ssim(img1_np, img2_np, full=True, multichannel=True, win_size=win_size, channel_axis=-1, data_range=img1_np.max() - img1_np.min())[0]

def compute_psnr(img1, img2):
    # Move img2 to the same device as img1 if they are on different devices
    if img2.device != img1.device:
        img2 = img2.to(img1.device)

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def compute_fid(real_images_path, fake_images_path, device):
    # Calculate FID with device specified for operations
    fid_value = calculate_fid_given_paths([real_images_path, fake_images_path], batch_size=50, device=device, dims=2048)
    return fid_value

def clean_up_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
            
# Function to determine device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clear_iter = False
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch, clear_iter)
    clear_iter = True

    start_batch_idx = iter_counter.epoch_iter // opt.batchSize
    
    for i, data_i in enumerate(dataloader):
        if i < start_batch_idx:
            continue

        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            # Log scalar values to TensorBoard
            for name, value in losses.items():
                writer.add_scalar(name, value, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            # Convert images from NHWC to CHW and ensure proper format
            for name, image in visuals.items():
                if isinstance(image, torch.Tensor):
                    if image.ndimension() == 4:
                        # image has shape (N, H, W, C), convert to (N, C, H, W)
                        image = image.permute(0, 3, 1, 2)
                    elif image.ndimension() == 3:
                        # image has shape (H, W, C), convert to (C, H, W)
                        image = image.permute(2, 0, 1)
                    elif image.ndimension() == 2:
                        # image has shape (H, W), add channel dimension to get (1, H, W)
                        image = image.unsqueeze(0)
                    writer.add_image(name, image, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

        # Compute and save metrics periodically
        if iter_counter.total_steps_so_far % 100 == 0:
            visuals = trainer.get_latest_generated()
            real_image = data_i['image']
            fake_image = visuals
            
            ssim_value = compute_ssim(fake_image.detach().cpu(), real_image.detach().cpu())
            psnr_value = compute_psnr(fake_image.detach(), real_image.detach())
            #print(f"SSIM: {ssim_value}, PSNR: {psnr_value}")

            # Save images for FID computation periodically
            save_image(real_image, os.path.join(real_images_dir, f'real_{epoch}_{iter_counter.total_steps_so_far}.png'), normalize=True)
            save_image(fake_image, os.path.join(fake_images_dir, f'fake_{epoch}_{iter_counter.total_steps_so_far}.png'), normalize=True)
        # Determine device
        device = get_device()
        # Compute FID periodically
        if iter_counter.total_steps_so_far % 1000 == 0:
            fid_value = compute_fid(real_images_dir, fake_images_dir, device)
            print(f"FID: {fid_value}")

            # Clean up saved images periodically
            clean_up_images(real_images_dir)
            clean_up_images(fake_images_dir)

            # Save metrics to CSV file
            with open(metrics_file, 'a', newline='') as csvfile:
                writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer_csv.writerow({'epoch': epoch, 'iteration': iter_counter.total_steps_so_far, 'ssim': ssim_value, 'psnr': psnr_value, 'fid': fid_value})

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)
        
# Close TensorBoard writer
writer.close()

print('Training was successfully finished.')
