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
import torch
import os
from torchvision.utils import save_image

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

