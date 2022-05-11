import argparse
import os
import tqdm
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import losses
import models
from utils import batch_to_device, batch_errors, batch_compute_utils, log_poses, log_errors

# Imports for Depth Estimator
import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import torchvision
import urllib
import zipfile


exec(open('./3d-ken-burns/common.py', 'r').read())

exec(open('./3d-ken-burns/models/disparity-estimation.py', 'r').read())
exec(open('./3d-ken-burns/models/disparity-adjustment.py', 'r').read())
exec(open('./3d-ken-burns/models/disparity-refinement.py', 'r').read())
exec(open('./3d-ken-burns/models/pointcloud-inpainting.py', 'r').read())

class Depthestim:
    def __init__(self):
        assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

        # torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

        # torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    def get_file_names(self, dataset_folder):
        raw_data_paths = []
        for file in os.listdir(dataset_folder):
            if file[-4:] in [".pgm", ".tif", ".png"]:
                raw_data_paths.append(os.path.join(dataset_folder, file))
        return raw_data_paths

    def get_x_min_max(self, base_path, image_paths):
        x_min = []
        x_max = []
        with torch.no_grad():
            for image_path in image_paths:
                npyImage = cv2.imread(filename=base_path + image_path, flags=cv2.IMREAD_COLOR)
            
                fltFocal = max(npyImage.shape[1], npyImage.shape[0]) / 2.0
                fltBaseline = 40.0
                
                tenImage = torch.FloatTensor(np.ascontiguousarray(npyImage.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
                tenDisparity = disparity_estimation(tenImage)
                tenDisparity = disparity_adjustment(tenImage, tenDisparity)
                tenDisparity = disparity_refinement(torch.nn.functional.interpolate(input=tenImage, size=(tenDisparity.shape[2] * 4, tenDisparity.shape[3] * 4), mode='bilinear', align_corners=False), tenDisparity)
                tenDisparity = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False) * (max(tenImage.shape[2], tenImage.shape[3]) / 256.0)
                tenDepth = (fltFocal * fltBaseline) / (tenDisparity + 0.0000001)

                npyDisparity = tenDisparity[0, 0, :, :].cpu().numpy()
                npyDepth = tenDepth[0, 0, :, :].cpu().numpy()

                depth_data = np.sort(npyDepth, axis=None)
                # print(f'depth map shape is {depth_data.shape}')

                depth_count = depth_data.shape[0]

                x_min.append(depth_data[int(depth_count/3)])
                x_max.append(depth_data[int(depth_count/9)])
        xmin = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(x_min), axis=1), axis=1)
        xmax = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(x_max), axis=1), axis=1)
        return xmin, xmax



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path', metavar='DATA_PATH',
        help='path to the dataset directory, e.g. "/home/data/KingsCollege"'
    )
    parser.add_argument(
        '--loss', help='loss function for training',
        choices=['local_homography', 'global_homography', 'posenet', 'homoscedastic', 'geometric', 'dsac'],
        default='local_homography'
    )
    parser.add_argument('--epochs', help='number of epochs for training', type=int, default=5000)
    parser.add_argument('--batch_size', help='training batch size', type=int, default=64)
    parser.add_argument(
        '--weights', metavar='WEIGHTS_PATH',
        help='path to weights with which the model will be initialized'
    )
    parser.add_argument(
        '--cuda', action='store_true',
        help='train the model on GPU (may crash if cuda is not available)'
    )
    args = parser.parse_args()

    # Set seed for reproductibility
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create a Depth Estimator
    depth_estimator = Depthestim()

    # Set the device
    device = 'cpu'
    if args.cuda:
        device = 'cuda:1'


    # Load model
    model = models.load_model(args.weights)
    model.train()
    model.to(device)

    # Load dataset
    dataset = datasets.CambridgeDataset(args.path)

    # Wrapper for use with PyTorch's DataLoader
    train_dataset = datasets.RelocDataset(dataset.train_data)
    test_dataset = datasets.RelocDataset(dataset.test_data)

    # Creating data loaders for train and test data
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=datasets.collate_fn,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=datasets.collate_fn
    )

    # Adam optimizer default epsilon parameter is 1e-8
    eps = 1e-8

    # Instantiate loss
    if args.loss == 'local_homography':
        criterion = losses.LocalHomographyLoss(device=device)
        eps = 1e-14  # Adam optimizer epsilon is set to 1e-14 for homography losses
    elif args.loss == 'global_homography':
        criterion = losses.GlobalHomographyLoss(
            xmin=dataset.train_global_xmin,
            xmax=dataset.train_global_xmax,
            device=device
        )
        eps = 1e-14  # Adam optimizer epsilon is set to 1e-14 for homography losses
    elif args.loss == 'posenet':
        criterion = losses.PoseNetLoss(beta=500)
    elif args.loss == 'homoscedastic':
        criterion = losses.HomoscedasticLoss(s_hat_t=0.0, s_hat_q=-3.0, device=device)
    elif args.loss == 'geometric':
        criterion = losses.GeometricLoss()
    elif args.loss == 'dsac':
        criterion = losses.DSACLoss()
    else:
        raise Exception(f'Loss {args.loss} not recognized...')

    # Instantiate adam optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=1e-4, eps=eps)

    # Set up tensorboard
    writer = SummaryWriter(os.path.join('logs', os.path.basename(os.path.normpath(args.path)), args.loss))

    # Set up folder to save weights
    if not os.path.exists(os.path.join(writer.log_dir, 'weights')):
        os.makedirs(os.path.join(writer.log_dir, 'weights'))

    # Set up file to save logs
    log_file_path = os.path.join(writer.log_dir, 'epochs_poses_log.csv')
    with open(log_file_path, mode='w') as log_file:
        log_file.write('epoch,image_file,type,w_tx_chat,w_ty_chat,w_tz_chat,chat_qw_w,chat_qx_w,chat_qy_w,chat_qz_w\n')

    epoch = 0 # made it like that just for testing (has no meaning)
    
    with torch.no_grad():

        # Set the model to eval mode for test data
        model.eval()
        t_errors, q_errors, reprojection_errors, img_names = [], [], [], []

        for batch in test_loader:

            # Compute test poses estimations
            batch = batch_to_device(batch, device)
            batch['w_t_chat'], batch['chat_q_w'] = model(batch['image']).split([3, 4], dim=1)
            x_min, x_max = depth_estimator.get_x_min_max('/mundus/vgarg872/Documents/machine-perception/homography/datasets/ShopFacade/', batch['image_file'])
            x_min = x_min.cpu().detach().numpy()
            x_max = x_max.cpu().detach().numpy()
            batch_x_min = batch['xmin'].view(-1, 1, 1).cpu().detach().numpy()
            batch_x_max = batch['xmax'].view(-1, 1, 1).cpu().detach().numpy()
            batch_compute_utils(batch)

            # Log test poses
            with open(log_file_path, mode='a') as log_file:
                log_poses(log_file, batch, epoch, 'test')

            # Compute test errors
            batch_t_errors, batch_q_errors, batch_reprojection_errors = batch_errors(batch)
            t_errors.append(batch_t_errors)
            q_errors.append(batch_q_errors)
            reprojection_errors += batch_reprojection_errors

            mean_batch_intensities = (torch.mean(batch['image'], dim = [1, 2, 3]) - torch.min(batch['image']))/(torch.max(batch['image']) - torch.min(batch['image']))
            for i in range(len(batch)):
                # img_names += [(batch['image_file'][i], batch_reprojection_errors[i].mean(), batch['image'][i])]
                img_names += [(batch['image_file'][i], batch_reprojection_errors[i].mean().cpu().detach().numpy(), mean_batch_intensities[i].cpu().detach().numpy(), x_min[i], x_max[i], batch_x_min[i], batch_x_max[i])]
        imgs_data = np.array(sorted(img_names, key = lambda x: x[1].item()))
        
        # mean_intensities = [torch.mean(img_data[2], dim = [0, 1, 2]) for img_data in imgs_data ]
        print(f'Mean Intensities are {imgs_data}')
        # print(f'Depth Data {depth_estimates}')
        # Log test errors
        log_errors(t_errors, q_errors, reprojection_errors, writer, epoch, 'test')
        np.save('intensity_data.npy',imgs_data)
        # Log loss parameters, if there are any
        for p_name, p in criterion.named_parameters():
            writer.add_scalar(p_name, p, epoch)

        writer.flush()
        model.train()

        # Save model and optimizer weights every 10 epochs:
        if epoch % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(writer.log_dir, 'weights', f'epoch_{epoch}.pth'))

    writer.close()
