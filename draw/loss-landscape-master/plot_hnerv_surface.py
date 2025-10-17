"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import dataloader
import projection as proj
import net_plotter
import plot_2D
import plot_1D
import model_loader
import scheduler
from multiprocessing import Lock
import random
import yaml

import evaluation_nerv

sys.path.append('/sjq/NeuroQuant/')
from models import NeRV, HNeRV

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import center_crop

# Video dataset
class VideoDataSet(Dataset):
    def __init__(self, cfg, args):
        self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]

        # Resize the input video and center crop
        self.crop_h, self.crop_w = cfg['crop_h'], cfg['crop_w']
        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)
        
        self.diff = cfg['diff_enc']

    def img_load(self, idx):
        if isinstance(self.video, list):
            img = read_image(self.video[idx])
        else:
            img = self.video[idx].permute(-1,0,1)
        return img / 255.

    def img_transform(self, img):
        img = center_crop(img, (self.crop_h, self.crop_w))
        return img

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        if self.diff:
            id_p = idx-1 if idx!=0 else idx
            id_f = idx+1 if idx!=len(self.video)-1 else idx
            
            tensor_image = self.img_transform(self.img_load(idx))
            tensor_image_p = self.img_transform(self.img_load(id_p))
            tensor_image_f = self.img_transform(self.img_load(id_f))
            
            sample = {
			"img_id": idx,
			"img_gt": tensor_image, #3 h w
			"img_p" : tensor_image_p, #3 h w
			"img_f" : tensor_image_f, #3 h w 
            }
            
        else:
            tensor_image = self.img_transform(self.img_load(idx))
            norm_idx = float(idx) / len(self.video)
            sample = {'img': tensor_image, 'idx': idx, 'norm_idx': norm_idx}
        
        return sample
    
    


def get_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        return config
    
def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def Handle_Parameter(conv, b_W):
    """
        Simple symmetrical quantization.
        Args:
            conv: the layer to be quantized
            b_W: the bit-width of quantization
        Return: 
            conv: quantized layer
            weight - w_hat: quantization error, i.e., delta_w in paper.
    """
    weight = conv.weight.data
    bit_range = 2 ** (b_W - 1)

    scale = bit_range/weight.abs().max()

    w_hat = torch.round(weight * scale) /scale

    conv.weight.data = w_hat

    return conv, weight - w_hat

def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """
    criterion = nn.MSELoss()
    loss, acc = evaluation_nerv.eval_loss(net, criterion, dataloader, args.cuda)

    f = h5py.File(surf_file, 'r+')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    

    lock = Lock()
    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = evaluation_nerv.eval_loss(net, criterion, dataloader, args.cuda)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        # if rank == 0:
        #     f[loss_key][:] = losses
        #     f[acc_key][:] = accuracies
        #     f.flush()
            
        with lock:
            try:
                f[loss_key][:] = losses
                f[acc_key][:] = accuracies
                f.flush()
            except BrokenPipeError as e:
                print(f"Error occurred: {e}")

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=1, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # model parameters
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')
    
    # initial
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--arch', type=str, help='the architecture of NeRV')
    parser.add_argument('--weight', default='None', type=str, help='model for test')
    parser.add_argument('--lid', type=int, help='layer index for visualization')
    parser.add_argument('--b_w', type=int, help='the bit-width of quantization for visualization')
    
    # dataset
    parser.add_argument('--data_path', type=str, help='data path for vid')
    parser.add_argument('--vid', type=str, help='video id',)

    args = parser.parse_args()

    torch.manual_seed(123)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
              (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    cfg = get_config(args.config)
    # build model
    if args.arch == 'hnerv':
        net = HNeRV(cfg)
    elif args.arch == 'nerv':
        net = NeRV(cfg)

    assert args.weight != 'None'
    checkpoint = torch.load(args.weight, map_location='cpu')
    net.load_state_dict(checkpoint, strict=False)
    net.cpu()
    
    if args.lid == 0:
        net.decoder[0], error = Handle_Parameter(net.decoder[0], args.b_w)
    elif args.lid == 6:
        net.head_layer, error = Handle_Parameter(net.head_layer, args.b_w)
    elif args.lid == -1:
        pass
    else:
        net.decoder[args.lid].conv[0], error = Handle_Parameter(net.decoder[args.lid].conv[0], args.b_w)
    
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    
    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    full_dataset = VideoDataSet(cfg, args)
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True, sampler=None, drop_last=False, worker_init_fn=worker_init_fn)
    
    
    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = net_plotter.name_direction_file(args) # name the direction file
    if rank == 0:
        net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    
    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch(surf_file, net, w, s, d, full_dataloader, 'train_loss', 'train_acc', comm, rank, args)
    # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if args.plot and rank == 0:
        if args.y and args.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
        elif args.y:
            plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
        else:
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
