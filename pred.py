"""Training a model for predicting spray painting
trajectory, given object point-cloud in input

    Examples:
        - Quick: python train.py --epochs 200 --pc_points 512 --traj_points 200 -bs 4 --loss chamfer --seed 3 --debug
        - Complete (cuboids): python train.py --epochs 1250 --pc_points 5120 --traj_points 2000 -bs 32 --loss chamfer rich_attraction_chamfer --seed 3 --backbone pointnet2 --pretrained --lambda_points 4 --extra_data orientnorm --weight_orient 0.25 --weight_rich_attraction_chamfer 0.5
        - Reproduce paper results:
            - python train.py --config cuboids_stable_v1.json --seed 42
            - python train.py --config cuboids_lambda1_v1.json --seed 42
            - python train.py --config windows_stable_v1.json --seed 42
            - python train.py --config shelves_stable_v1.json --seed 42
            - python train.py --config containers_stable_v1.json --seed 42
"""
import pdb
import sys
import argparse
from pprint import pprint
import time
import socket
import shutil

import numpy as np
import torch
from functorch.dim import Tensor

from paintnet_utils import *
from paintnet_loader import PaintNetDataloader
from model_utils import get_model, init_from_pretrained
from loss_handler import LossHandler
from metrics_handler import MetricsHandler

import trimesh



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name',           default=None, type=str, help='Run name')
    parser.add_argument('--mesh_file',           default=None, type=str, help='Mesh file used for prediction')
    parser.add_argument('--backbone',       default='pointnet2', type=str, help='Backbone [pointnet2]')
    parser.add_argument('--pc_points',      default=5120, type=int, help='Number of points to sub-sample for each point-cloud')
    parser.add_argument('--eval_ckpt',      default='best', type=str, help='Checkpoint for evaluating final results (best, last)')
    parser.add_argument('--seed',           default=0, type=int, help='Random seed (not set when equal to zero)')
    parser.add_argument('--model_dir',     default='runs', type=str, help='Dir for saved models and results')
    parser.add_argument('--output_dir',     default='predictions', type=str, help='Dir for saving predicted results')

    parser.add_argument('--extra_data',     default=[], type=str, nargs='+', help="list of str [vel, orientquat, orientrotvec, orientnorm]")
    parser.add_argument('--lambda_points',  default=1, type=int, help='Traj is considered as point-cloud made of vectors of <lambda> ordered points (Default=1, meaning that' \
                                                                      'chamfer distance would be computed normally on each traj point)')
    parser.add_argument('--traj_points',    default=500, type=int, help='Number of points to sub-sample for each trajectory')
    parser.add_argument('--overlapping',    default=0, type=int, help='Number of overlapping points between subsequent mini-sequences (only valid when lambda_points > 1)')
    parser.add_argument('--weight_orient',  default=1.0, type=float, help='Weight for L2-norm between orientation w.r.t. positional L2-norm')

    parser.add_argument('--config',         default=None, type=str, help='name of .json file in configs/ dir')

    return parser.parse_args()


args = parse_args()
config = get_train_config(args.config)
config = {**args.__dict__, **config}

def main():
    set_seed(args.seed)

    run_name = args.run_name
    save_dir = os.path.join(args.model_dir, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['eval_ckpt'] == 'best':
        eval_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=torch.device(device))
    elif config['eval_ckpt'] == 'last':
        eval_checkpoint = torch.load(os.path.join(save_dir, 'last_checkpoint.pth'), map_location=torch.device(device))
    else:  # default
        print('\n\nWARNING! Falling back to best_model.pth as eval_ckpt has invalid name.\n\n')
        eval_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=torch.device(device))

    model = get_model(config['backbone'], config=config)
    model.load_state_dict(eval_checkpoint['model'], strict=True)
    model.to(device)
    model.eval()

    point_cloud = convert_mesh_to_pointcloud(config["mesh_file"], config["pc_points"])

    point_cloud = torch.from_numpy(point_cloud)
    point_cloud = point_cloud.unsqueeze(0)
    point_cloud = point_cloud.to(device, dtype=torch.float)
    point_cloud = point_cloud.permute(0, 2, 1)

    traj_pred = model(point_cloud)[0].cpu().detach().numpy()

    visualize_sequence_traj(traj_pred, extra_data=config['extra_data'])

    if config["output_dir"]:
        expected_outdim = get_dim_traj_points(config['extra_data'])
        traj_pred = traj_pred.reshape(-1, expected_outdim)
        traj_pred = remove_padding(traj_pred, config['extra_data'])

        mesh_filename = os.path.splitext(os.path.basename(config["mesh_file"]))[0]
        np.savetxt(os.path.join(config["output_dir"], run_name + "_" + mesh_filename + '.txt'), traj_pred)

def convert_mesh_to_pointcloud(filename, pc_points):
    v, f = pcu.load_mesh_vf(filename)
    f_i, bc = pcu.sample_mesh_poisson_disk(v, f, 10000)  # Num of points (not guaranteed), radius for poisson sampling
    points = pcu.interpolate_barycentric_coords(f, f_i, bc, v)

    assert points.ndim == 2 and points.shape[-1] == 3
    centroid = get_mean_mesh(filename) # np.mean(point_cloud, axis=0)
    points -= centroid

    max_distance = get_max_distance(filename)
    points /= max_distance

    assert points.shape[0] >= pc_points
    choice = np.random.choice(points.shape[0], pc_points, replace=False)  # Sub-sample point-cloud randomly
    points = points[choice, :]

    return points

if __name__ == '__main__':
    main()
