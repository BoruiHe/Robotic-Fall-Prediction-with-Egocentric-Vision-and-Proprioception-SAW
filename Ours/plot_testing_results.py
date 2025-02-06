import os
import sys
sys.path.insert(0, os.getcwd())
import yaml
import torch
import argparse
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from Ours.test import test_siMLPe_step, test_traj_step
from Ours.dataset import SAWdataset
from torch.utils.data import DataLoader
from Ours.model import siMLPe as Model
from utils.miscellaneous import set_random_seeds


def get_JA_for_visualization(dataset_name, exp_name, ps, seeds=None):

    if not seeds:
        seeds_list = [608, 1247, 3224]
    else:
        seeds_list = seeds

    for seed in seeds_list:
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'parameters.yml'), 'r') as infile:
            hyperparameters = yaml.safe_load(infile)
        p = os.path.join(os.getcwd(),
                        hyperparameters['parent_dir'].split(os.sep)[-3],
                        hyperparameters['parent_dir'].split(os.sep)[-2], 
                        hyperparameters['parent_dir'].split(os.sep)[-1],
                        str(seed))
        print('Get data for plotting', p)
        # a = os.path.isfile(os.path.join(p, 'motion_gt.pkl'))
        # b = os.path.isfile(os.path.join(p, 'motion_pred.pkl'))
        # c = os.path.isfile(os.path.join(p, 'motion_start_timestep.pkl'))
        # if a and b and c:
        #     print('Skip testing.')
        
        motion_pred, motion_gt, motion_start_timestep = {}, {}, {}
        set_random_seeds(int(seed))

        with open(os.path.join(p, 'intermediate_files', 'video_idx_splits.yml'), 'r') as infile:
            testing_idx = yaml.safe_load(infile)['testing_idx']

        dataset = SAWdataset(hyperparameters, 'testing', False, for_visualization=True, ps=ps)
        dataloader = DataLoader(dataset, batch_size=1,
                                num_workers=8, drop_last=False,
                                sampler=None, shuffle=False, pin_memory=True)

        model_state_dict_pth = os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), f'best_siMLPe.pth')

        model = Model(hyperparameters)
        state_dict = torch.load(model_state_dict_pth)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.cuda()

        motion_pred[seed], motion_gt[seed], motion_start_timestep[seed] = {}, {}, {}
        
        if hyperparameters['dataset_name'] == 'real_poppy':
            for ri in testing_idx:
                motion_pred[seed][ri+1], motion_gt[seed][ri+1], motion_start_timestep[seed][ri+1] = [], [], []
        elif hyperparameters['dataset_name'] == 'vir_poppy':
            for ri in dataset._idx_reps:
                motion_pred[seed][ri], motion_gt[seed][ri], motion_start_timestep[seed][ri] = [], [], []

        for eir in dataloader:
            for EachStep in eir:
                input_joint_motion, future_joint_motion, rep_idx, start_timestep = EachStep[0].cuda(), EachStep[1].cuda(), int(EachStep[-2]), int(EachStep[-1])
                pred, gt = test_siMLPe_step(input_joint_motion, future_joint_motion, model, ps, mode='visualization')
                motion_pred[seed][rep_idx].append(pred.squeeze())
                motion_gt[seed][rep_idx].append(gt.squeeze())
                motion_start_timestep[seed][rep_idx].append(start_timestep)

            motion_pred[seed][rep_idx] = np.array(motion_pred[seed][rep_idx])
            motion_gt[seed][rep_idx] = np.array(motion_gt[seed][rep_idx])
            motion_start_timestep[seed][rep_idx] = np.array(motion_start_timestep[seed][rep_idx])        

        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'motion_pred.pkl'), 'wb') as handle:
            pk.dump(motion_pred, handle, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'motion_gt.pkl'), 'wb') as handle:
            pk.dump(motion_gt, handle, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'motion_start_timestep.pkl'), 'wb') as handle:
            pk.dump(motion_start_timestep, handle, protocol=pk.HIGHEST_PROTOCOL)
        print('Done.')

def plot_JA_curves(dataset_name, exp_name, ps, seeds=None):

    if not seeds:
        seeds_list = [608, 1247, 3224]
    else:
        seeds_list = seeds

    for seed in seeds_list:
        set_random_seeds(int(seed))

        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'motion_pred.pkl'), 'rb') as handle:
            motion_pred = pk.load(handle)
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'motion_gt.pkl'), 'rb') as handle:
            motion_gt = pk.load(handle)
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'motion_start_timestep.pkl'), 'rb') as handle:
            motion_start_timestep = pk.load(handle)
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'parameters.yml'), 'r') as infile:
            hyperparameters = yaml.safe_load(infile)
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'intermediate_files', 'video_idx_splits.yml'), 'r') as infile:
            idx_list = yaml.safe_load(infile)['testing_idx']
        plots_path = os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'plots')
        if not os.path.isdir(plots_path):
            os.makedirs(plots_path)

        if dataset_name == 'RP':
            for rep_idx in idx_list:
                _, axes = plt.subplots(5, 5, figsize=(10,10))
                for axis, j in zip(axes.reshape(25), list(range(1,26))):
                    axis.ticklabel_format(axis='y', style='sci', scilimits=(-1,-2), useMathText=True)
                    axis.yaxis.offsetText.set_fontsize(15)
                    # axis.title.set_text(f'Joint: {j}')
                rep_idx += 1
                for pred, gt, start_timestep in zip(motion_pred[seed][rep_idx], motion_gt[seed][rep_idx], motion_start_timestep[seed][rep_idx]):
                    timestep = start_timestep + hyperparameters['N'] + ps
                    pred = pred[ps-1]
                    gt = gt[ps-1]
                    for i in range(25):
                        axes[int(i//5)][i%5].tick_params(axis='x', labelsize=15)
                        axes[int(i//5)][i%5].tick_params(axis='y', labelsize=15)
                        axes[int(i//5)][i%5].plot(timestep, pred[i], color='black', marker='o', markersize=2)
                        axes[int(i//5)][i%5].plot(timestep, gt[i], 'r+', markersize=2)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_path, 'Rep_{}_PS_{}.pdf'.format(rep_idx, ps)))
        elif dataset_name == 'VP':
            idx_list = list(motion_gt[seed].keys())
            for rep_idx in idx_list:
                _, axes = plt.subplots(7, 6, figsize=(10,10))
                for axis, j in zip(axes.reshape(42), list(range(42))):
                    axis.ticklabel_format(axis='y', style='sci', scilimits=(-1,-2), useMathText=True)
                    # axis.title.set_text(f'Joint: {j+1}')
                    # if j//6 == 6:
                    #     axes[int(j//6)][j%6].set_xlabel('Time steps')
                    if j//6 != 6:
                        axes[int(j//6)][j%6].set_xticks([])
                    # if j%6==0:
                    #     axes[int(j//6)][j%6].set_ylabel('Joint angle (rad)')
                for pred, gt, start_timestep in zip(motion_pred[seed][rep_idx], motion_gt[seed][rep_idx], motion_start_timestep[seed][rep_idx]):
                    timestep = start_timestep + hyperparameters['N'] + ps
                    pred = pred[ps-1]
                    gt = gt[ps-1]
                    for i in range(42):
                        axes[int(i//6)][i%6].tick_params(axis='x', labelsize=15)
                        axes[int(i//6)][i%6].tick_params(axis='y', labelsize=15)
                        axes[int(i//6)][i%6].plot(timestep, pred[i], color='black', marker='o', markersize=2)
                        axes[int(i//6)][i%6].plot(timestep, gt[i], 'r+', markersize=2)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_path, 'Rep_{}_PS_{}.pdf'.format(rep_idx, ps)))

def get_traj_for_visualization(dataset_name, exp_name, seeds=None):

    if not seeds:
        seeds_list = [608, 1247, 3224]
    else:
        seeds_list = seeds

    for seed in seeds_list:
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'parameters.yml'), 'r') as infile:
            hyperparameters = yaml.safe_load(infile)
        p = os.path.join(os.getcwd(),
                        hyperparameters['parent_dir'].split(os.sep)[-3],
                        hyperparameters['parent_dir'].split(os.sep)[-2], 
                        hyperparameters['parent_dir'].split(os.sep)[-1],
                        str(seed))
        print('Get data for plotting', p+'.')
        a = os.path.isfile(os.path.join(p, 'motion_gt.pkl'))
        b = os.path.isfile(os.path.join(p, 'motion_pred.pkl'))
        c = os.path.isfile(os.path.join(p, 'motion_start_timestep.pkl'))
        if a and b and c:
            print('Skip testing.')
        
        pred_traj, gt_traj = {}, {}
        set_random_seeds(int(seed))

        dataset = SAWdataset(hyperparameters, 'testing', False, for_visualization=True)
        dataloader = DataLoader(dataset, batch_size=1,
                                num_workers=8, drop_last=False,
                                sampler=None, shuffle=False, pin_memory=True)
        
        model_state_dict_pth = os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), f'best_siMLPe_traj.pth')

        model = Model(hyperparameters, traj=True)
        state_dict = torch.load(model_state_dict_pth)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.cuda()

        pred_traj[seed], gt_traj[seed] = {}, {}
        
        for datapoints in dataloader:
            for dp in datapoints:
                traj, idx = dp[0], dp[1]
                idx = idx.item()
                pred, gt = test_traj_step(traj, traj, model, mode='visualization')
                if hyperparameters['dataset_name'] == 'real_poppy':
                    pred_traj[seed][idx+1] = np.array(pred.squeeze())
                    gt_traj[seed][idx+1] = np.array(gt.squeeze())
                elif hyperparameters['dataset_name'] == 'vir_poppy':
                    pred_traj[seed][idx] = np.array(pred.squeeze())
                    gt_traj[seed][idx] = np.array(gt.squeeze())    

        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'pred_traj.pkl'), 'wb') as handle:
            pk.dump(pred_traj, handle, protocol=pk.HIGHEST_PROTOCOL)
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'gt_traj.pkl'), 'wb') as handle:
            pk.dump(gt_traj, handle, protocol=pk.HIGHEST_PROTOCOL)
        print('Done.')

def plot_traj(dataset_name, exp_name, seeds=None):

    if not seeds:
        seeds_list = [608, 1247, 3224]
    else:
        seeds_list = seeds

    for seed in seeds_list:
        set_random_seeds(int(seed))

        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'pred_traj.pkl'), 'rb') as handle:
            pred_traj = pk.load(handle)
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'gt_traj.pkl'), 'rb') as handle:
            gt_traj = pk.load(handle)
        with open(os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'intermediate_files', 'video_idx_splits.yml'), 'r') as infile:
            idx_list = yaml.safe_load(infile)['testing_idx']
        plots_path = os.path.join(os.getcwd(), 'exps', dataset_name, exp_name, str(seed), 'plots')
        if not os.path.isdir(plots_path):
            os.makedirs(plots_path)

        if dataset_name == 'RP':
            for rep_idx in idx_list:
                _, axes = plt.subplots(5, 5, figsize=(10,10))
                for axis, j in zip(axes.reshape(25), list(range(1,26))):
                    axis.title.set_text(f'Joint: {j}')
                rep_idx = int(rep_idx)+1
                for i in range(25):
                    axes[int(i//5)][i%5].scatter(range(300), pred_traj[seed][rep_idx][:,i], c='black', s=2)
                    axes[int(i//5)][i%5].scatter(range(300), gt_traj[seed][rep_idx][:,i], c='r', s=2)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_path, 'Rep_{}.pdf'.format(rep_idx)))
        elif dataset_name == 'VP':
            for rep_idx in pred_traj[seed].keys():
                _, axes = plt.subplots(6, 7, figsize=(10,10))
                for axis, j in zip(axes.reshape(42), list(range(42))):
                    axis.ticklabel_format(axis='y', style='sci', scilimits=(-1,-2), useMathText=True)
                    axis.title.set_text(f'Joint: {j}')
                rep_idx = int(rep_idx)
                for i in range(42):
                    axes[int(i//7)][i%7].scatter(range(1354), pred_traj[seed][rep_idx][:,i][:1354], c='black', s=2)
                    axes[int(i//7)][i%7].scatter(range(1354), gt_traj[seed][rep_idx][:,i][:1354], c='r', s=2)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_path, 'Rep_{}.pdf'.format(rep_idx)))


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    
    parser = argparse.ArgumentParser(description='Process user defined hyperparameters')
    parser.add_argument('--dataset', type=str, help='Name the abbreviation dataset that the model was trained on.')
    parser.add_argument('--exp_name', type=str, help='The name of experiment in ./SAW/exps/.')
    parser.add_argument('--ps', type=int, nargs='*', help='The prediction span should be GEQ 1 and LEQ the prediciton span recorded in the parameter.yml')
    parser.add_argument('--seeds_input', nargs='*', help='Your seeds input.')

    args = parser.parse_args('--dataset RP --exp_name ours_siMLPe_pl --ps 12 24 --seeds_input 608'.split())
    # args = parser.parse_args('--dataset RP --exp_name PS_1e-5 --ps 60 --seeds_input 608'.split())
    
    for i in args.ps:
        get_JA_for_visualization(args.dataset, args.exp_name, i, args.seeds_input)
        plot_JA_curves(args.dataset, args.exp_name, i, args.seeds_input)

    # get_traj_for_visualization(args.dataset, args.exp_name, args.seeds_input)
    # plot_traj(args.dataset, args.exp_name, args.seeds_input)