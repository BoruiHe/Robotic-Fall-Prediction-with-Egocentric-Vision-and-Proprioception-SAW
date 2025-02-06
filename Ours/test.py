import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from Ours.dataset import SAWdataset
from Ours.model import siMLPe, Popeyes, Traj, quickMLP
from utils.logger import get_logger, print_and_log_info
from utils.miscellaneous import set_random_seeds, WeightedNormLoss, MSE, MSE_traj, MSE_siMLPe


def test_siMLPe_step(input_joint_motion, gt, model, output_seq_length, mode):
    motion_input = input_joint_motion.cuda()
    with torch.no_grad():
        motion_input_ = motion_input.clone()
        output = model(motion_input_.cuda())
        motion_pred = output[:, :output_seq_length, :]
        motion_pred = motion_pred.detach().cpu()
        motion_gt = gt.detach().cpu()
        offset = input_joint_motion[:, -1:].cpu()
        motion_pred += offset

    if mode == 'visualization': # return prediction and ground truth for plotting JA curves
        return motion_pred, motion_gt
    
    elif mode == 'validation': # return validation loss defined by loss_fn
        loss = WeightedNormLoss(motion_pred.cuda(), gt.cuda(), 'sum')
        return loss.detach().cpu()
    
    elif mode == 'testing': # Mean absolute error and the std of error
        mean, std = MSE_siMLPe(motion_pred.cuda(), motion_gt.cuda())
        return mean.detach().cpu().numpy(), std.detach().cpu().numpy()
    
    elif mode == 'feature_extraction':
        return motion_pred

def test_siMLPe(hyperparameters, seed, model_state_dict_pth, prediction_span):
    print(f'***** Testing on prediction span: {prediction_span} *****')
    set_random_seeds(seed)

    dataset = SAWdataset(hyperparameters, 'testing', False, ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs_siMLPe'],
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=False, pin_memory=True)

    model = siMLPe(hyperparameters)
    state_dict = torch.load(model_state_dict_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    mse_ave, std_ave = np.zeros([prediction_span]), np.zeros([prediction_span])
    for (input_joint_motion, future_joint_motion, _, _) in dataloader:
        mean, std = test_siMLPe_step(input_joint_motion, future_joint_motion, model, prediction_span, 'testing')
        mse_ave += mean
        std_ave += std

    mse_ave = mse_ave / len(dataset)
    std_ave = std_ave / len(dataset)
    log_path = os.path.join(hyperparameters['log_dir_path'], 'siMLPe.log')
    logger, hdlr = get_logger(log_path, 'test')
    print_and_log_info(logger, f'\nps: {prediction_span}\n{mse_ave}')
    logger.removeHandler(hdlr)

    logger_std, hdlr_std = get_logger(os.path.join(hyperparameters['log_dir_path'], 'siMLPe_std.log'), 'test')
    print_and_log_info(logger_std, f'\nps: {prediction_span}\n{std_ave}')
    logger_std.removeHandler(hdlr_std)

def test_traj_step(input_joint_motion, model, mode):
    with torch.no_grad():
        motion_input_ = input_joint_motion.flatten(start_dim=1).clone().cuda()
        output = model(motion_input_.cuda())
        motion_pred = output

    if mode == 'visualization': # return prediction and ground truth for plotting JA curves
        return motion_pred, input_joint_motion
    
    elif mode == 'validation': # return validation loss defined by loss_fn
        loss = MSE_traj(motion_pred.cuda(), input_joint_motion.flatten(start_dim=1).cuda(), mode='sum')
        return loss.detach().cpu()
    
    elif mode == 'testing': # Mean absolute error and the std of error
        mse = np.array(MSE_traj(motion_pred.cuda(), input_joint_motion.flatten(start_dim=1).cuda(), mode='sum').cpu())
        return mse
    
    elif mode == 'feature_extraction':
        return motion_pred.cpu()

def test_traj(hyperparameters, seed, model_state_dict_path, ps):
    print(f'***** siMLPe_traj testing *****')
    set_random_seeds(seed)

    dataset = SAWdataset(hyperparameters, 'testing', False, ps=ps)
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs_Traj'],
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=False, pin_memory=True)

    model = Traj(ps, hyperparameters)
    state_dict = torch.load(model_state_dict_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    mse_ave = 0
    for (_, _, traj, _) in dataloader:
    # for traj in dataloader:
        mse = test_traj_step(traj, model, 'testing')
        mse_ave += mse
    
    mse_ave = mse_ave / len(dataset)

    log_path = os.path.join(hyperparameters['log_dir_path'], f'Traj_{ps}.log')
    logger, hdlr = get_logger(log_path, 'test')
    print_and_log_info(logger, f'\nAverage MSE on {ps}\n{mse_ave}')
    logger.removeHandler(hdlr)

def test_Popeyes_step(input_frames, model, mode):
    with torch.no_grad():
        frames_input = input_frames.cuda()
        reconstructed_frames_features = model(frames_input)

    if mode == 'visualization':
        return reconstructed_frames_features.permute(0,2,3,4,1), input_frames
    elif mode == 'feature_extraction':
        return reconstructed_frames_features.detach().cpu()
    elif mode == 'validation' or mode == 'testing':
        loss = MSE(reconstructed_frames_features, frames_input, 'sum')
        return loss.detach().cpu()

def test_Popeyes(hyperparameters, seed, model_state_dict_pth, prediction_span):
    print(f'***** Testing on prediction span: {prediction_span} *****')
    set_random_seeds(seed)

    dataset = SAWdataset(hyperparameters, 'testing', True, ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs_Popeyes'],
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=False, pin_memory=True)

    model = Popeyes(hyperparameters['N'], hyperparameters['img_height'], hyperparameters['img_width'], hyperparameters['latent_size'])
    state_dict = torch.load(model_state_dict_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    ave_loss = 0
    for (_, _, _, frame, _) in dataloader:
        loss = test_Popeyes_step(frame, model, 'testing')
        ave_loss += loss

    ave_loss = ave_loss / len(dataset)
    
    log_path = os.path.join(hyperparameters['log_dir_path'], 'Popeyes.log')
    logger, hdlr = get_logger(log_path, 'test')
    print_and_log_info(logger, f'\nps: {prediction_span}\n{ave_loss}')
    logger.removeHandler(hdlr)

def test_quickMLP_step(input_joint_motion, traj, frame, label, model, mode):
    with torch.no_grad():
        motion_input_ = input_joint_motion.flatten(start_dim=1).clone()
        traj_input_ = traj.flatten(start_dim=1).clone()
        frame_input_ = frame.clone()
        # input_ = torch.cat((motion_input_, traj_input_, frame_input_), dim=1)
        pred = model(motion_input_.cuda(), traj_input_.cuda(), frame_input_.cuda())

    if mode == 'validation':
        loss_fn = torch.nn.BCELoss(reduction='sum')
        loss = loss_fn(pred, label.float().reshape(-1,1).cuda())
        return loss
    else:
        label_ = label.reshape(-1,1).clone()
        return (pred.cpu().round() == label_).sum()

def test_quickMLP(hyperparameters, seed, model_state_dict_path, ps):
    print(f'***** quickMLP testing *****')
    set_random_seeds(seed)

    dataset = SAWdataset(hyperparameters, 'testing', True, ps=ps)
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs'],
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=False, pin_memory=True)

    model = quickMLP(ps, hyperparameters)
    state_dict = torch.load(model_state_dict_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    acc = 0
    for (input_joint_motion, _, traj, frame, label) in dataloader:
        acc += test_quickMLP_step(input_joint_motion, traj, frame, label, model, 'testing')
    
    acc = acc / len(dataset)

    log_path = os.path.join(hyperparameters['log_dir_path'], f'quickMLP_{ps}.log')
    logger, hdlr = get_logger(log_path, 'test')
    print_and_log_info(logger, f'\nAverage Acc on {ps}\n{acc}')
    logger.removeHandler(hdlr)