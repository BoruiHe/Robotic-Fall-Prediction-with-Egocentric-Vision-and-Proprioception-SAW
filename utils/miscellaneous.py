import torch
import random
import numpy as np


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

def rgba2rgb(rgba, background=(255,255,255)):
    n, row, col, _ = rgba.shape
    rgb = np.zeros((n, row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,:,0], rgba[:,:,:,1], rgba[:,:,:,2], rgba[:,:,:,3]
    a = np.asarray(a, dtype='float32') / 255

    rgb[:,:,:,0] = r * a + (1.0 - a) * background[0]
    rgb[:,:,:,1] = g * a + (1.0 - a) * background[1]
    rgb[:,:,:,2] = b * a + (1.0 - a) * background[2]

    return np.asarray(rgb, dtype='uint8')

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def MSE_siMLPe(pred, gt):
    mse = torch.nn.MSELoss(reduction='none')(pred, gt)
    return mse.mean(dim=2, keepdim=True).sum(dim=0, keepdim=True).squeeze(), mse.std(dim=2, keepdim=True).sum(dim=0, keepdim=True).squeeze()

def MSE(pred, gt, mode):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    b,c,n,h,w = gt.shape
    if mode == 'mean':
        return loss_fn(pred, gt)/(b*n)
    elif mode == 'sum':
        return loss_fn(pred, gt)/n

def MSE_traj(pred, gt, mode):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    b,_ = gt.shape
    if mode == 'mean':
        return loss_fn(pred, gt)/b
    elif mode == 'sum':
        return loss_fn(pred, gt)

def WeightedNormLoss(pred, gt, mode):

    diff = pred-gt
    b,n,c = diff.shape
    weight_matrix = torch.ones((n,c))
    if c == 25: # RP
        active_joints_index = [0,1,4,8,9,12,17,18,19,22,23,24]
    elif (c == 42*3) or (c == 42): # VP
        active_joints_index = [2,3,4,8,9,10,11,14,15,16,18]

    for i in active_joints_index:
        weight_matrix[:,i] *= 10
    weight_matrix = torch.stack([weight_matrix]*b).cuda()
    
    if mode == 'sum':
        return ((diff**2)*weight_matrix).sum()
    elif mode == 'mean':
        return ((diff**2)*weight_matrix).sum()/b


if __name__ == '__main__':
    gt = torch.ones((2,60,25))
    pred = torch.zeros((2,60,25))
    ave = np.zeros([60])
    pass