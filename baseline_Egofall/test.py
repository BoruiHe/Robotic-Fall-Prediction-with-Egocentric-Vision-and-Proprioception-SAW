import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import pickle as pk
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader
from utils.miscellaneous import set_random_seeds
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import cross_validate


def test_Egofall_step(video_clips, model, mode):
    if mode == 'iter':
        # bs, N, c, h, w = video_clips.shape -> (8,10,3,224,224)
        res_features = []
        with torch.no_grad():
            '''
            Here, frames is a tensor of shape (N,3,h,w). (10,3,224,224)
            Resnet50 will regard N as batch size but the coherence between adject frames means nothing to a 2D CNN.
            It should be fine as long as res_feature are concatenated properly.
            '''
            for frames in video_clips:
                res_feature = model(frames.cuda()).cpu()
                res_features.append(res_feature.flatten().numpy()) # (N,2048)->(20480,)->[(20480,)*8]
            # out = torch.stack(res_features, dim=0) # (8,20480)
        return res_features
    elif mode == 'flat':
        '''
        What if merge dimensions instead of iterate over a mini batch?
        Might be worth further discussion or even research.
        '''
        with torch.no_grad():
            ipt = video_clips.reshape(-1,3,224,224)
            res_features = model(ipt.cuda()).cpu()
            
        return res_features.reshape(-1,10,2048).reshape(-1,20480).numpy()
    else:
        raise ValueError('Mode not implemented.')

def test_Egofall_backbone(HpParams, seed, task_mode, ps=None):
    dataset_name = HpParams['dataset_name']
    print(f'\n***** Egofall on {dataset_name.upper()} with seed {seed} *****')
    set_random_seeds(seed)
    resnet= resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    resnet.cuda()

    # dataset
    if task_mode == 'ps':
        file_name = f'feats_labs_{ps}'
        from baseline_Egofall.dataset_ps import Egofalldataset
        dataset = Egofalldataset(HpParams, ps)
    else:
        file_name = 'feats_labs'
        from baseline_Egofall.dataset import Egofalldataset
        dataset = Egofalldataset(HpParams)
    print(f'dataset: {len(dataset)}')
    dataloader = DataLoader(dataset, batch_size=HpParams['bs'],
                                     num_workers=8, drop_last=False,
                                     sampler=None, shuffle=True, pin_memory=True)

    features_i, labels_i = [], []
    features_f, labels_f = [], []

    for frames, label, _ in dataloader: # frames: [bs, N, 3, h, w]
        for step_mode in ['iter', 'flat']:
            res_features = test_Egofall_step(frames, resnet, step_mode)
            if step_mode == 'iter':
                features_i.extend(res_features)
                labels_i.append(label.numpy())
            elif step_mode == 'flat':
                features_f.extend(res_features)
                labels_f.append(label.numpy())

    features_i = np.array(features_i)
    labels_i = np.concatenate(labels_i)
    print(features_i.shape, labels_i.shape)
    np.savez(os.path.join(HpParams['parent_dir'], str(seed), file_name+'_iter.npz'), feats=features_i, labels=labels_i)

    features_f = np.array(features_f)
    labels_f = np.concatenate(labels_f)
    print(features_f.shape, labels_f.shape)
    np.savez(os.path.join(HpParams['parent_dir'], str(seed), file_name+'_flat.npz'), feats=features_f, labels=labels_f)

def test_Egofall_clf(HpParams, score, seed, task_mode, ps=None):
    if ps: print(f'{ps}:')
    set_random_seeds(seed)
    for mode in ['iter', 'flat']:
        print(f'{mode.upper()}:')
        if task_mode == 're':
            thing = np.load(os.path.join(HpParams['parent_dir'], str(seed), f'feats_labs_{mode}.npz'), allow_pickle=True)
        else:
            thing = np.load(os.path.join(HpParams['parent_dir'], str(seed), f'feats_labs_{ps}_{mode}.npz'), allow_pickle=True)
        features = thing['feats']
        labels = thing['labels']
        print(f'features vector shape:{features.shape}, labels vector shpae: {labels.shape}')
        clf = LinearSVC(dual='auto', C=0.19, class_weight='balanced', random_state=seed)
        # Performance on RP:
        # RobusScaler           -> converge fails 6 times           -> ITER: 0.425, FLAT: 0.425
        # QuantileTransformer   -> converge fails too many times    -> ITER: 0.475, FLAT: 0.475
        # Normalizer            -> converge fails 0 times           -> ITER: 0.508, FLAT: 0.508
        # StandardScaler        -> converge fails 0 times           -> ITER: 0.425, FLAT: 0.425
        pipe = Pipeline([('scaler', Normalizer()), ('estimator', clf)])
        results = cross_validate(pipe, features, labels, scoring=('accuracy', 'balanced_accuracy'), cv=HpParams['k_fold'])
        print('testing accuracy: {}; testing balanced accuracy: {}.'.format(round(results['test_accuracy'].mean()*100, 2), round(results['test_balanced_accuracy'].mean()*100, 2)))
        print(results['test_balanced_accuracy'])
 
        score[seed][mode].append((ps, results['test_balanced_accuracy'].mean()))

    with open(os.path.join(HpParams['parent_dir'], 'score.pkl'), 'wb') as f:
        pk.dump(score, f)
    pass
