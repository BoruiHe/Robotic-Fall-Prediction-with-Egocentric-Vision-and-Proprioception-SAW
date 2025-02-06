import os
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from sklearn import metrics, svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Ours.test import test_siMLPe_step, test_Popeyes_step, test_traj_step, test_quickMLP_step
from Ours.dataset_scenes import SAWdataset
from Ours.model import siMLPe, siMLPe_, Popeyes, Traj, quickMLP
from utils.logger import get_logger, print_and_log_info
from utils.miscellaneous import set_random_seeds, WeightedNormLoss, MSE, MSE_traj, gen_velocity


def update_lr_S(nb_iter, optimizer, dataset) :
    current_lr = optimizer.param_groups[0]['lr']
    if dataset == 'vir_poppy':
        num_steps = 4000
        if (nb_iter+1) > num_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
    elif dataset == 'real_poppy':
        num_steps = 4000
        if (nb_iter+1) > num_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5

    return current_lr

def update_lr_T(nb_iter, optimizer, dataset) :
    current_lr = optimizer.param_groups[0]['lr']
    if dataset == 'vir_poppy':
        num_steps = 10000
        if (nb_iter+1) % num_steps == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
    elif dataset == 'real_poppy':
        num_steps = 10000
        if (nb_iter+1) % num_steps == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8

    return current_lr

def update_lr_P(nb_iter, optimizer, dataset) :
    current_lr = optimizer.param_groups[0]['lr']
    if dataset == 'vir_poppy':
        num_steps = 1000
        if (nb_iter+1) % num_steps == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
    elif dataset == 'real_poppy':
        num_steps = 1000
        if (nb_iter+1) % num_steps == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8

    return current_lr

def train_siMLPe_step(input_joint_motion, gt, model, optimizer, nb_iter, output_seq_length, dataset):

    # # (batch size, seq_len, motion dim)
    b,n,c = gt.shape
    motion_input_ = input_joint_motion.clone().cuda()
    motion_pred = model(motion_input_)[:, :output_seq_length]
    offset = input_joint_motion[:, -1:].cuda()
    motion_pred = motion_pred + offset

    loss = WeightedNormLoss(motion_pred, gt.cuda(), 'mean')
    motion_pred = motion_pred.reshape(b,n,c,1)
    # gt = gt.cuda().reshape(b,n,c,1).reshape(-1,1)
    # loss = torch.mean(torch.norm(motion_pred - gt, 2, 1))
    dmotion_pred = gen_velocity(motion_pred)
    motion_gt = gt.reshape(b,n,c,1).cuda()
    dmotion_gt = gen_velocity(motion_gt)
    dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,1), 1, 1))
    tloss = loss+dloss
    
    optimizer.zero_grad()
    tloss.backward()
    optimizer.step()
    current_lr = update_lr_S(nb_iter, optimizer, dataset)

    return tloss.item(), current_lr

def train_siMLPe(hyperparameters, seed):
    set_random_seeds(seed)

    model = siMLPe(hyperparameters)
    model.train()
    model.cuda()

    dataset = SAWdataset(hyperparameters, 'training', False)
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs_siMLPe'],
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True)
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=hyperparameters['lr'],
                                weight_decay=hyperparameters['weight_decay'])

    log_path = os.path.join(hyperparameters['log_dir_path'], 'siMLPe.log')
    logger, hdlr = get_logger(log_path, 'train')
    print(f'\n***** siMLPe training --- Sequence length: {hyperparameters['N']} *****')
    nb_iter = 0
    avg_lr = 0.
    avg_train_loss = 0.
    best_val_loss = torch.inf
    while (nb_iter + 1) < hyperparameters['total_iterations']:
        for (input_joint_motion, future_joint_motion, _, _) in dataloader:
            loss, current_lr = train_siMLPe_step(input_joint_motion, future_joint_motion, model, optimizer, nb_iter, int(hyperparameters['N']/2), hyperparameters['dataset_name'])
            avg_lr += current_lr
            avg_train_loss += loss

            if (nb_iter + 1) % 100 ==  0:
                avg_lr /= 100
                avg_train_loss /= 100
                val_dataset = SAWdataset(hyperparameters, 'validation', False)
                val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters['bs_siMLPe'],
                                            num_workers=8, drop_last=False,
                                            sampler=None, shuffle=False, pin_memory=True)

                avg_val_loss = 0.
                model.eval()
                for (input_joint_motion_, future_joint_motion_, _, _) in val_dataloader:
                    avg_val_loss += test_siMLPe_step(input_joint_motion_, future_joint_motion_, model, int(hyperparameters['N']/2), mode='validation')
                model.train()
                avg_val_loss /= len(val_dataset)
                if avg_val_loss < best_val_loss:
                    best_model_state_dic = deepcopy(model.state_dict())

                print_and_log_info(logger, 'Iter {} Summary: '.format(nb_iter + 1))
                print_and_log_info(logger, f'\t lr: {avg_lr} \t Training loss: {avg_train_loss} \t Validation loss: {avg_val_loss}')
                avg_lr = 0
                avg_train_loss = 0.

            if (nb_iter + 1) == hyperparameters['total_iterations']:
                break

            nb_iter += 1
    torch.save(best_model_state_dic, os.path.join(hyperparameters['log_dir_path'], 'model_states', 'best_siMLPe.pth'))
    logger.removeHandler(hdlr)

def train_traj_step(input_joint_motion, model, optimizer, nb_iter, dataset) :

    # # (batch size, seq_len, motion dim)
    # b,n,c = future_joint_motion.shape
    motion_input_ = input_joint_motion.flatten(start_dim=1).clone().cuda()
    motion_pred = model(motion_input_)
    loss = MSE_traj(motion_pred, input_joint_motion.flatten(start_dim=1).cuda(), mode='mean')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    current_lr = update_lr_T(nb_iter, optimizer, dataset)

    return loss.item(), current_lr

def train_traj(hyperparameters, seed, ps):
    set_random_seeds(seed)

    model = Traj(ps, hyperparameters)
    model.train()
    model.cuda()

    dataset = SAWdataset(hyperparameters, 'training', False, ps=ps)
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs_Traj'],
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=hyperparameters['lr_Traj'],
                                weight_decay=hyperparameters['weight_decay_Traj'])

    log_path = os.path.join(hyperparameters['log_dir_path'], f'Traj_{ps}.log')
    logger, hdlr = get_logger(log_path, 'train')
    print(f'\n***** Tarj training: {ps} *****')
    nb_iter = 0
    avg_lr = 0.
    avg_train_loss = 0.
    best_val_loss = torch.inf
    while (nb_iter + 1) < hyperparameters['total_iterations_Traj']:
        for (_, _, traj, _) in dataloader:
            loss, current_lr = train_traj_step(traj, model, optimizer, nb_iter, hyperparameters['dataset_name'])
            avg_lr += current_lr
            avg_train_loss += loss

            if (nb_iter + 1) % 100 ==  0:
                avg_lr /= 100
                avg_train_loss /= 100
                val_dataset = SAWdataset(hyperparameters, 'validation', False, ps=ps)
                val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters['bs_Traj'],
                                            num_workers=8, drop_last=False,
                                            sampler=None, shuffle=False, pin_memory=True)

                avg_val_loss = 0.
                model.eval()
                for (_, _, traj, _) in val_dataloader:
                    avg_val_loss += test_traj_step(traj, model, mode='validation')
                model.train()
                avg_val_loss /= len(val_dataset)
                if avg_val_loss < best_val_loss:
                    best_model_state_dic = deepcopy(model.state_dict())

                print_and_log_info(logger, 'Iter {} Summary: '.format(nb_iter + 1))
                print_and_log_info(logger, f'\t lr: {avg_lr} \t Training loss: {avg_train_loss} \t Validation loss: {avg_val_loss}')
                avg_lr = 0
                avg_train_loss = 0.

            if (nb_iter + 1) == hyperparameters['total_iterations_Traj']:
                break

            nb_iter += 1
    torch.save(best_model_state_dic, os.path.join(hyperparameters['log_dir_path'], 'model_states', f'best_Traj_{ps}.pth'))
    logger.removeHandler(hdlr)

def train_Popeyes_step(input_frames, model, optimizer, nb_iter, dataset):
    frames_input = input_frames.cuda()
    reconstructed_frames = model(frames_input)
    loss = MSE(reconstructed_frames, frames_input, 'mean')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    current_lr = update_lr_P(nb_iter, optimizer, dataset)

    return loss.item(), current_lr

def train_Popeyes(hyperparameters, seed):
    set_random_seeds(seed)
    hyperparameters['seed'] = seed

    model = Popeyes(hyperparameters['N'], hyperparameters['img_height'], hyperparameters['img_width'], hyperparameters['latent_size'])
    model.train()
    model.cuda()

    dataset = SAWdataset(hyperparameters, 'training', True)
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs_Popeyes'],
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True)
    
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr_Popeyes'], weight_decay=hyperparameters['weight_decay_Popeyes'])

    log_path = os.path.join(hyperparameters['log_dir_path'], 'Popeyes.log')
    logger, hdlr = get_logger(log_path, 'train')
    print(f'\n***** Popeyes training --- Sequence length: {hyperparameters['N']} *****')
    nb_iter = 0
    avg_lr = 0.
    avg_train_loss = 0.
    best_val_loss = torch.inf
    while (nb_iter + 1) < hyperparameters['total_iterations_Popeyes']:
        for (_, _, _, frame, _) in dataloader:
            loss, current_lr = train_Popeyes_step(frame, model, optimizer, nb_iter, hyperparameters['dataset_name'])
            avg_lr += current_lr
            avg_train_loss += loss

            if (nb_iter + 1) % 100 ==  0:
                avg_lr /= 100
                avg_train_loss /= 100

                val_dataset = SAWdataset(hyperparameters, 'validation', True)
                val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters['bs_Popeyes'],
                                            num_workers=8, drop_last=False,
                                            sampler=None, shuffle=False, pin_memory=True)

                avg_val_loss = 0.
                model.eval()
                for (_, _, _, frame_, _) in val_dataloader:
                    avg_val_loss += test_Popeyes_step(frame_, model, mode='validation')
                model.train()
                avg_val_loss /= len(val_dataset)
                if avg_val_loss < best_val_loss:
                    best_model_state_dic = deepcopy(model.state_dict())
                
                print_and_log_info(logger, 'Iter {} Summary: '.format(nb_iter + 1))
                print_and_log_info(logger, f'\t lr: {avg_lr} \t Training loss: {avg_train_loss} \t Validation loss: {avg_val_loss}')
                avg_lr = 0
                avg_train_loss = 0.

            if (nb_iter + 1) == hyperparameters['total_iterations_Popeyes']:
                break

            nb_iter += 1
    torch.save(best_model_state_dic, os.path.join(hyperparameters['log_dir_path'], 'model_states', 'best_Popeyes.pth'))
    logger.removeHandler(hdlr)

def svc_S(hyperparameters, seed, siMLPe_weights, prediction_span):
    set_random_seeds(seed)
    # if hyperparameters['dataset_name'] == 'vir_poppy':
    #     bs = 4
    # else:
    bs = hyperparameters['bs_siMLPe']
    # feature extraction for SVC
    model_s = siMLPe_(hyperparameters)
    state_dict_s = torch.load(siMLPe_weights, map_location=torch.device('cpu'))
    model_s.load_state_dict(state_dict_s, strict=False)
    model_s.eval()
    model_s.cuda()

    features_1, labels = [], []

    dataset = SAWdataset(hyperparameters, 'training', False, ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (input_joint_motion, future_joint_motion, _, label) in dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', False, ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=False,
                                sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, _, label) in val_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', False, ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=False,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, _, label) in test_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation + Testing] features and labels: {len(features_1)}, {len(labels)}, {len(test_dataset)}')
    print(f'Data balance: {(np.asarray(labels) == 0).sum()} walking VCs, {(np.asarray(labels) == 1).sum()} falling VCs\n')

    features = features_1

    if (len(features) > 50000) and (len(labels) > 50000):
        features, labels = np.array(features), np.array(labels)
        idx = np.random.choice(len(features), 50000, replace=False)
        features, labels = features[idx], labels[idx]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_S.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=5e-3)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = metrics.classification_report(y_test, predicted, digits=4)
        f.write(f'***** {'LinearSVC'} *****\n')
        f.write(report)
        f.write('\n')

def svc_P(hyperparameters, seed, Popeyes_weights, prediction_span):
    set_random_seeds(seed)
    # if hyperparameters['dataset_name'] == 'vir_poppy':
    #     bs = 4
    # else:
    bs = hyperparameters['bs_Popeyes']
    # feature extraction for SVC
    model_p = Popeyes(hyperparameters['N'], hyperparameters['img_height'], hyperparameters['img_width'], hyperparameters['latent_size'])
    model_p.return_features()
    state_dict_p = torch.load(Popeyes_weights, map_location=torch.device('cpu'))
    model_p.load_state_dict(state_dict_p, strict=False)
    model_p.eval()
    model_p.cuda()

    features_1, labels = [], []

    dataset = SAWdataset(hyperparameters, 'training', True, ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (_, _, _, frame, label) in dataloader:
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        features_1.extend([img_repre[i,:].flatten().numpy() for i in range(img_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', True, ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=False,
                                sampler=None, shuffle=True, pin_memory=True)
    for (_, _, _, frame, label) in val_dataloader:
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        features_1.extend([img_repre[i,:].flatten().numpy() for i in range(img_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', True, ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=False,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (_, _, _, frame, label) in test_dataloader:
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        features_1.extend([img_repre[i,:].flatten().numpy() for i in range(img_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation + Testing] features and labels: {len(features_1)}, {len(labels)}, {len(test_dataset)}')
    print(f'Data balance: {(np.asarray(labels) == 0).sum()} walking VCs, {(np.asarray(labels) == 1).sum()} falling VCs\n')

    features = features_1

    if (len(features) > 50000) and (len(labels) > 50000):
        features, labels = np.array(features), np.array(labels)
        idx = np.random.choice(len(features), 50000, replace=False)
        features, labels = features[idx], labels[idx]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_P.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=0.03)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = metrics.classification_report(y_test, predicted, digits=4)
        f.write(f'***** {'LinearSVC'} *****\n')
        f.write(report)
        f.write('\n')

def svc_SP(hyperparameters, seed, siMLPe_weights, Popeyes_weights, prediction_span):
    set_random_seeds(seed)
    if hyperparameters['dataset_name'] == 'vir_poppy':
        bs = 4
    else:
        bs = hyperparameters['bs_Popeyes']
    # feature extraction for SVC
    model_s = siMLPe_(hyperparameters)
    state_dict_s = torch.load(siMLPe_weights, map_location=torch.device('cpu'))
    model_s.load_state_dict(state_dict_s, strict=False)
    model_s.eval()
    model_s.cuda()

    model_p = Popeyes(hyperparameters['N'], hyperparameters['img_height'], hyperparameters['img_width'], hyperparameters['latent_size'])
    model_p.return_features()
    state_dict_p = torch.load(Popeyes_weights, map_location=torch.device('cpu'))
    model_p.load_state_dict(state_dict_p, strict=False)
    model_p.eval()
    model_p.cuda()

    features_1, features_2, labels = [], [], []

    dataset = SAWdataset(hyperparameters, 'training', True, ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (input_joint_motion, future_joint_motion, _, frame, label) in dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:].flatten().numpy() for i in range(img_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', True, ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=False,
                                sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, _, frame, label) in val_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:].flatten().numpy() for i in range(img_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', True, ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=False,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, _, frame, label) in test_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:].flatten().numpy() for i in range(img_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation + Testing] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(test_dataset)}')
    print(f'Data balance: {(np.asarray(labels) == 0).sum()} walking VCs, {(np.asarray(labels) == 1).sum()} falling VCs\n')

    features = [np.concatenate((f1, f2)) for f1, f2 in zip(features_1, features_2)]

    if (len(features) > 50000) and (len(labels) > 50000):
        features, labels = np.array(features), np.array(labels)
        idx = np.random.choice(len(features), 50000, replace=False)
        features, labels = features[idx], labels[idx]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_SP_JA.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=5e-3)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = metrics.classification_report(y_test, predicted, digits=4)
        f.write(f'***** {'LinearSVC'} *****\n')
        f.write(report)
        f.write('\n')

def svc_ST(hyperparameters, seed, siMLPe_weights, traj_weights, prediction_span):
    set_random_seeds(seed)
    # if hyperparameters['dataset_name'] == 'vir_poppy':
    #     bs = 4
    # else:
    bs = hyperparameters['bs_Traj']
    # feature extraction for SVC
    model_s = siMLPe_(hyperparameters)
    state_dict_s = torch.load(siMLPe_weights, map_location=torch.device('cpu'))
    model_s.load_state_dict(state_dict_s, strict=False)
    model_s.eval()
    model_s.cuda()

    model_t = Traj(prediction_span, hyperparameters, featuer_extraction=True)
    state_dict_t = torch.load(traj_weights, map_location=torch.device('cpu'))
    model_t.load_state_dict(state_dict_t, strict=False)
    model_t.eval()
    model_t.cuda()

    features_1, features_2, labels = [], [], []

    dataset = SAWdataset(hyperparameters, 'training', False, ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (input_joint_motion, future_joint_motion, traj, label) in dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', False, ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=False,
                                sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, traj, label) in val_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', False, ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=False,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, traj, label) in test_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation + Testing] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(test_dataset)}')
    print(f'Data balance: {(np.asarray(labels) == 0).sum()} walking VCs, {(np.asarray(labels) == 1).sum()} falling VCs\n')

    features = [np.concatenate((f1, f2)) for f1, f2 in zip(features_1, features_2)]

    if (len(features) > 50000) and (len(labels) > 50000):
        features, labels = np.array(features), np.array(labels)
        idx = np.random.choice(len(features), 50000, replace=False)
        features, labels = features[idx], labels[idx]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_ST.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=5e-3)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = metrics.classification_report(y_test, predicted, digits=4)
        f.write(f'***** {'LinearSVC'} *****\n')
        f.write(report)
        f.write('\n')

def svc_PT(hyperparameters, seed, Popeyes_weights, Traj_weights, prediction_span):
    set_random_seeds(seed)
    # if hyperparameters['dataset_name'] == 'vir_poppy':
    #     bs = 4
    # else:
    bs = hyperparameters['bs_Popeyes']

    model_p = Popeyes(hyperparameters['N'], hyperparameters['img_height'], hyperparameters['img_width'], hyperparameters['latent_size'])
    model_p.return_features()
    state_dict_p = torch.load(Popeyes_weights, map_location=torch.device('cpu'))
    model_p.load_state_dict(state_dict_p, strict=False)
    model_p.eval()
    model_p.cuda()

    model_t = Traj(prediction_span, hyperparameters, featuer_extraction=True)
    state_dict_t = torch.load(Traj_weights, map_location=torch.device('cpu'))
    model_t.load_state_dict(state_dict_t, strict=False)
    model_t.eval()
    model_t.cuda()

    features_1, features_2, labels = [], [], []

    dataset = SAWdataset(hyperparameters, 'training', True, ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (_, _, traj, frame, label) in dataloader:
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([img_repre[i,:,:,:,:].flatten().numpy() for i in range(img_repre.shape[0])])
        features_2.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', True, ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=False,
                                sampler=None, shuffle=True, pin_memory=True)
    for (_, _, traj, frame, label) in val_dataloader:
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([img_repre[i,:,:,:,:].flatten().numpy() for i in range(img_repre.shape[0])])
        features_2.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', True, ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=False,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (_, _, traj, frame, label) in test_dataloader:
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([img_repre[i,:,:,:,:].flatten().numpy() for i in range(img_repre.shape[0])])
        features_2.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation + Testing] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(test_dataset)}')
    print(f'Data balance: {(np.asarray(labels) == 0).sum()} walking VCs, {(np.asarray(labels) == 1).sum()} falling VCs\n')

    features = [np.concatenate((f1, f2)) for f1, f2 in zip(features_1, features_2)]

    if (len(features) > 50000) and (len(labels) > 50000):
        features, labels = np.array(features), np.array(labels)
        idx = np.random.choice(len(features), 50000, replace=False)
        features, labels = features[idx], labels[idx]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_PT.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=0.03)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = metrics.classification_report(y_test, predicted, digits=4)
        f.write(f'***** {'LinearSVC'} *****\n')
        f.write(report)
        f.write('\n')

def svc_SPT(hyperparameters, seed, siMLPe_weights, Popeyes_weights, Traj_weights, prediction_span):
    set_random_seeds(seed)
    # if hyperparameters['dataset_name'] == 'vir_poppy':
    #     bs = 4
    # else:
    bs = hyperparameters['bs_Popeyes']
    # feature extraction for SVC
    model_s = siMLPe_(hyperparameters)
    state_dict_s = torch.load(siMLPe_weights, map_location=torch.device('cpu'))
    model_s.load_state_dict(state_dict_s, strict=False)
    model_s.eval()
    model_s.cuda()

    model_p = Popeyes(hyperparameters['N'], hyperparameters['img_height'], hyperparameters['img_width'], hyperparameters['latent_size'])
    model_p.return_features()
    state_dict_p = torch.load(Popeyes_weights, map_location=torch.device('cpu'))
    model_p.load_state_dict(state_dict_p, strict=False)
    model_p.eval()
    model_p.cuda()

    model_t = Traj(prediction_span, hyperparameters, featuer_extraction=True)
    state_dict_t = torch.load(Traj_weights, map_location=torch.device('cpu'))
    model_t.load_state_dict(state_dict_t, strict=False)
    model_t.eval()
    model_t.cuda()

    features_1, features_2, features_3, labels = [], [], [], []

    dataset = SAWdataset(hyperparameters, 'training', True, ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (input_joint_motion, future_joint_motion, traj, frame, label) in dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:,:,:,:].flatten().numpy() for i in range(img_repre.shape[0])])
        features_3.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(features_2)}, {len(features_3)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', True, ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=False,
                                sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, traj, frame, label) in val_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:,:,:,:].flatten().numpy() for i in range(img_repre.shape[0])])
        features_3.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(features_2)}, {len(features_3)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', True, ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=False,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, traj, frame, label) in test_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, prediction_span, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:,:,:,:].flatten().numpy() for i in range(img_repre.shape[0])])
        features_3.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation + Testing] features and labels: {len(features_1)}, {len(features_2)}, {len(features_3)}, {len(labels)}, {len(test_dataset)}')
    print(f'Data balance: {(np.asarray(labels) == 0).sum()} walking VCs, {(np.asarray(labels) == 1).sum()} falling VCs\n')

    features = [np.concatenate((f1, f2, f3)) for f1, f2, f3 in zip(features_1, features_2, features_3)]

    if (len(features) > 50000) and (len(labels) > 50000):
        features, labels = np.array(features), np.array(labels)
        idx = np.random.choice(len(features), 50000, replace=False)
        features, labels = features[idx], labels[idx]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed, shuffle=True)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_SPT_JA.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=5e-3)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = metrics.classification_report(y_test, predicted, digits=4)
        f.write(f'***** {'LinearSVC'} *****\n')
        f.write(report)
        f.write('\n')

def update_lr_Q(nb_iter, optimizer, dataset) :
    current_lr = optimizer.param_groups[0]['lr']
    if dataset == 'vir_poppy':
        num_steps = 200
        if (nb_iter+1) % num_steps == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
    elif dataset == 'real_poppy':
        num_steps = 100
        if (nb_iter+1) % num_steps == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    return current_lr

def train_quickMLP_step(input_joint_motion, traj, frame, label, model, optimizer, nb_iter, dataset) :

    # # (batch size, seq_len, motion dim)
    # b,n,c = future_joint_motion.shape
    motion_input_ = input_joint_motion.flatten(start_dim=1).clone()
    traj_input_ = traj.flatten(start_dim=1).clone()
    frame_input_ = frame.clone()
    # input_ = torch.cat((motion_input_, traj_input_, frame_input_), dim=1)
    pred = model(motion_input_.cuda(), traj_input_.cuda(), frame_input_.cuda())
    loss_fn = torch.nn.BCELoss(reduction='none')
    loss = loss_fn(pred, label.float().reshape(-1,1).cuda())
    
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()
    current_lr = update_lr_Q(nb_iter, optimizer, dataset)

    return loss.cpu(), current_lr

def train_quickMLP(hyperparameters, seed, ps):
    set_random_seeds(seed)

    model = quickMLP(ps, hyperparameters)
    model.train()
    model.cuda()

    dataset = SAWdataset(hyperparameters, 'training', True, ps=ps)
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs'],
                            num_workers=8, drop_last=False,
                            sampler=None, shuffle=True, pin_memory=True)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=hyperparameters['lr'],
                                weight_decay=hyperparameters['weight_decay'])

    log_path = os.path.join(hyperparameters['log_dir_path'], f'quickMLP_{ps}.log')
    logger, hdlr = get_logger(log_path, 'train')
    print(f'\n***** qucikMLP training: {ps} *****')
    nb_iter = 0
    avg_lr = 0.
    avg_train_loss = 0.
    best_val_loss = torch.inf
    while (nb_iter + 1) < hyperparameters['total_iterations']:
        for (input_joint_motion, _, traj, frame, label) in dataloader:
            loss, current_lr = train_quickMLP_step(input_joint_motion, traj, frame, label, model, optimizer, nb_iter, hyperparameters['dataset_name'])
            avg_lr += current_lr
            avg_train_loss += loss.sum()

            if (nb_iter + 1) % 100 ==  0:
                avg_lr /= 100
                avg_train_loss /= 100
                val_dataset = SAWdataset(hyperparameters, 'validation', True, ps=ps)
                val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters['bs'],
                                            num_workers=8, drop_last=False,
                                            sampler=None, shuffle=False, pin_memory=True)

                avg_val_loss = 0.
                model.eval()
                for (input_joint_motion, _, traj, frame, label) in val_dataloader:
                    avg_val_loss += test_quickMLP_step(input_joint_motion, traj, frame, label, model, 'validation')
                model.train()
                avg_val_loss /= len(val_dataset)
                if avg_val_loss < best_val_loss:
                    best_model_state_dic = deepcopy(model.state_dict())

                print_and_log_info(logger, 'Iter {} Summary: '.format(nb_iter + 1))
                print_and_log_info(logger, f'\t lr: {avg_lr} \t Training loss: {avg_train_loss} \t Validation loss: {avg_val_loss}')
                avg_lr = 0
                avg_train_loss = 0.

            if (nb_iter + 1) == hyperparameters['total_iterations']:
                break

            nb_iter += 1
    torch.save(best_model_state_dic, os.path.join(hyperparameters['log_dir_path'], 'model_states', f'best_quickMLP_{ps}.pth'))
    logger.removeHandler(hdlr)