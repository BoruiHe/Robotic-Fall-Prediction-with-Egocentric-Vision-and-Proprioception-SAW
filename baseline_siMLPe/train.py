import os
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from sklearn import metrics, svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from baseline_siMLPe.test import test_siMLPe_step, test_Popeyes_step, test_traj_step
from baseline_siMLPe.dataset import SAWdataset
from baseline_siMLPe.model import siMLPe, siMLPe_, Popeyes, Traj
from utils.logger import get_logger, print_and_log_info
from utils.miscellaneous import set_random_seeds


def update_lr_multistep(nb_iter, optimizer) :
    if nb_iter > 7500:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_siMLPe_step(input_joint_motion, gt, model, optimizer, nb_iter):
    b, input_seq_length, _, _ = input_joint_motion.shape
    input_ = input_joint_motion.reshape(b, input_seq_length, -1).clone()
    c = input_.shape[-1]
    motion_pred = model(input_.cuda())
    output_seq_length = gt.shape[1]
    offset = input_[:, -1:].cuda()
    motion_pred = motion_pred[:, :output_seq_length] + offset
    n = gt.shape[1]
    motion_pred = motion_pred.reshape(b,n,c,1).reshape(-1,1)
    gt = gt.reshape(b, output_seq_length, -1).cuda().reshape(b,n,c,1).reshape(-1,1)
    loss = torch.mean(torch.norm(motion_pred - gt, 2, 1))

    motion_pred = motion_pred.reshape(b,n,c,1)
    dmotion_pred = gen_velocity(motion_pred)
    motion_gt = gt.reshape(b,n,c,1)
    dmotion_gt = gen_velocity(motion_gt)
    dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,1), 2, 1))
    loss = loss + dloss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, optimizer)

    return loss.item(), optimizer, current_lr

def train_siMLPe(hyperparameters, seed):
    set_random_seeds(seed)

    model = siMLPe(hyperparameters)
    model.train()
    model.cuda()

    dataset = SAWdataset(hyperparameters, 'training', False, data_type=hyperparameters['data_type'])
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs_siMLPe'],
                            num_workers=8, drop_last=True,
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
            loss, optimizer, current_lr = train_siMLPe_step(input_joint_motion, future_joint_motion, model, optimizer, nb_iter)
            avg_lr += current_lr
            avg_train_loss += loss

            if (nb_iter + 1) % 100 ==  0:
                avg_lr /= 100
                avg_train_loss /= 100
                val_dataset = SAWdataset(hyperparameters, 'validation', False, data_type=hyperparameters['data_type'])
                val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters['bs_siMLPe'],
                                            num_workers=8, drop_last=True,
                                            sampler=None, shuffle=False, pin_memory=True)

                avg_val_loss = 0.
                model.eval()
                for (input_joint_motion_, future_joint_motion_, _, _) in val_dataloader:
                    avg_val_loss += test_siMLPe_step(input_joint_motion_, future_joint_motion_, model, mode='validation')
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

def svc_S(hyperparameters, seed, siMLPe_weights, prediction_span):
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

    features_1, labels = [], []

    dataset = SAWdataset(hyperparameters, 'training', False, data_type=hyperparameters['data_type'], ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=True,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (input_joint_motion, future_joint_motion, _, label) in dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', False, data_type=hyperparameters['data_type'], ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=True,
                                sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, _, label) in val_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', False, data_type=hyperparameters['data_type'], ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=True,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, _, label) in test_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
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
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_S_WTF5e-3.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=5e-3)
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

    dataset = SAWdataset(hyperparameters, 'training', True, data_type=hyperparameters['data_type'], ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=True,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (input_joint_motion, future_joint_motion, _, frame, label) in dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:].flatten().numpy() for i in range(img_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', True, data_type=hyperparameters['data_type'], ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=True,
                                sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, _, frame, label) in val_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:].flatten().numpy() for i in range(img_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', True, data_type=hyperparameters['data_type'], ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=True,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, _, frame, label) in test_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
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
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_SP.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=5e-5)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = metrics.classification_report(y_test, predicted, digits=4)
        f.write(f'***** {'LinearSVC'} *****\n')
        f.write(report)
        f.write('\n')

def svc_ST(hyperparameters, seed, siMLPe_weights, traj_weights, prediction_span):
    set_random_seeds(seed)
    if hyperparameters['dataset_name'] == 'vir_poppy':
        bs = 4
    else:
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

    dataset = SAWdataset(hyperparameters, 'training', False, data_type=hyperparameters['data_type'], ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=True,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (input_joint_motion, future_joint_motion, traj, label) in dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', False, data_type=hyperparameters['data_type'], ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=True,
                                sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, traj, label) in val_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(features_2)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', False, data_type=hyperparameters['data_type'], ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=True,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, traj, label) in test_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
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
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_ST_WTF.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=5e-3)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = metrics.classification_report(y_test, predicted, digits=4)
        f.write(f'***** {'LinearSVC'} *****\n')
        f.write(report)
        f.write('\n')

def svc_SPT(hyperparameters, seed, siMLPe_weights, Popeyes_weights, Traj_weights, prediction_span):
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

    model_t = Traj(prediction_span, hyperparameters, featuer_extraction=True)
    state_dict_t = torch.load(Traj_weights, map_location=torch.device('cpu'))
    model_t.load_state_dict(state_dict_t, strict=False)
    model_t.eval()
    model_t.cuda()

    features_1, features_2, features_3, labels = [], [], [], []

    dataset = SAWdataset(hyperparameters, 'training', True, data_type=hyperparameters['data_type'], ps=prediction_span)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=8, drop_last=True,
                            sampler=None, shuffle=True, pin_memory=True) 
    for (input_joint_motion, future_joint_motion, traj, frame, label) in dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:,:,:,:].flatten().numpy() for i in range(img_repre.shape[0])])
        features_3.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training] features and labels: {len(features_1)}, {len(features_2)}, {len(features_3)}, {len(labels)}, {len(dataset)}')

    val_dataset = SAWdataset(hyperparameters, 'validation', True, data_type=hyperparameters['data_type'], ps=prediction_span)
    val_dataloader = DataLoader(val_dataset, batch_size=bs,
                                num_workers=8, drop_last=True,
                                sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, traj, frame, label) in val_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
        img_repre = test_Popeyes_step(frame, model_p, mode='feature_extraction')
        traj_repre = test_traj_step(traj, model_t, mode='feature_extraction')
        features_1.extend([motion_pred[i,:,:].flatten().numpy() for i in range(motion_pred.shape[0])])
        features_2.extend([img_repre[i,:,:,:,:].flatten().numpy() for i in range(img_repre.shape[0])])
        features_3.extend([traj_repre[i,:].flatten().numpy() for i in range(traj_repre.shape[0])])
        labels.extend([label[i].numpy().astype(np.uint8) for i in range(label.shape[0])])
    print(f'[Training + Validation] features and labels: {len(features_1)}, {len(features_2)}, {len(features_3)}, {len(labels)}, {len(val_dataset)}')

    test_dataset = SAWdataset(hyperparameters, 'testing', True, data_type=hyperparameters['data_type'], ps=prediction_span)
    test_dataloader = DataLoader(test_dataset, batch_size=bs,
                                 num_workers=8, drop_last=True,
                                 sampler=None, shuffle=True, pin_memory=True)
    for (input_joint_motion, future_joint_motion, traj, frame, label) in test_dataloader:
        motion_pred = test_siMLPe_step(input_joint_motion, future_joint_motion, model_s, mode='feature_extraction')
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
    with open(os.path.join(hyperparameters['log_dir_path'], 'clf_SPT.txt'), 'a') as f:
        f.write(f'***** prediction span: {prediction_span} *****\n')
        clf = svm.LinearSVC(dual='auto', C=5e-5)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        report = metrics.classification_report(y_test, predicted, digits=4)
        f.write(f'***** {'LinearSVC'} *****\n')
        f.write(report)
        f.write('\n')