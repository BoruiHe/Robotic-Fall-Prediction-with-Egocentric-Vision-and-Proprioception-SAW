import os
import torch
import random
import torchvision
import numpy as np
import pickle as pk
from torch.utils.data import Dataset
import sys
sys.path.append(os.getcwd())
from utils.miscellaneous import rgba2rgb


class Egofalldataset(Dataset):
    def __init__(self, dataset_param, ps):
        super().__init__()
        self.dataset = dataset_param['dataset_name']
        self.N = dataset_param['N'] # N = 2*fps = 2*30
        self.ps = ps
        self.resize = torchvision.transforms.Resize(size=(224, 224))

        # which dataset you use
        if self.dataset == 'vir_poppy':
            self.data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'virtual_poppy')
            with open(os.path.join(os.path.dirname(os.getcwd()), 'data', 'virtual_poppy', 'videos.pkl'), 'rb') as f:
                self.video_dict = pk.load(f)
            self.num_samples = dataset_param['num_samples']
            if self.num_samples:
                assert self.num_samples % 2 == 0, 'num_sample should be an even integer.'

            self.fall_epi = list(range(5000))
            self.walk_epi = list(range(5000,10000))
            rep_idx = self.walk_epi + self.fall_epi

            self._idx = self._get_idx_for_everything_VP(rep_idx)

        elif self.dataset == 'real_poppy':
            self.path_to_RP = os.path.join(os.path.dirname(os.getcwd()), 'data', 'real_poppy')
            self.data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'real_poppy_go')

            self.walk_epi = [1, 6, 7, 9, 10, 11, 12, 16, 17, 21]
            self.fall_epi = [0, 2, 3, 4, 5, 8, 13, 14, 15, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29]
            rep_idx = self.walk_epi + self.fall_epi

            self._idx = self._get_idx_for_everything_RP(rep_idx)

    def __len__(self):
        return len(self._idx)
    
    def __getitem__(self, index):
        rep_idx, start_timestep = self._idx[index]
        if self.dataset == 'real_poppy':
            input_frames, future_labels = self._get_everything_RP(rep_idx, start_timestep)
            return input_frames, future_labels, rep_idx
        elif self.dataset == 'vir_poppy':
            input_frames, future_labels = self._get_everything_VP(rep_idx, start_timestep)
            return input_frames, future_labels, rep_idx
    
    def _split(self, video_dict, dataset_param):
        if dataset_param['num_samples']:
            half_length = int(dataset_param['num_samples']/2)
            half_ds_length = int(len(video_dict['folder'])/2)
        else:
            half_length = half_ds_length = int(len(video_dict['folder'])/2)

        # split training, validation and testing set for each fold
        splits = [int(half_length*i) for i in dataset_param['splits']]
        rep = []

        print('dataset size: {}, number of samples: {}, training set: {}, validation set: {} and testing set: {}'.format(
            len(video_dict['folder']), dataset_param['num_samples'], splits[0]*2, splits[1]*2, splits[2]*2
        ))
        remained_idx = random.sample(range(int(len(video_dict['folder'])/2)), half_length)
        # training
        training = random.sample(remained_idx, splits[0])
        remained_idx = list(set(remained_idx).difference(training))
        training = training + [half_ds_length + i for i in training]
        training.sort()
        # validation
        validation = random.sample(remained_idx, splits[1])
        remained_idx = list(set(remained_idx).difference(validation))
        validation = validation + [half_ds_length + i for i in validation]
        validation.sort()
        # testing
        testing = random.sample(remained_idx, splits[2])
        remained_idx = list(set(remained_idx).difference(testing))
        testing = testing + [half_ds_length + i for i in testing]
        testing.sort()

        rep.append(training)
        rep.append(validation)
        rep.append(testing)
        return rep
    
    def _get_idx_for_everything_VP(self, idx_reps):
        idx = []
        fall_idx, walk_idx = [], []
        video_list = list(zip(self.video_dict['class'], self.video_dict['folder'], self.video_dict['line']))

        ''' The fact is that the number of walking video clips are more than that of falling video clips.
            To create a perfect balanced set, get falling video clips (label->1) first, then extract the exactly same number of walking video clips (label->0). '''
        for _, index in enumerate(idx_reps):
            path = os.path.join(self.data_path, video_list[index][0], video_list[index][1])

            with open(os.path.join(path, 'seg.pkl'), 'rb') as f:
                seg = pk.load(f)
            with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
                hc_z = pk.load(f)
            stop_seg = video_list[index][2]
            if 0 <= stop_seg < 10: # fall episodes
                # find the line interval, the 1st element is the stop line for video clips extraction
                if seg[stop_seg][0] <= self.N+self.ps:
                    valid_start_timestep = np.arange(0, 1)
                else:
                    stop_line = min(seg[stop_seg][0]-1, seg[9][1]-30)
                    stop_line -= (self.N + self.ps)
                    valid_start_timestep = np.arange(stop_line, stop_line+30)
                # check whether the labels are 1 because the framwise lables of a falling episodes are not always 1
                valid_start_timestep = valid_start_timestep.tolist()
                valid_start_timestep = [start_timestep for start_timestep in valid_start_timestep if np.array(hc_z[start_timestep + self.N + self.ps - 1]) < 0.7]
                fall_idx.extend(zip([index] * len(valid_start_timestep), valid_start_timestep))
            elif stop_seg == 10: # walking episodes
                # find the line interval, the 1st element is the stop line for video clips extraction
                stop_line = seg[9][1]
                stop_line -= (self.N + self.ps)
                valid_start_timestep = np.arange(0, stop_line)
                walk_idx.extend(zip([index] * len(valid_start_timestep), valid_start_timestep))
        
        if 2*len(fall_idx) > self.num_samples:
            walk_idx = random.sample(walk_idx, int(self.num_samples/2))
            fall_idx = random.sample(fall_idx, int(self.num_samples/2))
        else:
            walk_idx = random.sample(walk_idx, len(fall_idx))

        idx.extend(walk_idx)
        idx.extend(fall_idx)

        return idx

    def _get_everything_VP(self, idx_of_rep, start_timestep):
        if idx_of_rep >= 5000:
            catgy = 'walk'
        else:
            catgy = 'fall'
        path = os.path.join(self.data_path, catgy, str(idx_of_rep))

        with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
            hc_z = pk.load(f)

        imgs = np.load(os.path.join(path, 'imgs.npz'))['imgs']
        imgs = imgs[start_timestep : start_timestep + self.N][[i*6 for i in range(10)]]
        imgs = (rgba2rgb(imgs)/255).transpose(0,3,1,2).astype('float32')
        imgs = self.resize(torch.from_numpy(imgs))
        output = [imgs, # input frames
                  np.array(hc_z[start_timestep + self.N + self.ps - 1]) < 0.7 # future labels, also the ground truth, 0->walking, 1->fall
                ]

        return output
        
    def _get_idx_for_everything_RP(self, idx_reps):
        idx = []
        fall_idx, walk_idx = [], []
        for rep in idx_reps:
            '''
            results: the number of completed transitions before a fall (4 is a complete step without falling)
            bufs[k][i]: buffer data for ith waypoint of kth transition in the selected repetition
            bufs[k][i] is a tuple (flag, buffers, elapsed)
                flag: True if motion was completed safely (e.g., low motor temperature), False otherwise
                buffers['position'][t, j]: actual angle of jth joint at timestep t of motion
                buffers['target'][t, j]: target angle of jth joint at timestep t of motion
                elapsed[t]: elapsed time since start of motion at timestep t of motion
            ''' 
            with open(os.path.join(self.path_to_RP, 'results_%d.pkl' % (-rep-1)), 'rb') as f:
                (_, results, bufs) = pk.load(f, encoding='latin1') # motor_names, results[rep], bufs[rep]
            '''
                (results)   transition_index
                (1)         0.                  Initial -> Shift
                (2)         1.                  Shift   -> Push
                (3)         2.                  Push    -> Lift
                (3)         3.                  Lift    -> Kick
                (4)         4.                  Kick    -> Initial
                
                Near the end there is a transition to lift, and then a transition to kick.
                In practice, these two transitions happened very quickly, because one of Poppy's feet does not touch the ground,
                and it can easily lose balance if it stays on only one foot for too long.
                They were in fact so quick that I could not reliably distinguish them when I recorded successes by hand.
                So these two transitions are lumped together for the purposes of recording success.
            '''
            # num of completed transitions -> idx of the last completed transition
            if results >= 3:
                last_compl_transition = results # 3->3, 4->4
                # ckp = (22 + 2 + 3*(results-2))*10
            else:
                last_compl_transition = results-1 # 1->0, 2->1
                # ckp = 11*results*10

            labels, elapsed_time = [], []
            for k in range(5): # k is the index of each transition
                _, buffers, elapsed = zip(*bufs[k]) # (flag, buffers, elapsed)
                # actual labels
                if k <= last_compl_transition:
                    label = 0 # still walking
                else:
                    label = 1 # a fall happens
                '''
                each transition is decomposed into short periods ~ len(buffers)
                each short period contains 10 time steps
                '''
                # labels
                for _ in range(len(buffers)*10):
                    labels.append(label)
                # accumulate elapsed time
                for i in range(1, len(elapsed)):
                    elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
                elapsed = np.concatenate(elapsed)
                elapsed_time.append(elapsed)
                
            labels = np.array(labels)
            for i in range(1, len(elapsed_time)):
                elapsed_time[i][:] = elapsed_time[i] + elapsed_time[i-1][-1]
            elapsed_time = np.concatenate(elapsed_time)

            interval = elapsed_time[-1]/elapsed_time.size
            ideal_time = [interval*i for i in range(len(elapsed_time))]
            idx_higher_boundary = np.searchsorted(elapsed_time, ideal_time).astype(int)
            idx_boundary = []
            for ilb in idx_higher_boundary:
                if ilb == 0:
                    idx_boundary.append((0,0))
                else:
                    idx_boundary.append((ilb-1, ilb))
                    
            itpol_labels = []
            for index, (lowb, highb) in enumerate(idx_boundary):
                if lowb == highb == 0:
                    itpol_label = labels[0]
                else:                   
                    # the most recent rule: labels
                    if ideal_time[index] < 0.5*(elapsed_time[highb]+elapsed_time[lowb]):
                        itpol_label = labels[lowb]
                    else:
                        itpol_label = labels[highb]
                itpol_labels.append(itpol_label)

            # each transition has different number of waypoints, each waypoint has 10 timesteps
            if results >= 3:
                # calculate the stop line index in raw data
                line = (22+2+3*(results-2))*10
                # convert it into the stop line index in itpol_labels
                line = np.searchsorted(idx_higher_boundary, line)
                if results == 4:
                    line = line-self.N-self.ps
                    valid_start_timestep = np.arange(0, line).tolist()
                    valid_start_timestep = [vst for vst in valid_start_timestep if not itpol_labels[vst+self.N+self.ps]]
                    walk_idx.extend(zip([rep+1] * len(valid_start_timestep), valid_start_timestep))
                else:
                    line = min(line-self.N-self.ps, 300-self.N-self.ps-30)
                    valid_start_timestep = np.arange(line, line+30)
                    valid_start_timestep = [vst for vst in valid_start_timestep if itpol_labels[vst+self.N+self.ps]]
                    fall_idx.extend(zip([rep+1] * len(valid_start_timestep), valid_start_timestep))
            else:
                # calculate the stop line index in raw data
                line = (11*results)*10
                # convert it into the stop line index in itpol_labels
                line = np.searchsorted(idx_higher_boundary, line)
                line = max(line-self.N-self.ps, 0)
                valid_start_timestep = np.arange(line, line+30)
                valid_start_timestep = [vst for vst in valid_start_timestep if itpol_labels[vst+self.N+self.ps]]
                fall_idx.extend(zip([rep+1] * len(valid_start_timestep), valid_start_timestep))
        
        idx.extend(fall_idx)
        walk_idx = random.sample(walk_idx, len(fall_idx))
        idx.extend(walk_idx)
        return idx

    def _get_everything_RP(self, idx_of_rep, start_timestep):
        with open(os.path.join(self.data_path, f'Labels_{idx_of_rep}.pkl'), 'rb') as f:
            itpol_labels = pk.load(f)
        with open(os.path.join(self.data_path, f'Frames_{idx_of_rep}.pkl'), 'rb') as f:
            itpol_frames = pk.load(f)
        imgs = np.stack(itpol_frames[start_timestep : start_timestep + self.N])[[i*2 for i in range(10)]]
        imgs = imgs[:,:,:,[2,1,0]].transpose(0,3,1,2)
        imgs = (imgs/255).astype('float32')
        imgs = torch.from_numpy(imgs)
        imgs = self.resize(imgs)
        output = [imgs, # input frames
                  np.array(itpol_labels[start_timestep + self.N + self.ps]) # future labels, also the ground truth
                ]
        return output


if __name__ == '__main__':
    # import sys
    from torch.utils.data import DataLoader
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from hp import hyperparameters_virtual, hyperparameters_real
    from utils.miscellaneous import set_random_seeds

    hyperparameters = hyperparameters_virtual
    set_random_seeds(608)
    dataset = Egofalldataset(
        dataset_param= hyperparameters,
        ps = 10,
        )
    print('dataset length:', len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, num_workers=8, drop_last=False, sampler=None, shuffle=True, pin_memory=True)
    for (frames, labels, rep_idx) in dataloader:
        print(frames.shape, labels.shape, rep_idx.shape)
