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
    def __init__(self, dataset_param):
        super().__init__()
        self.dataset = dataset_param['dataset_name']
        self.resize = torchvision.transforms.Resize(size=(224, 224))

        # which dataset you use
        if self.dataset == 'vir_poppy':
            self.data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'virtual_poppy')
            self.num_samples = dataset_param['num_samples']

            self.fall_epi = list(range(5000))
            self.walk_epi = list(range(5000,10000))
            if self.num_samples:
                assert self.num_samples % 2 == 0, 'num_sample should be an even integer.'
                self.fall_epi = random.sample(self.fall_epi, int(self.num_samples/2))
                self.walk_epi = random.sample(self.walk_epi, int(self.num_samples/2))
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
        rep_idx, frame_indices = self._idx[index]
        if self.dataset == 'real_poppy':
            input_frames, labels = self._get_everything_RP(rep_idx, frame_indices)
        elif self.dataset == 'vir_poppy':
            input_frames, labels = self._get_everything_VP(rep_idx, frame_indices)
        return input_frames, labels, rep_idx
    
    def _get_idx_for_everything_VP(self, idx_reps):
        idx = []
        for rep in idx_reps:
            if rep >= 5000:
                catgy = 'walk'
            else:
                catgy = 'fall'
            path = os.path.join(self.data_path, catgy, str(rep))
            length = np.load(os.path.join(path, 'imgs.npz'))['imgs'].shape[0]
            idx_equally_spaced = np.linspace(0, length, num=10, endpoint=False).astype('uint16').tolist()
            idx.append((rep, idx_equally_spaced))
        return idx

    def _get_everything_VP(self, idx_of_rep, frame_indices):
        if idx_of_rep >= 5000:
            catgy = 'walk'
        else:
            catgy = 'fall'
        path = os.path.join(self.data_path, catgy, str(idx_of_rep))
        with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
            hc_z = pk.load(f)
       
        path = os.path.join(self.data_path, catgy, str(idx_of_rep))
        imgs = np.load(os.path.join(path, 'imgs.npz'))['imgs'][frame_indices]
        imgs = (rgba2rgb(imgs)/255).transpose(0,3,1,2).astype('float32')
        imgs = torch.from_numpy(imgs)
        imgs = self.resize(imgs)
        output = [imgs, # input frames
                  np.array(hc_z)[frame_indices] < 0.7 # labels, also the ground truth, 0->walking, 1->fall
                ]
        return output
        
    def _get_idx_for_everything_RP(self, idx_reps):
        idx = []
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
                (_, _, bufs) = pk.load(f, encoding='latin1') # motor_names, results[rep], bufs[rep]
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

            elapsed_time = []
            for k in range(5): # k is the index of each transition
                _, _, elapsed = zip(*bufs[k]) # (flag, buffers, elapsed)
                '''
                each transition is decomposed into short periods ~ len(buffers)
                each short period contains 10 time steps
                '''
                # accumulate elapsed time
                for i in range(1, len(elapsed)):
                    elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
                elapsed = np.concatenate(elapsed)
                elapsed_time.append(elapsed)
                
            for i in range(1, len(elapsed_time)):
                elapsed_time[i][:] = elapsed_time[i] + elapsed_time[i-1][-1]
            elapsed_time = np.concatenate(elapsed_time)

            interval = elapsed_time[-1]/10
            ideal_time = [interval*i for i in range(10)]
            idx_higher_boundary = np.searchsorted(elapsed_time, ideal_time).astype(int)

            idx.append((rep+1, idx_higher_boundary))        
        return idx

    def _get_everything_RP(self, idx_of_rep, frame_indices):
        with open(os.path.join(self.data_path, f'Labels_{idx_of_rep}.pkl'), 'rb') as f:
            itpol_labels = pk.load(f)
        with open(os.path.join(self.data_path, f'Frames_{idx_of_rep}.pkl'), 'rb') as f:
            itpol_frames = pk.load(f)

        imgs = np.stack(itpol_frames)[:,:,:,[2,1,0]].transpose(0,3,1,2)[frame_indices]
        imgs = (imgs/255).astype('float32')
        imgs = torch.from_numpy(imgs)
        imgs = self.resize(imgs)
        output = [imgs, # input frames
                  np.array(itpol_labels)[frame_indices[-1]] # labels, also the ground truth
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
        dataset_param=hyperparameters,
        debug_path= os.path.join(os.getcwd(), 'debugging'),
        )
    print('RP:', len(dataset))
    dataloader = DataLoader(dataset, batch_size=hyperparameters['bs'], num_workers=8, drop_last=False, sampler=None, shuffle=True, pin_memory=True)
    for (vc, labels, index) in dataloader:
        print(vc.shape, labels.shape, index)