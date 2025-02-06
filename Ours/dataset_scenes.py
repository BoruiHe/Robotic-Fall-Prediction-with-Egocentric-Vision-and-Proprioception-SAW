import os
import yaml
import random
import numpy as np
import pickle as pk
from torch.utils.data import Dataset
import sys
sys.path.append(os.getcwd())
from utils.miscellaneous import rgba2rgb


class SAWdataset(Dataset):
    def __init__(self, dataset_param, mode, output_frames, data_type='JA', for_visualization=False, ps=None, debug_path=None):
        super().__init__()
        self.dataset = dataset_param['dataset_name']
        self.mode = mode
        self.N = dataset_param['N'] # N = 2*fps = 2*30
        if ps:
            self.ps = ps
        else:
            self.ps = int(self.N/2) # default ps = fps = 30
        self.data_type = data_type
        self.output_frames = output_frames
        self.for_visualization = for_visualization
        if debug_path:
            path = debug_path
        else:
            path = dataset_param['log_dir_path']
        video_idx_splits = {}

        # which dataset you use
        if self.dataset == 'vir_poppy':
            self.data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'virtual_poppy_scenes')
            with open(os.path.join(self.data_path, 'videos_.pkl'), 'rb') as f:
                self.video_dict = pk.load(f)
            if self.mode == 'training':
                if os.path.isfile(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml')):
                    with open(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml'), 'r') as infile:
                        idx_reps = yaml.safe_load(infile)['training_idx']
                else:
                    split_idx = self._split(dataset_param, unseen_test=True)
                    with open(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml'), 'w') as outfile:
                        video_idx_splits['training_idx'] = idx_reps = split_idx[0]
                        video_idx_splits['validation_idx'] = split_idx[1]
                        video_idx_splits['testing_idx'] = split_idx[2]
                        yaml.dump(video_idx_splits, outfile, default_flow_style=False)
            elif self.mode == 'validation':
                with open(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml'), 'r') as infile:
                    idx_reps = yaml.safe_load(infile)['validation_idx']        
            elif self.mode == 'testing':
                with open(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml'), 'r') as infile:
                    idx_reps = yaml.safe_load(infile)['testing_idx']

            self._idx = self._get_idx_for_everything_VP(idx_reps, self.video_dict, self.data_path)
            if self.for_visualization:
                if len(idx_reps) > 10:
                    idx_occur = [j[0] for j in self._idx]
                    significant_reps = [i for i in idx_reps if idx_occur.count(i)>=20]
                    self._idx_reps = np.random.choice(significant_reps, size=10, replace=False).tolist()
                else:
                    self._idx_reps = idx_reps
        elif self.dataset == 'real_poppy':
            self.temp_path = os.path.join(path, 'intermediate_files')
            self.path_to_RP = os.path.join(os.path.dirname(os.getcwd()), 'data', 'real_poppy')
            self.data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'real_poppy_go')
            self.downsample_factor = 1
            if self.mode == 'training':
                if os.path.isfile(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml')):
                    with open(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml'), 'r') as infile:
                        rep_idx = yaml.safe_load(infile)['training_idx']
                else:
                    walk_epi = {1, 6, 7, 9, 10, 11, 12, 16, 17, 21}
                    fall_epi = {0, 2, 3, 4, 5, 8, 13, 14, 15, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29}
                    # Traininig set
                    training_epi = random.sample(list(walk_epi), 8)
                    walk_epi = walk_epi - set(training_epi)
                    training_epi.extend(random.sample(list(fall_epi), 16))
                    fall_epi = fall_epi - set(training_epi)
                    training_epi.sort()
                    # Validation set
                    validation_epi = random.sample(list(walk_epi), 1)
                    walk_epi = walk_epi - set(validation_epi)
                    validation_epi.extend(random.sample(list(fall_epi), 2))
                    fall_epi = fall_epi - set(validation_epi)
                    validation_epi.sort()
                    # Testing set
                    testing_epi = list(walk_epi) + list(fall_epi)
                    testing_epi.sort()
                    with open(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml'), 'w') as outfile:
                        video_idx_splits['training_idx'] = training_epi
                        video_idx_splits['validation_idx'] = validation_epi
                        video_idx_splits['testing_idx'] = testing_epi
                        yaml.dump(video_idx_splits, outfile, default_flow_style=False)
                    rep_idx = training_epi
            elif self.mode == 'validation':
                with open(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml'), 'r') as infile:
                    rep_idx = yaml.safe_load(infile)['validation_idx']
            elif self.mode == 'testing':
                with open(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml'), 'r') as infile:
                    rep_idx = yaml.safe_load(infile)['testing_idx']
            if self.for_visualization:
                self._idx_reps = rep_idx
            self._idx = self._get_idx_for_everything_RP(rep_idx)

    def __len__(self):
        if self.for_visualization:
            return len(self._idx_reps)
        return len(self._idx)
    
    def __getitem__(self, index):
        if self.for_visualization:
            if self.dataset == 'real_poppy':
                rep_idx = self._idx_reps[index] + 1
            elif self.dataset == 'vir_poppy':
                rep_idx = self._idx_reps[index]
            everyframe_in_rep = []
            for (_, start_timestep) in [t for t in self._idx if t[0] == rep_idx]:
                if self.output_frames:
                    if self.dataset == 'real_poppy':
                        input_joint_angles, future_joint_angles, input_frames, traj, future_labels = self._get_everything_RP(rep_idx, start_timestep)
                        everyframe_in_rep.append((input_joint_angles, future_joint_angles, traj, input_frames, future_labels, rep_idx, start_timestep)) 
                    elif self.dataset == 'vir_poppy':
                        input_joint_angles, future_joint_angles, input_frames, traj, future_labels = self._get_everything_VP(rep_idx, self.video_dict, start_timestep, self.data_path)
                        everyframe_in_rep.append((input_joint_angles, future_joint_angles, traj, input_frames, future_labels, rep_idx, start_timestep))
                else:
                    if self.dataset == 'real_poppy':
                        input_joint_angles, future_joint_angles, traj, future_labels = self._get_everything_RP(rep_idx, start_timestep)
                        everyframe_in_rep.append((input_joint_angles, future_joint_angles, traj, future_labels, rep_idx, start_timestep))
                    elif self.dataset == 'vir_poppy':
                        input_joint_angles, future_joint_angles, traj, future_labels = self._get_everything_VP(rep_idx, self.video_dict, start_timestep, self.data_path)
                        everyframe_in_rep.append((input_joint_angles, future_joint_angles, traj, future_labels, rep_idx, start_timestep))
            return everyframe_in_rep
        else:
            rep_idx, start_timestep = self._idx[index]
            if self.output_frames:
                if self.dataset == 'real_poppy':
                    input_joint_angles, future_joint_angles, traj, input_frames, future_labels = self._get_everything_RP(rep_idx, start_timestep)
                    return input_joint_angles, future_joint_angles, traj, input_frames, future_labels
                elif self.dataset == 'vir_poppy':
                    input_joint_angles, future_joint_angles, traj, input_frames, future_labels = self._get_everything_VP(rep_idx, self.video_dict, start_timestep, self.data_path)
                    return input_joint_angles, future_joint_angles, traj, input_frames, future_labels
            else:
                if self.dataset == 'real_poppy':
                    input_joint_angles, future_joint_angles, traj, future_labels = self._get_everything_RP(rep_idx, start_timestep)
                    return input_joint_angles, future_joint_angles, traj, future_labels
                elif self.dataset == 'vir_poppy':
                    input_joint_angles, future_joint_angles, traj, future_labels = self._get_everything_VP(rep_idx, self.video_dict, start_timestep, self.data_path)
                    return input_joint_angles, future_joint_angles, traj, future_labels
    
    def _split(self, dataset_param, unseen_test=False):
        
        rep, training, validation, testing = [], [], [], []
        with open(os.path.join(self.data_path, 'texture_combi.pkl'), 'rb') as pf:
            texture_combi = pk.load(pf)
        if unseen_test:
            remained_fall, remained_walk = [], []
            for texture in list(texture_combi.keys())[:60]:
                assert len(texture_combi[texture]) == 30, 'Virtual Poppy Scenes is incomplete or needs purification.'
                remained_fall.extend(texture_combi[texture][:20])
                remained_walk.extend(texture_combi[texture][20:])
            # split training, validation and testing set for each fold
            splits_fall = [int(len(remained_fall)*i) for i in dataset_param['splits']]
            splits_walk = [int(len(remained_walk)*i) for i in dataset_param['splits']]
            print(splits_fall, splits_walk)            
            sample_fall = random.sample(remained_fall, splits_fall[0])
            sample_walk = random.sample(remained_walk, splits_walk[0])
            training.extend(sample_fall)
            training.extend(sample_walk)

            remained_fall = list(set(remained_fall)-set(sample_fall))
            remained_walk = list(set(remained_walk)-set(sample_walk))
            sample_fall = random.sample(remained_fall, splits_fall[1])
            sample_walk = random.sample(remained_walk, splits_walk[1])
            validation.extend(sample_fall)
            validation.extend(sample_walk)

            remained_fall = list(set(remained_fall)-set(sample_fall))
            remained_walk = list(set(remained_walk)-set(sample_walk))
            sample_fall = random.sample(remained_fall, splits_fall[2])
            sample_walk = random.sample(remained_walk, splits_walk[2])
            testing.extend(sample_fall)
            testing.extend(sample_walk)
        else:
            # split training, validation and testing set for each fold
            splits_fall = [int(20*i) for i in dataset_param['splits']]
            splits_walk = [int(10*i) for i in dataset_param['splits']]
            for texture in texture_combi.keys():
                assert len(texture_combi[texture]) == 30, 'Virtual Poppy Scenes is incomplete or needs purification.'

                remained_fall = texture_combi[texture][:20]
                remained_walk = texture_combi[texture][20:]
                sample_fall = random.sample(remained_fall, splits_fall[0])
                sample_walk = random.sample(remained_walk, splits_walk[0])
                training.extend(sample_fall)
                training.extend(sample_walk)

                remained_fall = list(set(remained_fall)-set(sample_fall))
                remained_walk = list(set(remained_walk)-set(sample_walk))
                sample_fall = random.sample(remained_fall, splits_fall[1])
                sample_walk = random.sample(remained_walk, splits_walk[1])
                validation.extend(sample_fall)
                validation.extend(sample_walk)

                remained_fall = list(set(remained_fall)-set(sample_fall))
                remained_walk = list(set(remained_walk)-set(sample_walk))
                sample_fall = random.sample(remained_fall, splits_fall[2])
                sample_walk = random.sample(remained_walk, splits_walk[2])
                testing.extend(sample_fall)
                testing.extend(sample_walk)

        rep.append(training)
        rep.append(validation)
        rep.append(testing)
        print(len(training), len(validation), len(testing))
        return rep
    
    def _get_idx_for_everything_VP(self, idx_reps, video_dict, path_to_VP):
        idx = []
        fall_idx, walk_idx = [], []
        video_list = list(zip(video_dict['class'], video_dict['folder'], video_dict['line']))

        ''' The fact is that the number of walking video clips are more than that of falling video clips.
            To create a perfect balanced set, get falling video clips (label->1) first, then extract the exactly same number of walking video clips (label->0). '''
        for folder in idx_reps:
            index = video_dict['folder'].index(folder)
            path = os.path.join(path_to_VP, video_list[index][0], video_list[index][1])

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
                fall_idx.extend(zip([folder] * len(valid_start_timestep), valid_start_timestep))
                
            # elif stop_seg == 10: # walking episodes
            #     # find the line interval, the 1st element is the stop line for video clips extraction
            #     stop_line = seg[9][1]
            #     stop_line -= (self.N + self.ps)
            #     valid_start_timestep = np.arange(0, stop_line)
            #     walk_idx.extend(zip([folder] * len(valid_start_timestep), valid_start_timestep))

            # stop_line = video_list[index][2]
            # if stop_line < len(hc_z): # fall episodes
            #     if stop_line < len(hc_z)-30-self.N-self.ps:
            #         valid_start_timestep = np.arange(stop_line, stop_line+30)
            #         # check whether the labels are 1 because the framwise lables of a falling episodes are not always 1
            #         valid_start_timestep = valid_start_timestep.tolist()
            #         try:
            #             valid_start_timestep = [start_timestep for start_timestep in valid_start_timestep if np.array(hc_z[start_timestep + self.N + self.ps - 1]) < 0.7]
            #         except:
            #             print(len(hc_z), stop_line, self.N, self.ps)
            #         fall_idx.extend(zip([folder] * len(valid_start_timestep), valid_start_timestep))

            else: # walking episodes
                # find the line interval, the 1st element is the stop line for video clips extraction
                stop_line = seg[9][1]
                stop_line -= (self.N + self.ps)
                valid_start_timestep = np.arange(0, stop_line)
                walk_idx.extend(zip([folder] * len(valid_start_timestep), valid_start_timestep))
        
        if not self.for_visualization:
            walk_idx = random.sample(walk_idx, len(fall_idx))
            idx.extend(walk_idx)
        idx.extend(fall_idx)
        return idx

    def _get_everything_VP(self, folder, video_dict, start_timestep, path_to_VP):

        video_list = list(zip(video_dict['class'], video_dict['folder'], video_dict['line']))
        index = video_dict['folder'].index(folder)
        path = os.path.join(path_to_VP, video_list[index][0], folder)
        if self.data_type == 'JA':
            actual_positions = np.load(os.path.join(path, 'actual_joint_pos.npz'))['ajp']
        elif self.data_type == 'pos':
            actual_positions = np.load(os.path.join(path, 'pos.npz'))['pos']
        with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
            hc_z = pk.load(f)
        traj = np.load(os.path.join(path, 'trajectory.npz'))['traj']
        if self.output_frames:
            imgs = np.load(os.path.join(path, 'imgs.npz'))['imgs']
            output = [actual_positions[start_timestep : start_timestep + self.N].astype('float32'), # input joint angles
                      actual_positions[start_timestep + self.N : start_timestep + self.N + self.ps].astype('float32'), # future joint angles, also the ground truth
                      np.stack(traj[start_timestep + self.N : start_timestep + self.N + self.ps]).astype('float32'),
                      imgs[start_timestep : start_timestep + self.N], # input frames
                      np.array(hc_z[start_timestep + self.N + self.ps - 1]) < 0.7 # future labels, also the ground truth, 0->walking, 1->fall
                      ]
            output[3] = (rgba2rgb(output[3])/255).transpose(3,0,1,2).astype('float32')
        else:
            output = [actual_positions[start_timestep : start_timestep + self.N].astype('float32'), # input joint angles
                      actual_positions[start_timestep + self.N : start_timestep + self.N + self.ps].astype('float32'), # future joint angles, also the ground truth
                      np.stack(traj[start_timestep + self.N : start_timestep + self.N + self.ps]).astype('float32'),
                      np.array(hc_z[start_timestep + self.N + self.ps - 1]) < 0.7 # future labels, also the ground truth, 0->walking, 1->fall
                      ]
        
        return output
        
    def _get_idx_for_everything_RP(self, idx_reps):
        if os.path.isfile(os.path.join(self.temp_path, f'ps_{self.ps}_idx.pkl')):
            with open(os.path.join(self.temp_path, f'ps_{self.ps}_idx.pkl'), 'rb') as f:
                idx_dict = pk.load(f)
            if self.mode in idx_dict.keys():
                idx = idx_dict[self.mode]
                return idx

        idx = []
        fall_idx, walk_idx = [], []
        if os.path.isfile(os.path.join(self.temp_path, f'ps_{self.ps}_idx.pkl')):
            with open(os.path.join(self.temp_path, f'ps_{self.ps}_idx.pkl'), 'rb') as f:
                idx_dict = pk.load(f)
        else:
            idx_dict = {}
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
            ''' 
            Check whether the labels at the falling moment: 300->(0,0), otherwise->(0,1)
            '''
            # if ckp == 300:
            #     print(f'{idx_of_rep}: {idx_boundary[ckp-1]}, {labels[ckp-2]}, {labels[ckp-1]}')
            # else:
            #     print(f'{idx_of_rep}: {idx_boundary[ckp-1]}, {labels[ckp-1]}, {labels[ckp]}')
            ''' 
            The ideal time should fall in the range of [elapsed_time[lower boundary], elapsed_time[higher boundary]]
            '''
            # for i in range(300):
            #     l, h = idx_boundary[i]
            #     if i!=0 and not (elapsed_time[l] <= ideal_time[i] <= elapsed_time[h]):
            #         print(f'Invalid index found! rep: {idx_of_rep}, time step: {i}')
                    
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
                
            if line < 0:
                raise Exception(f'Rep index: {rep}, line: {line}. No sequence could be extracted from this repetition!')
        
        idx.extend(fall_idx)
        if not self.for_visualization:
            walk_idx = random.sample(walk_idx, len(fall_idx))
            idx.extend(walk_idx)
        idx_dict[self.mode] = idx
        with open(os.path.join(self.temp_path, f'ps_{self.ps}_idx.pkl'), 'wb') as f2:
            pk.dump(idx_dict, f2)
        return idx

    def _get_everything_RP(self, idx_of_rep, start_timestep):
        with open(os.path.join(self.data_path, f'JA_{idx_of_rep}.pkl'), 'rb') as f:
            itpol_joint_angles = pk.load(f)
        with open(os.path.join(self.data_path, f'TA_{idx_of_rep}.pkl'), 'rb') as f:
            traj = pk.load(f)
        with open(os.path.join(self.data_path, f'Labels_{idx_of_rep}.pkl'), 'rb') as f:
            itpol_labels = pk.load(f)
        if self.output_frames:
            with open(os.path.join(self.data_path, f'Frames_{idx_of_rep}.pkl'), 'rb') as f:
                itpol_frames = pk.load(f)
            imgs = np.stack(itpol_frames[start_timestep : start_timestep + self.N])[:,:,:,[2,1,0]].transpose(3,0,1,2)
            # imgs = rgb_to_grayscale(from_numpy(imgs))
            output = [np.stack(itpol_joint_angles[start_timestep : start_timestep + self.N]), # input joint angles
                      np.stack(itpol_joint_angles[start_timestep + self.N : start_timestep + self.N + self.ps]), # future joint angles, also the ground truth
                      np.stack(traj[start_timestep + self.N : start_timestep + self.N + self.ps]).astype('float32'),
                      (imgs/255).astype('float32'), # input frames
                      np.array(itpol_labels[start_timestep + self.N + self.ps]) # future labels, also the ground truth
                      ]
        else:
            output = [np.stack(itpol_joint_angles[start_timestep : start_timestep + self.N]), # input joint angles
                      np.stack(itpol_joint_angles[start_timestep + self.N : start_timestep + self.N + self.ps]), # future joint angles, also the ground truth
                      np.stack(traj[start_timestep + self.N : start_timestep + self.N + self.ps]).astype('float32'),
                      np.array(itpol_labels[start_timestep + self.N + self.ps]) # future labels, also the ground truth
                      ]
        
        output[0] = np.deg2rad(output[0])
        output[1] = np.deg2rad(output[1])
        output[2] = np.deg2rad(output[2])
        return output


if __name__ == '__main__':
    # import sys
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from torch.utils.data import DataLoader
    from hp import hyperparameters_virtual, hyperparameters_real
    from utils.miscellaneous import set_random_seeds
    
    hyperparameters = hyperparameters_virtual
    seed = 608
    set_random_seeds(seed)
    path = os.path.join(os.getcwd(), 'debugging', str(seed))
    if os.path.isfile(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml')):
        os.remove(os.path.join(path, 'intermediate_files', 'video_idx_splits.yml'))
    dataset = SAWdataset(
        dataset_param= hyperparameters,
        mode='training',
        output_frames= True,
        data_type= 'JA',
        for_visualization=False,
        debug_path= path,
        )
    print('training:', len(dataset))
    dataset = SAWdataset(
        dataset_param= hyperparameters,
        mode='validation',
        output_frames= True,
        data_type= 'JA',
        for_visualization=False,
        debug_path= path,
        )
    print('validation:', len(dataset))
    dataset = SAWdataset(
        dataset_param= hyperparameters,
        mode='testing',
        output_frames= True,
        data_type= 'JA',
        for_visualization=False,
        debug_path= path,
        )
    print('testing:', len(dataset))

    # dataloader = DataLoader(dataset, batch_size=hyperparameters['bs_siMLPe'], num_workers=8, drop_last=False, sampler=None, shuffle=True, pin_memory=True)
    # for (input_joint_motion, future_joint_motion, _, frame, label) in dataloader:
    #     print(input_joint_motion.shape, future_joint_motion.shape, frame.shape, label.shape)
        # a,b = input_joint_motion.reshape(hyperparameters['bs_siMLPe'], hyperparameters['N'], -1), future_joint_motion.reshape(hyperparameters['bs_siMLPe'], int(hyperparameters['N']/2), -1)
        # print(a.shape, b.shape)