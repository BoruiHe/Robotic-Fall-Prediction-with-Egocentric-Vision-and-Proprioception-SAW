import os
import shutil
import random
import pickle as pk
import numpy as np
from math import inf


def renaming_virtual_poppy(num_videos=6):
    path = os.path.join(os.getcwd(), 'data', 'virtual_poppy')
    with open(os.path.join(path, 'renaming is done.txt'), 'w') as f:
        f.write('Before renaming:\n')
        fall = os.listdir(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'fall'))
        walk = os.listdir(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'walk'))
        f.write(f'fall: {len(fall)}\n')
        f.write(f'walk: {len(walk)}\n')

        # Drop videos for equal dataset.
        if len(fall) > num_videos:
            folders = random.sample(fall, len(fall)-num_videos)
            for folder in folders:
                shutil.rmtree(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'fall', folder))
            
        if len(walk) > num_videos:
            folders = random.sample(walk, len(walk)-num_videos)
            for folder in folders:
                shutil.rmtree(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'walk', folder))

        f.write('After renaming:\n')
        fall = os.listdir(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'fall'))
        walk = os.listdir(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'walk'))

        # Generate more videos if do not have enough, otherwise keep going.
        if len(fall) < num_videos:
            raise Exception('Generate more fall videos')
        else:
            f.write(f'fall: {len(fall)}\n')
        if len(walk) < num_videos:
            raise Exception('Generate mroe walk videos')
        else:
            f.write(f'walk: {len(walk)}\n')

        # Renaming
        # assign each file a wrong name
        a = random.sample(range(10000,30000), num_videos)
        b = random.sample(range(10000,30000), num_videos)

        for i in zip(fall, a):
            os.rename(os.path.join(path, 'fall', i[0]), os.path.join(path, 'fall', str(i[1])))
        for i in zip(walk, b):
            os.rename(os.path.join(path, 'walk', i[0]), os.path.join(path, 'walk', str(i[1])))

        # assign each file the correct name
        fall = os.listdir(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'fall'))
        walk = os.listdir(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'walk'))

        a = [str(i) for i in range(num_videos)]
        b = [str(i + num_videos) for i in range(num_videos)]

        for i in zip(fall, a):
            os.rename(os.path.join(path, 'fall', i[0]), os.path.join(path, 'fall', i[1]))
        for i in zip(walk, b):
            os.rename(os.path.join(path, 'walk', i[0]), os.path.join(path, 'walk', i[1]))

        f.write('Do not run the renaming function again!')
    print('VP renaming is done.')

def preprocessing_virtual_poppy():
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'virtual_poppy_scenes')
    video_dict = {'line': []}
    video_dict['folder'] = os.listdir(os.path.join(data_path, 'fall')) + os.listdir(os.path.join(data_path, 'walk'))
    video_dict['folder'] = [int(i) for i in video_dict['folder']]
    video_dict['folder'].sort()
    video_dict['folder'] = [str(i) for i in video_dict['folder']]
    video_dict['class'] = ['fall' for _ in range(int(len(video_dict['folder'])/2))] + ['walk' for _ in range(int(len(video_dict['folder'])/2))] 

    shortest_length = inf # the highest index of images of this full video
    for folder, clss in zip(list(video_dict['folder']), list(video_dict['class'])):
        path = os.path.join(data_path, clss, folder)
        print(path)
        imgs =  np.load(os.path.join(path, 'imgs.npz'))['imgs']
        trajectory = np.load(os.path.join(path, 'trajectory.npz'))['traj']
        ajp = np.load(os.path.join(path, 'actual_joint_pos.npz'))['ajp']
        if imgs.shape[0] == trajectory.shape[0] == ajp.shape[0]:
            if imgs.shape[0] < shortest_length:
                shortest_length = imgs.shape[0]
        else:
            raise Exception(f'Missing data!\nIMG shape:{imgs.shape[0]}; Traj shape:{trajectory.shape[0]}; ajp shape:{ajp.shape[0]}\npath: ' + path)
        
        with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
            hc_z_coordinates = np.array(pk.load(f))
            borderline = hc_z_coordinates > 0.7

        with open(os.path.join(path, 'seg.pkl'), 'rb') as f:
            seg = pk.load(f)

        # any value less than 10 -> fall, =10 -> walk
        # the value refers to the index of the transition where a fall "happened"
        if borderline.all() > 0:
            video_dict['line'].append(10)
        else:
            # settle stage is indexed with number -1, it is okay if z coordinates is LEQ to 0.7 in this stage. So skip it.
            idx_of_first_fall = np.where(borderline[(seg[0][0]):]==0)[0][0] + seg[0][0]
            # following transitions are indexed with 0 to 9
            for i in range(0, 10):
                if seg[i][0] <= idx_of_first_fall <= seg[i][1]:
                    video_dict['line'].append(i)
                    break
                if i == 9: 
                    raise Exception('No tanistion index found for a falling episode!')
        
    video_dict['sl'] = shortest_length
    with open(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'videos.pkl'), 'wb') as f:
        pk.dump(video_dict, f)
    print('VP preprocessing is done.')

def preprocessing_virtual_poppy_scenes():
    data_path = os.path.join(os.getcwd(), 'data', 'virtual_poppy_scenes')
    video_dict = {'line': []}
    fall = [i for i in os.listdir(os.path.join(data_path, 'fall'))]
    fall.sort()
    walk = [i for i in os.listdir(os.path.join(data_path, 'walk'))]
    walk.sort()
    video_dict['folder'] = fall + walk
    video_dict['folder'] = [str(i) for i in video_dict['folder']]
    video_dict['class'] = ['fall' for _ in range(len(os.listdir(os.path.join(data_path, 'fall'))))] + ['walk' for _ in range(len(os.listdir(os.path.join(data_path, 'walk'))))] 

    shortest_length = inf # the highest index of images of this full video
    for folder, clss in zip(list(video_dict['folder']), list(video_dict['class'])):
        path = os.path.join(data_path, clss, folder)
        print(path)
        imgs =  np.load(os.path.join(path, 'imgs.npz'))['imgs']
        trajectory = np.load(os.path.join(path, 'trajectory.npz'))['traj']
        ajp = np.load(os.path.join(path, 'actual_joint_pos.npz'))['ajp']
        if imgs.shape[0] == trajectory.shape[0] == ajp.shape[0]:
            if imgs.shape[0] < shortest_length:
                shortest_length = imgs.shape[0]
        else:
            raise Exception(f'Missing data!\nIMG shape:{imgs.shape[0]}; Traj shape:{trajectory.shape[0]}; ajp shape:{ajp.shape[0]}\npath: ' + path)
        
        with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
            hc_z_coordinates = np.array(pk.load(f))
            borderline = hc_z_coordinates > 0.7

        with open(os.path.join(path, 'seg.pkl'), 'rb') as f:
            seg = pk.load(f)

        # any value less than 10 -> fall, =10 -> walk
        # the value refers to the index of the transition where a fall "happened"
        if borderline.all() > 0:
            video_dict['line'].append(10)
        else:
            # settle stage is indexed with number -1, it is okay if z coordinates is LEQ to 0.7 in this stage. So skip it.
            idx_of_first_fall = np.where(borderline[(seg[0][0]):]==0)[0][0] + seg[0][0]
            # following transitions are indexed with 0 to 9
            for i in range(0, 10):
                if seg[i][0] <= idx_of_first_fall <= seg[i][1]:
                    video_dict['line'].append(i)
                    break
                if i == 9: 
                    raise Exception('No tanistion index found for a falling episode!')
        
    video_dict['sl'] = shortest_length
    with open(os.path.join(os.getcwd(), 'data', 'virtual_poppy_scenes', 'videos.pkl'), 'wb') as f:
        pk.dump(video_dict, f)
    print('VP preprocessing is done.')

def preprocessing_real_poppy():
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'real_poppy')
    target_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'real_poppy_go')

    if not os.path.isdir(target_path):
        os.makedirs(target_path)
        for rep in range(1, 31):
            '''
            results: the number of completed transitions before a fall (4 is a complete step without falling)
            bufs[k][i]: buffer data for ith waypoint of kth transition in the selected repetition
            bufs[k][i] is a tuple (flag, buffers, elapsed)
                flag: True if motion was completed safely (e.g., low motor temperature), False otherwise
                buffers['position'][t, j]: actual angle of jth joint at timestep t of motion
                buffers['target'][t, j]: target angle of jth joint at timestep t of motion
                elapsed[t]: elapsed time since start of motion at timestep t of motion
            ''' 
            with open(os.path.join(data_path, 'results_%d.pkl' % (-rep)), 'rb') as f:
                (_, results, bufs) = pk.load(f, encoding='latin1') # motor_names, results[rep], bufs[rep]
            
            with open(os.path.join(data_path, 'frames_%d.pkl' % (-rep)), 'rb') as f:
                raw_frames = pk.load(f, encoding='latin1')
            frames, itpol_frames = [], []
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

            labels, elapsed_time, joint_angles, target_angles = [], [], [], []
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
                # joint_angles, target_angles and frames
                for j in range(len(buffers)):
                    joint_angles.append(buffers[j]['position'])
                    target_angles.append(buffers[j]['target'])
                    for m in range(10):
                        frames.append(raw_frames[k][j][m])

            labels = np.array(labels)
            for i in range(1, len(elapsed_time)):
                elapsed_time[i][:] = elapsed_time[i] + elapsed_time[i-1][-1]
            elapsed_time = np.concatenate(elapsed_time)
            joint_angles = np.concatenate(joint_angles)
            target_angles = np.concatenate(target_angles)
            frames = np.stack(frames, axis=0)

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
                    
            itpol_joint_angles, itpol_labels = [], []
            for index, (lowb, highb) in enumerate(idx_boundary):
                if lowb == highb == 0:
                    itpol_j_a = joint_angles[0].astype('float32')
                    itpol_label = labels[0]          
                    itpol_frame = frames[0]
                else:
                    # linear interpolation: joint angles
                    itpol_j_a = joint_angles[index] + (ideal_time[index]-elapsed_time[lowb])*(joint_angles[highb]-joint_angles[lowb])/(elapsed_time[highb]-elapsed_time[lowb])
                    itpol_j_a = itpol_j_a.astype('float32')
                    # the most recent rule: labels and frames
                    if ideal_time[index] < 0.5*(elapsed_time[highb]+elapsed_time[lowb]):
                        itpol_label = labels[lowb]
                        itpol_frame = frames[lowb]
                    else:
                        itpol_label = labels[highb]
                        itpol_frame = frames[highb]

                itpol_joint_angles.append(itpol_j_a)
                itpol_labels.append(itpol_label)
                itpol_frames.append(itpol_frame)
            '''
            Display frames
            '''
            # for frs in itpol_frames:
            #     plt.imshow(frs)
            #     plt.show(block=False)
            #     plt.pause(1)
            #     plt.close()
            '''
            Check if there is any abnormal value
            '''
            # if (np.array(itpol_joint_angles).max() > 360) or (np.array(itpol_joint_angles).min() < -360):
            #     raise Exception('Abnormal value found!')
            
            with open(os.path.join(target_path, f'JA_{rep}.pkl'), 'wb') as f:
                pk.dump(itpol_joint_angles, f)
            with open(os.path.join(target_path, f'TA_{rep}.pkl'), 'wb') as f:
                pk.dump(target_angles, f)
            with open(os.path.join(target_path, f'Labels_{rep}.pkl'), 'wb') as f:
                pk.dump(itpol_labels, f)
            with open(os.path.join(target_path, f'Frames_{rep}.pkl'), 'wb') as f:
                pk.dump(itpol_frames, f)
    print('RP preprocessing is done.')


if __name__ == '__main__':
    # print('Renaming...')
    # if not os.path.exists(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'renaming is done.txt')):
    #     renaming_virtual_poppy(5000)

    # print('Preprocessing...')
    # if not os.path.exists(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'videos.pkl')):
    #     preprocessing_virtual_poppy()

    # with open(os.path.join(os.getcwd(), 'data', 'virtual_poppy', 'videos.pkl'), 'rb') as f:
    #     print(pk.load(f))

    # preprocessing_real_poppy()
    # target_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'real_poppy_go')
    # files = os.listdir(target_path)
    # joint_angles_f = [f for f in files if 'JA_' in f]
    # target_angles_f = [f for f in files if 'TA_' in f]
    # label_f = [f for f in files if 'Labels_' in f]
    # frames_f = [f for f in files if 'Frames_' in f]
    # print(f'Total: {len(files)}\tJoint angles: {len(joint_angles_f)}\tTarget angles: {len(target_angles_f)}\tLabels: {len(label_f)}\tFrames: {len(frames_f)}.')

    preprocessing_virtual_poppy_scenes()