import os
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt


def collect_VP_distribution_data():
    '''
    Cannot install LaTex on HPC. Collect data for plotting, then download data to local machines.
    '''
    num_joints = 6
    for gait in ['fall', 'walk']:
        p = os.path.join(os.path.dirname(os.getcwd()), 'data', 'virtual_poppy', gait)
        ajp = {}
        for j in range(num_joints):
            ajp[str(j)] = []
        if gait == 'fall':
            rg = list(range(100))
        else:
            rg = list(range(5000, 5100))
        for i in rg:
            actual_positions = np.load(os.path.join(p, str(i), 'actual_joint_pos.npz'))['ajp'][:1354, :6]
            for j in range(num_joints):
                ajp[str(j)].append(actual_positions[:, j])
        for j in range(num_joints):
            ajp[str(j)] = np.array(ajp[str(j)])
        with open(os.path.join(os.getcwd(), 'plots', f'{gait}_distribution_data.pkl'), 'wb') as f:
            pk.dump(ajp, f)

def plot_VP_distribution():
    # joint_names = ['r_hip_x', 'r_hip_z', 'r_hip_y', 'r_knee_y', 'r_ankle_y', 'r_toe']
    num_joints = 6
    _, axes = plt.subplots(1,6, figsize=(12,2.4))
    for j in range(num_joints):
        # axes[j%6].title.set_text(f'Joint: {joint_names[j]}')
        if j//6 == 0:
            axes[j%6].set_xlabel('Time steps', fontsize=20)
        if j%6==0:
            axes[j%6].set_ylabel('Joint angle (rad)', fontsize=20)
    for gait in ['fall', 'walk']:
        if gait == 'fall':
            line_clr = 'r'
            face_clr = 'tomato'
        else:
            line_clr = 'blue'
            face_clr = 'royalblue'
        with open(os.path.join(os.getcwd(), f'{gait}_distribution_data.pkl'), 'rb') as f:
            ajp = pk.load(f)
        for j in range(num_joints):
            axes[j%6].tick_params(axis='x', labelsize=15)
            axes[j%6].tick_params(axis='y', labelsize=15)
            avg = ajp[str(j)].mean(axis=0)
            std = ajp[str(j)].std(axis=0)
            axes[j%6].plot(range(1354), avg, c=line_clr)
            axes[j%6].fill_between(range(1354), avg-3*std, avg+3*std, fc=face_clr, ec='face', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'VP_distribution.pdf'))

def plot_RP_distribution():
    num_joints = 25
    _, axes = plt.subplots(5,5, figsize=(20,20))
    for j in range(num_joints):
        axes[int(j//5)][j%5].title.set_text(f'Joint: {j+1}')
        if j//5==4:
            axes[int(j//5)][j%5].set_xlabel('Time steps')
        if j%5 == 0:
            axes[int(j//5)][j%5].set_ylabel('Joint angle (deg)')
    p = os.path.join(os.path.dirname(os.getcwd()), 'data', 'real_poppy')
    ajp_fall, ajp_walk = {}, {}
    for j in range(num_joints):
        ajp_fall[str(j)] = []
        ajp_walk[str(j)] = []
    for i in range(1, 31):
        positions_temp = []
        with open(os.path.join(p, 'results_%d.pkl' % (-i)), 'rb') as f:
            (_, results, bufs) = pk.load(f, encoding='latin1')
        for k in range(5): # k is the index of each transition
            _, buffers, _ = zip(*bufs[k]) # (flag, buffers, elapsed)
            # joint_angles
            for j in range(len(buffers)):
                positions_temp.append(buffers[j]['position'])
        positions_temp = np.concatenate(positions_temp)

        for j in range(num_joints):
            if results == 4:
                ajp_walk[str(j)].append(positions_temp[:, j])
            else:
                ajp_fall[str(j)].append(positions_temp[:, j])
    for j in range(num_joints):
        ajp_walk[str(j)] = np.array(ajp_walk[str(j)])
        ajp_fall[str(j)] = np.array(ajp_fall[str(j)])
    for j in range(num_joints):
        avg = ajp_fall[str(j)].mean(axis=0)
        std = ajp_fall[str(j)].std(axis=0)
        axes[int(j//5)][j%5].plot(range(300), avg, c='r')
        axes[int(j//5)][j%5].fill_between(range(300), avg-3*std, avg+3*std, fc='tomato', ec='face', alpha=0.5)
        # axes[int(j//5)][j%5].plot(range(299), avg[1:]-avg[:-1], c='tomato', alpha=0.5)

        avg = ajp_walk[str(j)].mean(axis=0)
        std = ajp_walk[str(j)].std(axis=0)
        axes[int(j//5)][j%5].plot(range(300), avg, c='b')
        axes[int(j//5)][j%5].fill_between(range(300), avg-3*std, avg+3*std, fc='royalblue', ec='face', alpha=0.5)
        # axes[int(j//5)][j%5].plot(range(299), avg[1:]-avg[:-1], c='royalblue', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'RP_distribution.pdf'))

def plot_episodes_distribution():
    path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    std_list = [0.01, 0.02, 0.03, 0.09, 0.1, 0.11]
    _, axes = plt.subplots(2,3, figsize=(12, 10))
    for std in std_list:
        i = std_list.index(std)
        axes[i//3][i%3].set_yticks((0.1, 0.7, 0.8), labels=['0.1', '0.7', '0.8'])
        axes[i//3][i%3].set_xticks((0, 675, 1300), labels=['0', '675', '1300'])
        axes[i//3][i%3].set_ylim(0, 0.9)
        axes[i//3][i%3].tick_params(axis='x', labelsize=20)
        axes[i//3][i%3].tick_params(axis='y', labelsize=20)
        # axes[i//3][i%3].set_title(f'{std}', fontsize=fs)
        if i//3==1:
            axes[int(i//3)][i%3].set_xlabel('Time steps', fontsize=40)
        if i%3 == 0:
            axes[int(i//3)][i%3].set_ylabel('z-coordinate', fontsize=40)
        for idx in range(30):
            p = os.path.join(path, str(std), str(idx))
            with open(os.path.join(p, 'hc_z.pkl'), 'rb') as f:
                z_coor = pk.load(f)
            axes[i//3][i%3].plot(range(len(z_coor)), z_coor)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'plots', 'episodes_distribution.pdf'))


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['RP', 'VP', 'episodes'], help='Select from \'RP\', \'VP\', \'episodes\'.')
    args = parser.parse_args('--mode episodes'.split())
    if args.mode == 'RP':
        plot_RP_distribution()
    elif args.mode == 'VP':
        # collect_VP_distribution_data()
        plot_VP_distribution()
    elif args.mode == 'episodes':
        plot_episodes_distribution()