import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_TVT(dataset, testing_only=False):
    if dataset == 'VP':
        ps_list = ['10', '20', '30', '40', '50', '60']
        multiplier = -1.5
        name_list = ['JA_3e-3', 'JA_3e-4', 'JA_3e-5', 'pos_1e-3', 'pos_1e-4', 'pos_1e-5'] # 
        folder_list = ['ours_siMLPe_3e-3', 'ours_siMLPe', 'ours_siMLPe_3e-5', 'ours_siMLPe_pos_1e-3', 'ours_siMLPe_pos', 'ours_siMLPe_pos_1e-5'] #  
    elif dataset == 'RP':
        ps_list = ['4', '8', '12', '16', '20', '24']
        multiplier = -0
        name_list = ['JA_3e-3', 'JA_3e-4', 'JA_3e-5']
        # ['PS_1e-4', 'PS_1e-5', 'PS_1e-6', 'PS_1e-7']
        folder_list = ['ours_siMLPe_3e-3', 'ours_siMLPe', 'ours_siMLPe_3e-5']
        # ['PS_1e-4_sp', 'PS_1e-5_sp', 'PS_1e-6_sp', 'PS_1e-7_sp']

    seed_list = ['608', '1247', '3224']

    log_dict = {}
    dict_for_plotting = {}
    for name, date in zip(name_list, folder_list):
        log_dict[name] = {}
        dict_for_plotting[name] = {
            'Iter': [],
            'Training loss': 0,
            'Validation loss': 0,
            'Testing results': 0,
            'Testing STD': 0
        }
        for seed in seed_list:
            log_dict[name][seed] = {
                'Iter': [],
                'Training loss': [],
                'Validation loss': [],
                'Testing results': [],
                'Testing STD': []
            }

            path = os.path.join(os.getcwd(), 'exps', dataset, date, seed, 'siMLPe.log')
            with open(path, 'r') as fp:
                for line in fp.readlines():
                    if 'Iter' in line:
                        log_dict[name][seed]['Iter'].append(int(line.split('\n')[0].split(' ')[5]))
                    elif 'loss' in line:
                        log_dict[name][seed]['Training loss'].append(float(line.split('\n')[0].split('\t')[-2].split(': ')[1]))
                        log_dict[name][seed]['Validation loss'].append(float(line.split('\n')[0].split('\t')[-1].split(': ')[1]))
            path = os.path.join(os.getcwd(), 'exps', dataset, date, seed, 'siMLPe.log')
            with open(path, 'r') as fp:
                for line in fp.readlines():
                    if (']' in line) and not ('[test]' in line) and not ('[train]' in line):
                        results = line.split()[-1].split(']')[0]
                        log_dict[name][seed]['Testing results'].append(float(results))
            
            path = os.path.join(os.getcwd(), 'exps', dataset, date, seed, 'siMLPe_std.log')
            with open(path, 'r') as fp1:
                for line in fp1.readlines():
                    if (']' in line) and not ('[test]' in line):
                        results = line.split()[-1].split(']')[0]
                        log_dict[name][seed]['Testing STD'].append(float(results))

            dict_for_plotting[name]['Training loss'] += np.array(log_dict[name][seed]['Training loss'])
            dict_for_plotting[name]['Validation loss'] += np.array(log_dict[name][seed]['Validation loss'])
            dict_for_plotting[name]['Testing results'] += np.array(log_dict[name][seed]['Testing results'])
            dict_for_plotting[name]['Testing STD'] += np.array(log_dict[name][seed]['Testing STD'])

            # assert len(log_dict[name][seed]['Iter']) == 500, 'The number of iterations is less than 50000'
        dict_for_plotting[name]['Iter'] = log_dict[name][seed]['Iter']
        dict_for_plotting[name]['Training loss'] /= len(seed_list)
        dict_for_plotting[name]['Validation loss'] /= len(seed_list)
        dict_for_plotting[name]['Testing results'] /= len(seed_list)
        dict_for_plotting[name]['Testing STD'] /= len(seed_list)

    results_dict, std_dict = {}, {}
    for lr in name_list:
        results_dict[lr] = dict_for_plotting[lr]['Testing results']
        std_dict[lr] = dict_for_plotting[lr]['Testing STD']

    colors = ['silver', 'darkorange', 'gold', 'lightseagreen', 'dodgerblue', 'mediumpurple']
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 30,
            }
    
    if not testing_only:
        plt.figure(figsize=(20,10))
        ax = plt.subplot(131)
        plt.title('Training Loss', fontdict=font)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for name, cor in zip(name_list, colors):
            plt.plot(dict_for_plotting[name]['Iter'], dict_for_plotting[name]['Training loss'], linewidth=1, color=cor, label=name[3:])
        plt.xlabel('Iterations', fontdict=font)
        plt.ylabel('Loss', fontdict=font)
        plt.legend(title='Learning rates', handles=[Patch(facecolor=color, label=label[3:]) for label, color in zip(name_list, colors)])

        ax = plt.subplot(132)
        plt.title('Validation Loss', fontdict=font)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for name, cor in zip(name_list, colors):
            plt.plot(dict_for_plotting[name]['Iter'], dict_for_plotting[name]['Validation loss'], linewidth=1, color=cor, label=name[3:])
        plt.xlabel('Iterations', fontdict=font)
        plt.ylabel('Loss', fontdict=font)
        plt.legend(title='Learning rates', handles=[Patch(facecolor=color, label=label[3:]) for label, color in zip(name_list, colors)])

        ax = plt.subplot(133)
    else:
        plt.figure(figsize=(12, 10))
        ax = plt.subplot()

    width = 0.1
    x = np.arange(len(ps_list))
    # plt.title('Testing results', fontdict=font)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=font['size'])
    ax.tick_params(axis='y', labelsize=font['size'])
    ax.yaxis.offsetText.set_fontsize(font['size'])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,-1), useMathText=True)
    for ((name, measurement), std, cor) in zip(results_dict.items(), std_dict.values(), colors):
        offset = width * multiplier
        lower_std = []
        for i in range(6):
            if std[i]>=measurement[i]:
                lower_std.append(measurement[i])
            else:
                lower_std.append(std[i])
        ax.bar(x + offset, measurement, width=width, color=cor, label=name) # , yerr=np.stack((np.array(lower_std).astype('float32'), std))
        multiplier += 1
    ax.set_xticks(x+width, ps_list)
    plt.xlabel('Prediction spans', fontdict=font)
    plt.ylabel('MSE', fontdict=font)
    plt.legend(title='Learning rates', loc='upper left', fontsize=40, title_fontsize=40)

    plt.tight_layout()
    # plt.show(block=True)
    plt.savefig(os.path.join(os.getcwd(), 'plots', dataset+'_siMLPe_lr.pdf'))


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plot_TVT('RP', testing_only=True)