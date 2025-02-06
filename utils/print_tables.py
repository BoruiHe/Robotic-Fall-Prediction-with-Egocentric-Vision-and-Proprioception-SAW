import os
import numpy as np
from rich import print
from rich.table import Table


def print_RP_tables():
    path = os.path.join(os.getcwd(), 'exps', 'RP')
    seeds = [608, 1247, 3224]
    exp_name = ['Popeyes', 'Tps_debugging_5e-7_wd1e-5']
    # ['baseline_siMLPe', 'ours_siMLPe', 'Tps_debugging_5e-7_wd1e-3']
    # ['Tps_debugging_5e-7_wd1e-3', 'Tps_debugging_5e-7_wd1e-4']
    pred_span = [4, 8, 12, 16, 20, 24]

    print('[bold red]siMLPe[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy = {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr] = {}
            for ps in pred_span:
                accuracy[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_S.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                accuracy[lr][ps] /= len(seeds)

            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]siMLPe + Traj[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy = {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr] = {}
            for ps in pred_span:
                accuracy[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_ST.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                accuracy[lr][ps] /= len(seeds)

            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]siMLPe + Popeyes[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy = {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr] = {}
            for ps in pred_span:
                accuracy[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_SP.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                accuracy[lr][ps] /= len(seeds)

            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]siMLPe + Popeyes + Traj[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy = {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr] = {}
            for ps in pred_span:
                accuracy[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_SPT.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                accuracy[lr][ps] /= len(seeds)

            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]Popeyes[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy = {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr] = {}
            for ps in pred_span:
                accuracy[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_P.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                accuracy[lr][ps] /= len(seeds)

            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]Popeyes + Traj[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy = {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr] = {}
            for ps in pred_span:
                accuracy[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_PT.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                accuracy[lr][ps] /= len(seeds)

            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

def print_VP_tables():
    path = os.path.join(os.getcwd(), 'exps', 'VP')
    seeds = [608, 1247, 3224]
    # exp_name =  [] # ['baseline_siMLPe', 'baseline_siMLPe_pos', 'ours_siMLPe', 'ours_siMLPe_pos', 'Popeyes_1e-4']
    exp_name = ['scenes_20uu_all', 'scenes_40uu_all', 'scenes_60uu_all', 'scenes_80uu_all', 'scenes_100uu_all']
    pred_span = [10, 20, 30, 40, 50, 60]

    print('[bold red]siMLPe[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy, std = {}, {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr], std[lr] = {}, {}
            for ps in pred_span:
                accuracy[lr][ps] = []
                std[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_S.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)].append(float(line.split()[1]))
            for ps in pred_span:
                std[lr][ps] = np.std(accuracy[lr][ps])
                accuracy[lr][ps] = np.mean(accuracy[lr][ps])
            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]siMLPe + Traj[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy, std = {}, {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr], std[lr] = {}, {}
            for ps in pred_span:
                accuracy[lr][ps] = []
                std[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_ST.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)].append(float(line.split()[1]))
            for ps in pred_span:
                std[lr][ps] = np.std(accuracy[lr][ps])
                accuracy[lr][ps] = np.mean(accuracy[lr][ps])
            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]siMLPe + Popeyes[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy, std = {}, {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr], std[lr] = {}, {}
            for ps in pred_span:
                accuracy[lr][ps] = []
                std[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_SP.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)].append(float(line.split()[1]))
            for ps in pred_span:
                std[lr][ps] = np.std(accuracy[lr][ps])
                accuracy[lr][ps] = np.mean(accuracy[lr][ps])
            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]siMLPe + Popeyes + Traj[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy, std = {}, {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr], std[lr] = {}, {}
            for ps in pred_span:
                accuracy[lr][ps] = []
                std[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_SPT_JA.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)].append(float(line.split()[1]))
            for ps in pred_span:
                std[lr][ps] = np.std(accuracy[lr][ps])
                accuracy[lr][ps] = np.mean(accuracy[lr][ps])
            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]Popeyes[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy = {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr] = {}
            for ps in pred_span:
                accuracy[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_P.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                accuracy[lr][ps] /= len(seeds)

            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')

    print('[bold red]Popeyes + Traj[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy = {}
    for exp in exp_name:
        try:
            lr = exp
            accuracy[lr] = {}
            for ps in pred_span:
                accuracy[lr][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_PT.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[lr][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                accuracy[lr][ps] /= len(seeds)

            table_acc.add_row(lr, *['{:.2f}'.format(round(accuracy[lr][ps]*100, 2)) for ps in pred_span])
        except:
            pass
    print(table_acc)
    print('\n')


if __name__ == '__main__':
    # print_RP_tables()
    print_VP_tables()