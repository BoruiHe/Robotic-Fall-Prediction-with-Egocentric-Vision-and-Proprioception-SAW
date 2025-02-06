import os
import numpy as np
from rich import print
from rich.table import Table
from scipy.stats import ttest_ind_from_stats


def print_VP_sigtables():
    path = os.path.join(os.getcwd(), 'exps', 'VP')
    seeds = [608, 1247, 3224]
    exp_name =  ['ours_siMLPe'] # ,  'ours_siMLPe_pos'
    pred_span = [10, 20, 30, 40, 50, 60]
    sig_ave, sig_var = {}, {}

    print('[bold red]siMLPe[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy, var = {}, {}
    for exp in exp_name:
        try:
            accuracy[exp], var[exp] = {}, {}
            for ps in pred_span:
                accuracy[exp][ps] = []
                var[exp][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_S.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[exp][int(ps)].append(float(line.split()[1]))
            for ps in pred_span:
                var[exp][ps] = np.var(accuracy[exp][ps])
                accuracy[exp][ps] = np.mean(accuracy[exp][ps])
            table_acc.add_row(exp, *['{}'.format(var[exp][ps]) for ps in pred_span])
        except:
            pass
        if 'pos' in exp:
            sig_ave['M*'] = accuracy
            sig_var['M*'] = var
        else:
            sig_ave['M'] = accuracy
            sig_var['M'] = var
    print(table_acc)
    print('\n')
    

    print('[bold red]siMLPe + Traj[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy, var = {}, {}
    for exp in exp_name:
        try:
            accuracy[exp], var[exp] = {}, {}
            for ps in pred_span:
                accuracy[exp][ps] = []
                var[exp][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_ST.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[exp][int(ps)].append(float(line.split()[1]))
            for ps in pred_span:
                var[exp][ps] = np.var(accuracy[exp][ps])
                accuracy[exp][ps] = np.mean(accuracy[exp][ps])
            table_acc.add_row(exp, *['{}'.format(var[exp][ps]) for ps in pred_span])
        except:
            pass
        if 'pos' in exp:
            sig_ave['MT*'] = accuracy
            sig_var['MT*'] = var
        else:
            sig_ave['MT'] = accuracy
            sig_var['MT'] = var
    print(table_acc)
    print('\n')

    print('[bold red]siMLPe + Popeyes[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy, var = {}, {}
    for exp in exp_name:
        try:
            accuracy[exp], var[exp] = {}, {}
            for ps in pred_span:
                accuracy[exp][ps] = []
                var[exp][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_SP.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[exp][int(ps)].append(float(line.split()[1]))
            for ps in pred_span:
                var[exp][ps] = np.var(accuracy[exp][ps])
                accuracy[exp][ps] = np.mean(accuracy[exp][ps])
            table_acc.add_row(exp, *['{}'.format(var[exp][ps]) for ps in pred_span])
        except:
            pass
        if 'pos' in exp:
            sig_ave['MV*'] = accuracy
            sig_var['MV*'] = var
        else:
            sig_ave['MV'] = accuracy
            sig_var['MV'] = var
    print(table_acc)
    print('\n')

    print('[bold red]siMLPe + Popeyes + Traj[/bold red]')
    table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
    table_acc.add_column('LR\\PS')
    for ps in pred_span:
        table_acc.add_column(str(ps), width=12, justify='center')

    accuracy, var = {}, {}
    for exp in exp_name:
        try:
            accuracy[exp], var[exp] = {}, {}
            for ps in pred_span:
                accuracy[exp][ps] = []
                var[exp][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_SPT.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[exp][int(ps)].append(float(line.split()[1]))
            for ps in pred_span:
                var[exp][ps] = np.var(accuracy[exp][ps])
                accuracy[exp][ps] = np.mean(accuracy[exp][ps])
            table_acc.add_row(exp, *['{}'.format(float(var[exp][ps])) for ps in pred_span])
        except:
            pass
        if 'pos' in exp:
            sig_ave['MTV*'] = accuracy
            sig_var['MTV*'] = var
        else:
            sig_ave['MTV'] = accuracy
            sig_var['MTV'] = var
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
            accuracy[exp] = {}
            for ps in pred_span:
                accuracy[exp][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_P.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[exp][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                var[exp][ps] = np.var(accuracy[exp][ps])
                accuracy[exp][ps] = np.mean(accuracy[exp][ps])
            table_acc.add_row(exp, *['{}'.format(float(var[exp][ps])) for ps in pred_span])
        except:
            pass
        if 'pos' in exp:
            sig_ave['V*'] = accuracy
            sig_var['V*'] = var
        else:
            sig_ave['V'] = accuracy
            sig_var['V'] = var
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
            accuracy[exp] = {}
            for ps in pred_span:
                accuracy[exp][ps] = 0
            for seed in seeds:
                with open(os.path.join(path, exp, str(seed), 'clf_PT.txt'), 'r') as f:
                    for line in f.readlines():
                        if 'prediction span:' in line:
                            ps = line.split(' ')[3]
                        if 'accuracy' in line:
                            accuracy[exp][int(ps)] += float(line.split()[1])
            for ps in pred_span:
                var[exp][ps] = np.var(accuracy[exp][ps])
                accuracy[exp][ps] = np.mean(accuracy[exp][ps])
            table_acc.add_row(exp, *['{}'.format(float(var[exp][ps])) for ps in pred_span])
        except:
            pass
        if 'pos' in exp:
            sig_ave['TV*'] = accuracy
            sig_var['TV*'] = var
        else:
            sig_ave['TV'] = accuracy
            sig_var['TV'] = var
    print(table_acc)
    print('\n')


    best_jot = ['MTV', 'MTV', 'MTV', 'MTV', 'MTV', 'MTV']
    best_pos = ['MTV*', 'MTV*', 'MTV*', 'MTV*', 'MTV*', 'MTV*']
    for best, ps in zip(best_jot, pred_span):
        if '*' in best:
            best_model = 'ours_siMLPe_pos'
            best_index = best[:-1]
        else:
            best_model = 'ours_siMLPe'
            best_index = best
        print(f'----{ps}: {best, best_index, best_model}----')
        rest = list(sig_ave.keys())
        rest.remove(best)
        for plan in rest:
            for exp in exp_name:
                # if 'pos' in exp:
                #     curr = plan+'*'
                # else:
                #     curr = plan
                # if not (best == curr):
                print(best, 'VS', plan)
                print(ttest_ind_from_stats(
                    mean1=sig_ave[best]['ours_siMLPe'][ps], std1=np.sqrt(sig_var[best]['ours_siMLPe'][ps]), nobs1=3,
                    mean2=sig_ave[plan][exp][ps], std2=np.sqrt(sig_var[plan][exp][ps]), nobs2=3,
                    equal_var= False,
                    alternative= 'greater'
                    ))
                    

if __name__ == '__main__':
    print_VP_sigtables()