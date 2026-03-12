import os
import numpy as np
import pickle as pk
import yaml
from rich import print
from rich.table import Table


print('[bold red]L scheme[/bold red]')
table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
table_acc.add_column('')
table_acc.add_column('Accuracy', width=12, justify='center')
table_acc.add_column('# of Falling', width=12, justify='center')
table_acc.add_column('# of Walking', width=12, justify='center')

for exp in ['EGOFALLS_L300_re', 'EGOFALLS_L240_re', 'EGOFALLS_L180_re', 'EGOFALLS_L120_re', 'EGOFALLS_L60_re']:
    # print(f'\nSequence length: {exp[-6:-3]}')
    if '300' in exp:
        table_acc.add_row(exp[9:-3], '{:.2f}'.format(round(0.6*100, 2)), str(20), str(10))
    else:
        path = os.path.join('exps', 'RP', exp)
        with open(os.path.join(path, 'parameters.yml'), 'rb') as f:
            hp = yaml.safe_load(f)
        with open(os.path.join(path, 'score.pkl'), 'rb') as f:
            score = pk.load(f)
        assert hp['seeds'] == list(score.keys())

        for seed in hp['seeds']:
            a = np.load(os.path.join(path, str(seed), 'feats_labs_iter.npz'), allow_pickle=True)

        for mode in ['iter', 'flat']:
            acc = 0
            for seed in hp['seeds']:
                acc += score[seed][mode][0][1]
            acc /= len(hp['seeds'])
            # print(acc)
        table_acc.add_row(exp[9:-3], '{:.2f}'.format(round(acc*100, 2)), str((a['labels'] == 1).sum()), str((a['labels'] != 1).sum()))
print(table_acc, '\n')


print('[bold red]N scheme[/bold red]')
table_acc = Table(show_header=True, show_edge=False, show_lines=False, header_style='bold magenta')
table_acc.add_column('')
table_acc.add_column('Accuracy', width=12, justify='center')
table_acc.add_column('# of Falling', width=12, justify='center')
table_acc.add_column('# of Walking', width=12, justify='center')

for exp in ['EGOFALLS_N180_re', 'EGOFALLS_N210_re', 'EGOFALLS_N240_re', 'EGOFALLS_N270_re', 'EGOFALLS_N300_re']: 
    # print(f'\nSequence length: {exp[-6:-3]}')
    if '300' in exp:
        table_acc.add_row(exp[9:-3], '{:.2f}'.format(round(0.6*100, 2)), str(20), str(10))
    else:
        path = os.path.join('exps', 'RP', exp)
        with open(os.path.join(path, 'parameters.yml'), 'rb') as f:
            hp = yaml.safe_load(f)
        with open(os.path.join(path, 'score.pkl'), 'rb') as f:
            score = pk.load(f)
        assert hp['seeds'] == list(score.keys())

        for seed in hp['seeds']:
            a = np.load(os.path.join(path, str(seed), 'feats_labs_iter.npz'), allow_pickle=True)
            # print('falling: {}, walking: {}'.format((a['labels'] == 1).sum(), (a['labels'] != 1).sum()))
            # a = np.load(os.path.join(path, str(seed), 'feats_labs_iter.npz'), allow_pickle=True)
            # print('falling: {}, walking: {}'.format((a['labels'] == 1).sum(), (a['labels'] != 1).sum()))

        for mode in ['iter', 'flat']:
            acc = 0
            for seed in hp['seeds']:
                acc += score[seed][mode][0][1]
            acc /= len(hp['seeds'])
            # print(acc)
        table_acc.add_row(exp[9:-3], '{:.2f}'.format(round(acc*100, 2)), str((a['labels'] == 1).sum()), str((a['labels'] != 1).sum()))
print(table_acc, '\n')
pass