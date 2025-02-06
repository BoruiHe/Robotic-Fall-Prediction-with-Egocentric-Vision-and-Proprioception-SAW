import os
import yaml


def print_statistics_quickMLP(dataset):
    path = os.path.join(os.getcwd(), 'exps', dataset, 'quickMLP')
    seeds = [608, 1247, 3224]
    with open(os.path.join(path, 'parameters.yml'), 'r') as f:
        pred_span = yaml.safe_load(f)['prediction_span']
    acc = {}
    print(f'-------{dataset}-------')
    for ps in pred_span:
        acc[ps] = []
        for seed in seeds:
            with open(os.path.join(path, str(seed), f'quickMLP_{ps}.log'), 'r') as f:
                acc[ps].append(float(f.readlines()[-1][:-1]))
        print(f'ps {ps}: {sum(acc[ps])/3}')
    print(acc)


if __name__ == '__main__':
    print_statistics_quickMLP('VP')