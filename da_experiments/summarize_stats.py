from functools import partial
import argparse
from pathlib import Path
import json
import pandas as pd
from utils import compute_confidence_interval
from collections import defaultdict
from typing import Any
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--folder', type=str, default='results/',
                        help='Folder to search')
    parser.add_argument('--simu_params', nargs='+', type=str,
                        default=['source_data', 'target_data', 'method'])
    parser.add_argument('--group_by', type=str,
                        default='method')
    parser.add_argument('--out_type', type=str,
                        default='latex')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    # Recover all files that match .npy pattern in folder/
    p = Path(args.folder)
    all_files = p.glob('**/*.csv')

    # Group files by metric name
    files_dic = nested_default_dict(1, list)
    for path in all_files:
        root = path.parent
        with open(root / 'config.json') as f:
            config = json.load(f)
        df = pd.read_csv(path)
        best_test_acc = df['best_test_acc'][0]
        key = tuple([config[key] for key in args.simu_params])
        files_dic[key].append(best_test_acc)

    # Do one plot per metric
    all_records = []
    for exp, all_tests_acc in files_dic.items():
        rec = {key: val for key, val in zip(args.simu_params, exp)}
        # print(all_tests_acc)
        mean, std = compute_confidence_interval(all_tests_acc, axis=0)
        mean = np.round(mean * 100, 2)
        std = np.round(std * 100, 2)
        rec['mean'] = mean
        rec['std'] = std
        rec['# runs'] = len(all_tests_acc)
        all_records.append(rec)
    df = pd.DataFrame(all_records)
    if args.out_type == 'latex':
        ndf = df.groupby(['method']).agg(lambda x: list(x))
        for method in ndf.index.to_list():
            for source, target, mean, std in zip(ndf.loc[method, 'source_data'], ndf.loc[method, 'target_data'],
                                                 ndf.loc[method, 'mean'], ndf.loc[method, 'std']):
                ndf.loc[method, f'{source} -> {target}'] = f'{mean} (\pm {std})'
            ndf.loc[method, 'Mean'] = '{} (\pm {})'.format(np.round(np.mean(ndf.loc[method, 'mean']), 2),
                                                           np.round(np.mean(ndf.loc[method, 'std']), 2))
        ndf = ndf.drop(['mean', 'std', 'source_data', 'target_data', '# runs'], axis=1)
        print(ndf)
    elif args.out_type == 'plot':
        df = df.groupby([args.group_by]).mean()
        res_dic = df.to_dict()
        x = np.array(list(res_dic['mean'].keys()))
        y_means = np.array(list(res_dic['mean'].values()))
        y_std = np.array(list(res_dic['std'].values()))
        plt.plot(x, y_means)
        plt.fill_between(x, y_means - y_std, y_means + y_std, alpha=0.2)
        plt.xlabel(args.group_by)
        plt.ylabel('Accuracy')
        plt.show()


def nested_default_dict(depth: int, final_type: Any, i: int = 1):
    if i == depth:
        return defaultdict(final_type)
    fn = partial(nested_default_dict, depth=depth, final_type=final_type, i=i+1)
    return defaultdict(fn)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
