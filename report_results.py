import argparse
import json
from pathlib import Path

import pandas as pd

from common.project import ROOT


def main():
    parser = argparse.ArgumentParser(description='Print bundled lightweight publication results without the dataset or optimization')
    parser.add_argument('--section', choices=['all', 'models', 'predictors', 'iterations', 'final', 'replay'], default='all')
    args = parser.parse_args()
    tables = ROOT / 'docs/results'
    files = {'models': 'model_mechanical_comparison_2s.csv', 'predictors': 'mpc_predictor_comparison.csv',
             'iterations': 'slsqp_iteration_study.csv', 'final': 'final_gru_reference_comparison.csv', 'replay': 'recorded_replay_vs_mpc.csv'}
    for section, filename in files.items():
        if args.section not in ('all', section):
            continue
        print('\n' + section.upper(), flush=True)
        table = pd.read_csv(tables / filename)
        if section in ('predictors', 'final'):
            table = table.set_index('predictor' if section == 'predictors' else 'reference_mode').select_dtypes(include='number').T
        print(table.to_string(float_format=lambda value: f'{value:.8g}'), flush=True)
    print('\nPrimary full-record MPC metrics score every internal sample. Replay comparisons use matching control endpoints.', flush=True)
    print('Two-second surrogate validation and the 200-ms MPC planning horizon are different experiments.', flush=True)


if __name__ == '__main__':
    main()
