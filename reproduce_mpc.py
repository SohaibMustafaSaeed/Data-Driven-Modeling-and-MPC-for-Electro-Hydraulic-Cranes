import argparse
import json
from pathlib import Path
import subprocess
import sys

from common.project import ROOT, RAW_DATA_DIR, PREPARED_DIR


PROFILES = {
    'gru_compare': ('gru', 5, False, 'plant-generated'),
    'compact_compare': ('compact', 5, False, 'plant-generated'),
    'iteration5': ('gru', 5, True, 'plant-generated'),
    'iteration10': ('gru', 10, True, 'plant-generated'),
    'iteration20': ('gru', 20, True, 'plant-generated'),
    'reachable': ('gru', 5, False, 'plant-generated'),
    'recorded': ('gru', 5, False, 'recorded'),
}


def build_command(profile, output=None, no_show=False):
    config = json.loads((ROOT / 'MPC offline Model/configs/final_gru_mpc.json').read_text(encoding='utf-8'))
    predictor, maxiter, segment, reference = PROFILES[profile]
    checkpoint = config['checkpoint'] if predictor == 'gru' else config['compact_checkpoint']
    output = Path(output).resolve() if output else ROOT / 'MPC offline Model/results' / profile
    command = [sys.executable, str(ROOT / 'MPC offline Model/offline_mpc.py'), '--checkpoint', str(ROOT / checkpoint),
               '--mechanical-checkpoint', str(ROOT / config['mechanical_checkpoint']), '--test-file', str(RAW_DATA_DIR / 'Test' / config['test_file']),
               '--dataset-dir', str(PREPARED_DIR), '--output-dir', str(output), '--geometry-file', str(ROOT / 'MPC offline Model/geometry_patu.json'),
               '--tracking-space', 'cartesian', '--reference-mode', reference, '--maxiter', str(maxiter)]
    settings = config['arguments']
    for key in ['block_steps', 'horizon', 'moves', 'fd_step', 'ftol', 'position_scale', 'cartesian_velocity_scale', 'position_weight',
                'cartesian_velocity_weight', 'terminal_position_weight', 'terminal_cartesian_velocity_weight', 'angle_weight',
                'velocity_weight', 'terminal_angle_weight', 'terminal_velocity_weight', 'effort_weight', 'smoothness_weight',
                'threads', 'progress_every', 'reachability_angle_tolerance', 'reachability_velocity_tolerance', 'bound_samples_per_file']:
        command += ['--' + key.replace('_', '-'), str(settings[key])]
    command += ['--neutral-command', *map(str, settings['neutral_command'])]
    for index, name in enumerate(['lift', 'jib']):
        command += ['--' + name + '-bounds', str(config['command_bounds']['low'][index]), str(config['command_bounds']['high'][index])]
    if segment:
        command += ['--duration', str(config['iteration_segment']['duration']), '--start-seconds', str(config['iteration_segment']['start_seconds'])]
    else:
        command += ['--full-recording']
    if reference == 'recorded':
        command += ['--allow-reference-mismatch']
    if no_show:
        command += ['--no-show']
    return command, output


def main():
    parser = argparse.ArgumentParser(description='Reproduce one frozen MPC experiment; never starts another experiment automatically')
    parser.add_argument('--experiment', choices=PROFILES, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--print-command-only', action='store_true')
    args = parser.parse_args()
    command, output = build_command(args.experiment, args.output_dir, args.no_show)
    print(subprocess.list2cmdline(command), flush=True)
    if args.print_command_only:
        return
    if output.is_dir() and any(output.iterdir()):
        parser.error('Result directory is nonempty. Reopen it or specify a fresh --output-dir; no automatic overwrite.')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
