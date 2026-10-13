import argparse
import os
from pathlib import Path
import shutil
import sys

RELEASE=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description='Reproduce neural-policy training/evaluation into a new local output; released evidence stays read-only')
    parser.add_argument('--stage',choices=['all','audit','train','evaluate'],default='all')
    parser.add_argument('--data-root',type=Path,help='Directory containing raw Train/; external Test is never read')
    parser.add_argument('--preprocessed-dir',type=Path,help='Prepared scaler.json, training_manifest.json and training/*.npy')
    parser.add_argument('--dataset-dir',type=Path,default=Path(os.environ.get('MPC_TEACHER_DATASET_DIR',RELEASE/'dataset/generated')))
    parser.add_argument('--output-dir',type=Path,default=RELEASE.parent/'results/neural_reproduction')
    args=parser.parse_args()
    if args.data_root:os.environ['IMOTION_DATA_ROOT']=str(args.data_root.resolve())
    if args.preprocessed_dir:os.environ['IMOTION_PREPROCESSED_DIR']=str(args.preprocessed_dir.resolve())
    dataset=args.dataset_dir.resolve();output=args.output_dir.resolve()
    if output.is_relative_to(RELEASE):parser.error('Output must be outside the released Neural Controller/ directory')
    if not (dataset/'arrays/X_physical.npy').is_file():parser.error('Generate mpc_teacher_v1 first, or set --dataset-dir to an existing complete corpus')
    marker=output/'neural_reproduction.json'
    if output.exists() and any(output.iterdir()) and not marker.exists():parser.error('Nonempty output is not a neural reproduction directory; choose a fresh --output-dir')
    output.mkdir(parents=True,exist_ok=True)
    if not marker.exists():
        import json
        marker.write_text(json.dumps(dict(stage='neural policy reproduction',released_evidence_unchanged=True),indent=2),encoding='utf-8')
    os.environ['MPC_TEACHER_DATASET_DIR']=str(dataset);os.environ['NEURAL_OUTPUT_DIR']=str(output)
    for folder in ['configs','data_splits']:
        target=output/folder;target.mkdir(parents=True,exist_ok=True)
        for source in (RELEASE/folder).iterdir():
            if source.is_file() and not (target/source.name).exists():shutil.copy2(source,target/source.name)
    import numpy as np
    np.save(output/'data_splits/Y_command_nominal.npy',np.clip(np.load(dataset/'arrays/Y_command.npy'),-1,1),allow_pickle=False)
    if args.stage=='evaluate':
        for architecture in ['mlp','gru']:
            target=output/'models'/architecture
            if not target.exists():shutil.copytree(RELEASE/'models'/architecture,target)
    sys.argv=[sys.argv[0],'--stage',args.stage]
    from experiment import experiment_main
    experiment_main()
    if args.stage in ['all','evaluate']:
        from finalize_results import main as finalize
        finalize()

if __name__=='__main__':main()
