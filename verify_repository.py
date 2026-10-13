import argparse
import ast
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parent

def verify():
    broken=[];syntax=[];local_paths=[];commands=[];raw_test=[]
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and 'archive_working_repository' not in p.relative_to(ROOT).parts]
    for path in paths:
        if path.suffix=='.py':
            source=path.read_text(encoding='utf-8')
            try:ast.parse(source,filename=str(path))
            except SyntaxError as exc:syntax.append(f'{path.relative_to(ROOT)}: {exc}')
            if re.search(r'[A-Za-z]:\\(?:Data\\Nitin Project|Users\\)',source):local_paths.append(str(path.relative_to(ROOT)))
        if path.suffix=='.md':
            text=path.read_text(encoding='utf-8')
            for target in re.findall(r'\[[^\]]*\]\(([^)\n]+)\)',text):
                target=target.strip('<>').split('#',1)[0]
                if not target or '://' in target or target.startswith('mailto:'):continue
                if not (path.parent/unquote(target)).exists():broken.append(dict(document=str(path.relative_to(ROOT)),target=target))
            if path.name=='README.md':
                for match in re.finditer(r'^python\s+(?:"([^"]+\.py)"|([^\s]+\.py))',text,re.M):
                    script=match.group(1) or match.group(2)
                    if not (ROOT/script).is_file():commands.append(script)
        if path.suffix in ['.csv','.npy','.npz','.parquet'] and ('Test' in path.parts or 'testing' in path.parts or path.name.startswith('Simulation_Output_')):
            raw_test.append(str(path.relative_to(ROOT)))
    import numpy as np
    import pandas as pd
    neural=ROOT/'Neural Controller'
    split=json.loads((neural/'data_splits/source_split.json').read_text())
    sources=split['sources'];names=list(sources)
    assert all(not set(sources[a])&set(sources[b]) for i,a in enumerate(names) for b in names[i+1:])
    arrays=[np.load(neural/'data_splits'/f'{name}_indices.npy') for name in ['train','validation','internal_test']]
    assert [len(a) for a in arrays]==[17500,3750,3750]
    assert len(set(np.concatenate(arrays)))==25000
    schema=json.loads((neural/'dataset/metadata/feature_schema.json').read_text())
    assert [f['index'] for f in schema['supervised_inputs']]==list(range(32))
    assert len(schema['supervised_targets'])==2 and not schema['predictor_memory_is_supervised']
    for published,authority in [('dynamics_recursive_results.csv','model_mechanical_comparison_2s.csv'),('mpc_final_results.csv','final_gru_reference_comparison.csv')]:
        assert (ROOT/'docs/results'/published).read_bytes()==(ROOT/'docs/results'/authority).read_bytes()
    for published,authority in [('neural_controller_results.csv',neural/'results/final_comparison.csv'),('improvement_study_results.csv',neural/'results/improvement_study/final_improvement_comparison.csv')]:
        assert (ROOT/'docs/results'/published).read_bytes()==authority.read_bytes()
    teacher=json.loads((neural/'dataset/summaries/dataset_summary.json').read_text())
    table=pd.read_csv(ROOT/'docs/results/teacher_dataset_summary.csv').iloc[0]
    for key in ['accepted_samples','episodes','source_recordings_represented','rejected_samples','duplicate_episode_step_count','invalid_accepted_samples']:assert table[key]==teacher[key]
    timing=json.loads((neural/'results/realtime/realtime_metrics.json').read_text())
    table=pd.read_csv(ROOT/'docs/results/neural_latency_results.csv').set_index('controller')
    for row in timing:
        for key in ['mean_ms','median_ms','p95_ms','p99_ms','maximum_ms','deadline_miss_percent']:assert np.isclose(table.loc[row['controller'],key],row[key],rtol=1e-13,atol=1e-14)
    for artifact in json.loads((neural/'configs/release_checkpoint_manifest.json').read_text()).values():
        assert hashlib.sha256((ROOT/artifact['checkpoint']).read_bytes()).hexdigest()==artifact['sha256']
    missing=[]
    for package in ['numpy','pandas','scipy','torch','numba','matplotlib','pyarrow']:
        try:importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:missing.append(package)
    result=dict(files_checked=len(paths),broken_markdown_links=broken,missing_readme_scripts=commands,syntax_errors=syntax,
                hard_coded_runtime_local_paths=local_paths,missing_dependencies=missing,raw_external_Test_files=raw_test,
                source_leakage=0,split_samples=[17500,3750,3750],numerical_tables_match_authority=True,checkpoint_hashes_match=True,
                new_scientific_experiments_run=False,git_commit=False,git_push=False)
    print(json.dumps(result,indent=2))
    assert not any([broken,commands,syntax,local_paths,missing,raw_test]),'Repository verification failed'
    return result

def main():
    parser=argparse.ArgumentParser(description='Check links, source syntax, split/schema, released hashes and static result consistency; no inference or experiments')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=verify()
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
