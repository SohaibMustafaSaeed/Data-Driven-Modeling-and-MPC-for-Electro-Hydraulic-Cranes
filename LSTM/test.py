import sys
from pathlib import Path
import runpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project import ROOT
if __name__ == '__main__':
    sys.argv[1:1] = ['--architecture', 'lstm4']
    runpy.run_path(str(ROOT / 'common/evaluate.py'), run_name='__main__')
