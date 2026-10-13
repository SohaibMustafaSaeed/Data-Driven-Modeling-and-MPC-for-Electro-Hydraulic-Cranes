import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.neural_controller import CraneNeuralController


def main():
    fixture = json.loads((Path(__file__).parent / 'tests/equivalence_fixture.json').read_text())
    controller = CraneNeuralController()
    controller.reset()
    print('API demonstration using a tiny saved fixture; no external plant is simulated.')
    for inputs in fixture['episodes'][0]['inputs'][:3]:
        command = controller.step(inputs[:22], inputs[22:24], inputs[24:28], inputs[28:32])
        print('u_lift, u_jib [V]:', command)


if __name__ == '__main__':
    main()
