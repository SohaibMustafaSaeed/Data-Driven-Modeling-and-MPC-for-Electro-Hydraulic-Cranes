import argparse
import json
from pathlib import Path

from policy_runtime import Policy


def main():
    parser=argparse.ArgumentParser(description='Standalone CPU policy: 32 physical inputs -> two bounded physical commands')
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--inputs-json',type=Path,required=True,help='JSON array of 32 values, or a sequence of arrays for stateful GRU use')
    args=parser.parse_args()
    policy=Policy(args.model);inputs=json.loads(args.inputs_json.read_text())
    policy.reset()
    if inputs and isinstance(inputs[0],list):commands=[policy.step(row).tolist() for row in inputs]
    else:commands=policy.step(inputs).tolist()
    print(json.dumps(dict(architecture=policy.architecture,command_V=commands)))


if __name__=='__main__':main()
