"""Reproduce representative lost-success episodes without changing any policy."""
import argparse
import json
from pathlib import Path
import sys
from dendritic_nav.runners.launch_dend import launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    experiment = Path(args.experiment).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    checkpoint = experiment/"learner/checkpoints/pilot_final.ckpt"
    jobs=[]
    for case_index,(actor,episodes) in enumerate(((5,1),(6,1),(7,1),(16,4))):
        for baseline in (True,False):
            name="{}_actor_{:03d}".format("baseline" if baseline else "dend",actor)
            gpu=2*case_index+int(not baseline)
            command=[sys.executable,"-u","-m","dendritic_nav.diagnostics.trace_dend",
                     "--experiment",str(experiment),"--output",str(output/name),
                     "--role","dev","--actor-id",str(actor),"--actors","20",
                     "--episodes",str(episodes),"--steps","2500","--seed",str(123+actor),
                     "--checkpoint",str(checkpoint),"--visualize-every","1"]
            if baseline:command.append("--baseline")
            jobs.append(launch(command,gpu,output/(name+".log"),root))
            (output/"launch.json").write_text(json.dumps(dict(jobs=jobs),indent=2))
    print(json.dumps(dict(output=str(output),jobs=jobs),indent=2))


if __name__=="__main__":main()
