"""Run MP3D with the SNN wave planner and no weighted-A* fallback."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dependencies/astar_pycpp"))

from poni.runtime_neuron import add_neuron_arguments, export_options


def main():
    parser = argparse.ArgumentParser(add_help=False)
    add_neuron_arguments(parser)
    options, original_args = parser.parse_known_args()
    if options.neuron_mode != "neuron":
        parser.error("the strict entry point requires --neuron-mode neuron")
    export_options(options)
    sys.argv = [sys.argv[0]] + original_args
    import utils.planners as planners
    from utils.planners_strict_neuron import (
        PlannerActorSequentialStrictNeuron,
        PlannerActorVectorStrictNeuron,
    )
    planners.PlannerActorSequential = PlannerActorSequentialStrictNeuron
    planners.PlannerActorVector = PlannerActorVectorStrictNeuron
    from eval_poni import main as original_main
    from poni.wavefront_neuron import warmup
    warmup()
    return original_main()


if __name__ == "__main__":
    main()
