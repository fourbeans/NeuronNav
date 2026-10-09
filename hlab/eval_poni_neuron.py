"""Opt-in MP3D evaluator; both sequential and multiprocessing planners adapted."""
import argparse
from pathlib import Path
import sys

# Match the project-local pyastar dependency used by the validation scripts.
# This only configures imports in the opt-in process, never the original files.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dependencies/astar_pycpp"))

from poni.runtime_neuron import add_neuron_arguments, export_options


def main():
    parser = argparse.ArgumentParser(add_help=False)
    add_neuron_arguments(parser)
    options, original_args = parser.parse_known_args()
    export_options(options)
    sys.argv = [sys.argv[0]] + original_args
    import utils.planners as planners
    from utils.planners_neuron import PlannerActorSequentialNeuron, PlannerActorVectorNeuron
    planners.PlannerActorSequential = PlannerActorSequentialNeuron
    planners.PlannerActorVector = PlannerActorVectorNeuron
    from eval_poni import main as original_main
    from poni.wavefront_neuron import warmup
    if options.neuron_mode != "reference":
        warmup()
    return original_main()


if __name__ == "__main__":
    main()
