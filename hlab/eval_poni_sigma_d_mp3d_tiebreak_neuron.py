"""MP3D Sigma-D plus strict SNN wavefront with cost-preserving tie-break."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dependencies/astar_pycpp"))

from poni.runtime_neuron import add_neuron_arguments, export_options


def main():
    parser = argparse.ArgumentParser(add_help=False)
    add_neuron_arguments(parser)
    options, original_args = parser.parse_known_args()
    if options.neuron_mode != "neuron":
        parser.error("strict SNN wavefront requires --neuron-mode neuron")
    export_options(options)
    sys.argv = [sys.argv[0]] + original_args

    import utils.planners as planners
    from utils.planners_mp3d_tiebreak_neuron import (
        PlannerActorSequentialMP3DTiebreakNeuron,
        PlannerActorVectorMP3DTiebreakNeuron,
    )
    planners.PlannerActorSequential = PlannerActorSequentialMP3DTiebreakNeuron
    planners.PlannerActorVector = PlannerActorVectorMP3DTiebreakNeuron

    from policies.pf_exp_sigma_d import PFExpSigmaD
    from policies.policy_registry import policy_registry
    if policy_registry.get_policy("PFExpSigmaD") is not PFExpSigmaD:
        raise RuntimeError("PFExpSigmaD registration failed")

    import global_agent
    if (global_agent.PlannerActorSequential is not PlannerActorSequentialMP3DTiebreakNeuron
            or global_agent.PlannerActorVector is not PlannerActorVectorMP3DTiebreakNeuron):
        raise RuntimeError("GlobalAgent loaded a non-strict planner")

    from eval_poni import main as original_main
    from poni.wavefront_neuron import warmup as wave_warmup
    wave_warmup()
    print("MP3D_TIEBREAK: PFExpSigmaD + strict weighted SNN wave, "
          "equal-arrival straighter predecessor path", flush=True)
    return original_main()


if __name__ == "__main__":
    main()
