"""MP3D Sigma-D plus strict weighted SNN wave, turn-to-forward-only readout."""

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
    from utils.planners_mp3d_forward_gain_neuron import (
        PlannerActorSequentialMP3DForwardGainNeuron,
        PlannerActorVectorMP3DForwardGainNeuron,
    )
    planners.PlannerActorSequential = PlannerActorSequentialMP3DForwardGainNeuron
    planners.PlannerActorVector = PlannerActorVectorMP3DForwardGainNeuron

    from policies.pf_exp_sigma_d import PFExpSigmaD
    from policies.policy_registry import policy_registry
    if policy_registry.get_policy("PFExpSigmaD") is not PFExpSigmaD:
        raise RuntimeError("PFExpSigmaD registration failed")

    import global_agent
    if (global_agent.PlannerActorSequential is not PlannerActorSequentialMP3DForwardGainNeuron
            or global_agent.PlannerActorVector is not PlannerActorVectorMP3DForwardGainNeuron):
        raise RuntimeError("GlobalAgent loaded an unexpected planner")

    from eval_poni import main as original_main
    from poni.wavefront_neuron import warmup as wave_warmup
    wave_warmup()
    print("MP3D_FORWARD_GAIN: Sigma-D + strict weighted SNN wave, "
          "turn-to-forward-only equal-cost path readout", flush=True)
    return original_main()


if __name__ == "__main__":
    main()
