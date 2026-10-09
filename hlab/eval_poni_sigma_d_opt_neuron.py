"""MP3D Sigma-D + strict SNN wavefront, with one selectable waypoint readout."""

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dependencies/astar_pycpp"))

from poni.runtime_neuron import add_neuron_arguments, export_options


def main():
    parser = argparse.ArgumentParser(add_help=False)
    add_neuron_arguments(parser)
    parser.add_argument("--waveopt-readout", choices=("length", "visible"), required=True)
    parser.add_argument("--waveopt-lookahead-cm", type=float, default=25.0)
    options, original_args = parser.parse_known_args()
    if options.neuron_mode != "neuron":
        parser.error("strict SNN wavefront requires --neuron-mode neuron")
    if not 0 < options.waveopt_lookahead_cm <= 100:
        parser.error("lookahead must be in (0,100] cm")
    export_options(options)
    os.environ["PONI_WAVEOPT_READOUT"] = options.waveopt_readout
    os.environ["PONI_WAVEOPT_LOOKAHEAD_CM"] = str(options.waveopt_lookahead_cm)
    sys.argv = [sys.argv[0]] + original_args

    import utils.planners as planners
    from utils.planners_opt_neuron import (
        PlannerActorSequentialOptNeuron,
        PlannerActorVectorOptNeuron,
    )
    planners.PlannerActorSequential = PlannerActorSequentialOptNeuron
    planners.PlannerActorVector = PlannerActorVectorOptNeuron

    from policies.pf_exp_sigma_d import PFExpSigmaD
    from policies.policy_registry import policy_registry
    if policy_registry.get_policy("PFExpSigmaD") is not PFExpSigmaD:
        raise RuntimeError("PFExpSigmaD registration failed")

    import global_agent
    if (global_agent.PlannerActorSequential is not PlannerActorSequentialOptNeuron
            or global_agent.PlannerActorVector is not PlannerActorVectorOptNeuron):
        raise RuntimeError("GlobalAgent loaded a non-strict planner")

    from eval_poni import main as original_main
    from poni.wavefront_neuron import warmup as wave_warmup
    from poni.waypoint_opt_neuron import warmup as readout_warmup

    wave_warmup()
    readout_warmup()
    print(
        "WAVEOPT_MP3D: policy=PFExpSigmaD, planner=strict-SNN-weighted-wave, "
        f"readout={options.waveopt_readout}, lookahead_cm={options.waveopt_lookahead_cm}",
        flush=True,
    )
    return original_main()


if __name__ == "__main__":
    main()
