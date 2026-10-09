"""Independent old strict-SNN MP3D entry with episode/pose audit only."""

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
    options, original_args = parser.parse_known_args()
    if options.neuron_mode != "neuron":
        parser.error("strict SNN wavefront requires --neuron-mode neuron")
    export_options(options)
    os.environ["PONI_WAVEOPT_TRACE"] = "1"
    sys.argv = [sys.argv[0]] + original_args

    import utils.planners as planners
    from utils.planners_trace_neuron import (
        PlannerActorSequentialTraceNeuron,
        PlannerActorVectorTraceNeuron,
    )
    planners.PlannerActorSequential = PlannerActorSequentialTraceNeuron
    planners.PlannerActorVector = PlannerActorVectorTraceNeuron

    from policies.pf_exp_sigma_d import PFExpSigmaD
    from policies.policy_registry import policy_registry
    if policy_registry.get_policy("PFExpSigmaD") is not PFExpSigmaD:
        raise RuntimeError("PFExpSigmaD registration failed")

    import global_agent
    if (global_agent.PlannerActorSequential is not PlannerActorSequentialTraceNeuron
            or global_agent.PlannerActorVector is not PlannerActorVectorTraceNeuron):
        raise RuntimeError("GlobalAgent loaded a non-strict planner")

    from eval_poni import main as original_main
    from poni.wavefront_neuron import warmup
    warmup()
    print("TRACE_MP3D: Sigma-D + old strict SNN wavefront; pose audit only", flush=True)
    return original_main()


if __name__ == "__main__":
    main()
