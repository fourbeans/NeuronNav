"""MP3D Sigma-D plus strict SNN wavefront with bounded visible readout."""

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
    parser.add_argument("--mp3d-refine-cap-cm", type=float, required=True)
    parser.add_argument("--mp3d-refine-cost-guard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mp3d-refine-angle-deg", type=float, default=None)
    options, original_args = parser.parse_known_args()
    if options.neuron_mode != "neuron":
        parser.error("strict SNN wavefront requires --neuron-mode neuron")
    if not 30 <= options.mp3d_refine_cap_cm <= 100:
        parser.error("cap must be in [30, 100] cm")
    if options.mp3d_refine_angle_deg is not None and not 0 <= options.mp3d_refine_angle_deg <= 180:
        parser.error("angle must be in [0, 180] degrees")
    export_options(options)
    os.environ["PONI_MP3D_REFINE_CAP_CM"] = str(options.mp3d_refine_cap_cm)
    os.environ["PONI_MP3D_REFINE_COST_GUARD"] = str(options.mp3d_refine_cost_guard)
    if options.mp3d_refine_angle_deg is not None:
        os.environ["PONI_MP3D_REFINE_ANGLE_DEG"] = str(options.mp3d_refine_angle_deg)
    sys.argv = [sys.argv[0]] + original_args

    import utils.planners as planners
    from utils.planners_mp3d_refine_neuron import (
        PlannerActorSequentialMP3DRefineNeuron,
        PlannerActorVectorMP3DRefineNeuron,
    )
    planners.PlannerActorSequential = PlannerActorSequentialMP3DRefineNeuron
    planners.PlannerActorVector = PlannerActorVectorMP3DRefineNeuron

    from policies.pf_exp_sigma_d import PFExpSigmaD
    from policies.policy_registry import policy_registry
    if policy_registry.get_policy("PFExpSigmaD") is not PFExpSigmaD:
        raise RuntimeError("PFExpSigmaD registration failed")

    import global_agent
    if (global_agent.PlannerActorSequential is not PlannerActorSequentialMP3DRefineNeuron
            or global_agent.PlannerActorVector is not PlannerActorVectorMP3DRefineNeuron):
        raise RuntimeError("GlobalAgent loaded a non-strict planner")

    from eval_poni import main as original_main
    from poni.wavefront_neuron import warmup as wave_warmup
    wave_warmup()
    print(
        "MP3D_REFINE: policy=PFExpSigmaD, planner=strict-SNN-weighted-wave, "
        f"cap_cm={options.mp3d_refine_cap_cm}, cost_guard={options.mp3d_refine_cost_guard}, "
        f"angle_deg={options.mp3d_refine_angle_deg}",
        flush=True,
    )
    return original_main()


if __name__ == "__main__":
    main()
