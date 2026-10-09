"""Gibson Sigma-D + strict SNN wavefront, with one selectable waypoint readout."""

import argparse
import os
import sys

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

    import semexp.envs.habitat as environment
    from semexp.envs.habitat.sem_exp_opt_neuron import make_env_fn_opt_neuron
    from semexp import eval_poni as evaluator
    from semexp.model_pf_sigma_d import RL_PolicySigmaD
    from poni.wavefront_neuron import warmup as wave_warmup
    from poni.waypoint_opt_neuron import warmup as readout_warmup

    environment.make_env_fn = make_env_fn_opt_neuron
    evaluator.RL_Policy = RL_PolicySigmaD
    wave_warmup()
    readout_warmup()
    print(
        "WAVEOPT_GIBSON: policy=Sigma-D, planner=strict-SNN-wave, "
        f"readout={options.waveopt_readout}, lookahead_cm={options.waveopt_lookahead_cm}",
        flush=True,
    )
    return evaluator.main()


if __name__ == "__main__":
    main()
