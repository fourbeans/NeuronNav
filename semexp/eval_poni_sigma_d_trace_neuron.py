"""Independent old strict-SNN Gibson entry with pose tracing only."""

import argparse
import os
import sys

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

    import semexp.envs.habitat as environment
    from semexp.envs.habitat.sem_exp_trace_neuron import make_env_fn_trace_neuron
    from semexp import eval_poni as evaluator
    from semexp.model_pf_sigma_d import RL_PolicySigmaD
    from poni.wavefront_neuron import warmup

    environment.make_env_fn = make_env_fn_trace_neuron
    evaluator.RL_Policy = RL_PolicySigmaD
    warmup()
    print("TRACE_GIBSON: Sigma-D + old strict SNN wavefront; pose audit only", flush=True)
    return evaluator.main()


if __name__ == "__main__":
    main()
