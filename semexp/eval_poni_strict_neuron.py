"""Run Gibson with the SNN wave planner and no reference-planner fallback."""

import argparse
import sys

from poni.runtime_neuron import add_neuron_arguments, export_options


def main():
    parser = argparse.ArgumentParser(add_help=False)
    add_neuron_arguments(parser)
    options, original_args = parser.parse_known_args()
    if options.neuron_mode != "neuron":
        parser.error("the strict entry point requires --neuron-mode neuron")
    export_options(options)
    sys.argv = [sys.argv[0]] + original_args
    import semexp.envs.habitat as environment
    from semexp.envs.habitat.sem_exp_strict_neuron import make_env_fn_strict_neuron
    from semexp.eval_poni import main as original_main
    from poni.wavefront_neuron import warmup
    warmup()
    environment.make_env_fn = make_env_fn_strict_neuron
    return original_main()


if __name__ == "__main__":
    main()
