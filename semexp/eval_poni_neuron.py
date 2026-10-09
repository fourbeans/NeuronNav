"""Run unchanged Gibson evaluator with an importable neuron environment factory.

All original flags are forwarded. --neuron-mode reference is the instrumented
original baseline; guarded is the default. Logs must use a fresh directory.
"""
import argparse
import sys

from poni.runtime_neuron import add_neuron_arguments, export_options


def main():
    parser = argparse.ArgumentParser(add_help=False)
    add_neuron_arguments(parser)
    options, original_args = parser.parse_known_args()
    export_options(options)
    sys.argv = [sys.argv[0]] + original_args
    import semexp.envs.habitat as environment
    from semexp.envs.habitat.sem_exp_neuron import make_env_fn_neuron
    from semexp.eval_poni import main as original_main
    from poni.wavefront_neuron import warmup
    if options.neuron_mode != "reference":
        warmup()
    # Only changes a symbol in this opt-in process; original source stays intact.
    environment.make_env_fn = make_env_fn_neuron
    return original_main()


if __name__ == "__main__":
    main()
