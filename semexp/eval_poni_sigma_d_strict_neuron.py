"""Gibson ObjectNav: incremental Sigma-D PONI with strict SNN wave planning."""

import argparse
import sys

from poni.runtime_neuron import add_neuron_arguments, export_options


def main():
    parser = argparse.ArgumentParser(add_help=False)
    add_neuron_arguments(parser)
    options, original_args = parser.parse_known_args()
    if options.neuron_mode != "neuron":
        parser.error("the joint strict entry point requires --neuron-mode neuron")
    export_options(options)
    sys.argv = [sys.argv[0]] + original_args

    import semexp.envs.habitat as environment
    from semexp.envs.habitat.sem_exp_strict_neuron import make_env_fn_strict_neuron
    from semexp import eval_poni as evaluator
    from semexp.model_pf_sigma_d import RL_PolicySigmaD
    from poni.wavefront_neuron import warmup

    environment.make_env_fn = make_env_fn_strict_neuron
    evaluator.RL_Policy = RL_PolicySigmaD
    warmup()
    print("JOINT_GIBSON: policy=Sigma-D, planner=strict-SNN-wave+goal-projection", flush=True)
    return evaluator.main()


if __name__ == "__main__":
    main()
