"""Old strict Gibson SNN with pose trace, without changing its planner."""

import os

from habitat import make_dataset

from semexp.envs.habitat.sem_exp_opt_neuron import SemExpEnvAgentOptNeuron


def make_env_fn_trace_neuron(args, config_env, rank):
    selected_scene = os.environ.get("PONI_NEURON_SCENE", "")
    if selected_scene:
        if args.num_processes != 1:
            raise ValueError("Explicit scene selection requires --num_processes 1")
        config_env.defrost()
        config_env.DATASET.CONTENT_SCENES = [
            selected_scene if selected_scene.endswith(".glb") else selected_scene + ".glb"
        ]
        config_env.freeze()
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()
    if args.agent != "sem_exp":
        raise ValueError("trace supports the sem_exp agent only")
    # The opt subclass adds only the pose event.  Unlike its opt factory, this
    # factory never replaces strict_environment.gibson_plan_strict.
    env = SemExpEnvAgentOptNeuron(args=args, rank=rank,
                                  config_env=config_env, dataset=dataset)
    env.seed(rank + args.seed)
    return env
