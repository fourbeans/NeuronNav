"""Install the selected SNN-wave readout inside each Gibson forkserver worker."""

from functools import partial
import os

from habitat import make_dataset

from poni.planner_adapter_opt_neuron import gibson_plan_opt
import semexp.envs.habitat.sem_exp_strict_neuron as strict_environment


class SemExpEnvAgentOptNeuron(strict_environment.SemExpEnvAgentStrictNeuron):
    def _plan(self, planner_inputs):
        action = super()._plan(planner_inputs)
        if os.environ.get("PONI_WAVEOPT_TRACE") == "1":
            self._audit().write({
                "type": "pose",
                "seq": self._audit().seq,
                "episode_no": int(self.episode_no),
                "scene": str(self.scene_name),
                "step": int(self.timestep),
                "pose_pred": list(map(float, planner_inputs["pose_pred"])),
            })
        return action


def make_env_fn_opt_neuron(args, config_env, rank):
    strict_environment.gibson_plan_strict = partial(
        gibson_plan_opt,
        mode=os.environ["PONI_WAVEOPT_READOUT"],
        lookahead_cm=float(os.environ["PONI_WAVEOPT_LOOKAHEAD_CM"]),
        map_resolution_cm=float(args.map_resolution),
    )
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()
    if args.agent != "sem_exp":
        raise ValueError("waveopt supports the sem_exp agent only")
    env = SemExpEnvAgentOptNeuron(args=args, rank=rank, config_env=config_env, dataset=dataset)
    env.seed(rank + args.seed)
    return env
