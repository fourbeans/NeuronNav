"""Run the unmodified collector with additional episode traces, no policy changes.

All CLI options are those of collect_dend. Outputs must be a fresh directory.
"""
import json
import numpy as np
from dendritic_nav.adapters import bridge_dend


class TraceBridge(bridge_dend.EpisodeBridge):
    def finish_episode(self, transition):
        trace = {
            key: np.asarray([record[key] for record in self.records])
            for key in self.records[0]
        }
        stem = self.directory / "trace_{:04d}".format(self.episodes + 1)
        with open(str(stem) + ".npz", "xb") as stream:
            np.savez_compressed(stream, **trace)
        with open(str(stem) + ".json", "x") as stream:
            json.dump(transition, stream, indent=2, default=lambda x:x.item())
        super().finish_episode(transition)


def main():
    from dendritic_nav.runners import collect_dend
    original = bridge_dend.EpisodeBridge
    original_new_run = collect_dend.new_run
    bridge_dend.EpisodeBridge = TraceBridge
    collect_dend.new_run = lambda directory,args,kind: original_new_run(
        directory,args,"dendritic_nav.diagnostics.trace_dend")
    try:
        collect_dend.main()
    finally:
        bridge_dend.EpisodeBridge = original
        collect_dend.new_run = original_new_run


if __name__ == "__main__":
    main()
