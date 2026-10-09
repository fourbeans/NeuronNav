"""Offline paired trajectory/score plots, no model or simulator mutation."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--traces",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(args.traces);out=Path(args.output)
    if out.exists():raise FileExistsError(str(out))
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for column,(actor,episode) in enumerate(((5,1),(6,1),(7,1),(16,4))):
        traces={}
        for name,color in (("baseline","#2166ac"),("dend","#b2182b")):
            path=root/("{}_actor_{:03d}".format(name,actor))/"trace_{:04d}.npz".format(episode)
            if not path.exists():continue
            with np.load(path,allow_pickle=False) as archive:
                d={k:archive[k] for k in archive.files}
            traces[name]=d
            xy=d["agent_xy"]
            axes[0,column].plot(xy[:,0],xy[:,1],color=color,label="{} ({} steps)".format(name,len(xy)),linewidth=1.2)
            axes[0,column].scatter(xy[-1,0],xy[-1,1],color=color,s=30,marker="x")
        axes[0,column].set_title("Actor {} / episode {}".format(actor,episode))
        axes[0,column].set_xlabel("Map x (m)");axes[0,column].set_ylabel("Map y (m)")
        axes[0,column].axis("equal");axes[0,column].legend(fontsize=8)
        if "dend" in traces:
            d=traces["dend"];t=np.flatnonzero(d["action"]>=0);a=d["action"][t].astype(int)
            axes[1,column].plot(t,d["spatial"][t,0]-d["spatial"][t,a],label="Spatial score sacrificed",linewidth=1)
            axes[1,column].plot(t,d["residual"][t,a]-d["residual"][t,0],label="Residual advantage",linewidth=1)
            for collision in np.flatnonzero(d["collision"]):
                axes[1,column].axvline(collision,color="gray",alpha=.15,linewidth=.5)
            axes[1,column].set_title("Dendritic selections; gray = collision")
            axes[1,column].set_xlabel("Real step");axes[1,column].legend(fontsize=7)
    fig.suptitle("Exact deterministic replay of lost-success cases (no model changes)")
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out,dpi=140);plt.close(fig)


if __name__=="__main__":main()
