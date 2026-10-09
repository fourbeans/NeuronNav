"""Independent MP3D HLab entry point for Sigma-Delta PONI."""

# Import registers GLOBAL_AGENT.name="PFExpSigmaD" without changing the
# original policies/__init__.py registration path.
from policies.pf_exp_sigma_d import PFExpSigmaD  # noqa: F401
from eval_poni import main


if __name__ == "__main__":
    main()
