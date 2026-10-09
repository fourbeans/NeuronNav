"""Independent Gibson ObjectNav entry point for Sigma-Delta PONI."""

from semexp import eval_poni as evaluator
from semexp.model_pf_sigma_d import RL_PolicySigmaD


def main():
    # Patch only this process.  The original semexp/eval_poni.py and its normal
    # RL_Policy import remain unchanged and independently runnable.
    evaluator.RL_Policy = RL_PolicySigmaD
    return evaluator.main()


if __name__ == "__main__":
    main()
