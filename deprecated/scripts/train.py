"""Train a numerical-attractor model.

Usage::

    python scripts/train.py --preset default
    python scripts/train.py --preset weber --wandb
    python scripts/train.py --preset default --epochs 200
    python scripts/train.py --preset weber --seed 73
"""

from src.training.engine import make_parser, run_training


def main():
    parser = make_parser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
