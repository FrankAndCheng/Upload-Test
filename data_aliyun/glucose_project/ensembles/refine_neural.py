"""训练OOF驱动的去乙肝小网络补充对照。实际实现：review/sequential_experiments/refine_nn_features.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('neural-features', argv)


if __name__ == "__main__":
    raise SystemExit(main())
