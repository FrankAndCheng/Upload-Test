"""按既定权重网格补充小网络融合。实际实现：review/sequential_experiments/complete_nn_blends.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('neural-blend', argv)


if __name__ == "__main__":
    raise SystemExit(main())
