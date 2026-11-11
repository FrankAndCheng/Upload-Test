"""按顺序执行特征消融、联合搜索、树融合与小网络实验。实际实现：review/sequential_experiments/run_experiments.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('sequential', argv)


if __name__ == "__main__":
    raise SystemExit(main())
