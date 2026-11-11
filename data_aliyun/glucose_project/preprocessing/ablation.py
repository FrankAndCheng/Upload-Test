"""十模型标准化/PCA四组配对实验。实际实现：review/preprocessing_ablation/run_ablation.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('preprocessing', argv)


if __name__ == "__main__":
    raise SystemExit(main())
