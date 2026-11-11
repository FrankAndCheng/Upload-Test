"""恢复原六模型、重拟合并评估B。实际实现：review/external_b_comparison/evaluate_frozen.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('original-algorithms', argv)


if __name__ == "__main__":
    raise SystemExit(main())
