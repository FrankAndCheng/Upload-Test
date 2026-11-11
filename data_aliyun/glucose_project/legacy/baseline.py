"""早期独立Ridge基线检查，不属于当前融合选型。实际实现：review/baseline_check.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('legacy-baseline', argv)


if __name__ == "__main__":
    raise SystemExit(main())
