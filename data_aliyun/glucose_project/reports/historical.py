"""早期协议的事后误差分析；会重拟合部分参考模型。实际实现：review/optimization/analyze_results.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('report-historical', argv)


if __name__ == "__main__":
    raise SystemExit(main())
