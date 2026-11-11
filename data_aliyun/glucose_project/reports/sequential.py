"""读取顺序实验结果并生成报告/notebook。实际实现：review/sequential_experiments/render_results.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('report-sequential', argv)


if __name__ == "__main__":
    raise SystemExit(main())
