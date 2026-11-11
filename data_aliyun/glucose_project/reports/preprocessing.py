"""读取预处理结果并生成报告/notebook。实际实现：review/preprocessing_ablation/render_results.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('report-preprocessing', argv)


if __name__ == "__main__":
    raise SystemExit(main())
