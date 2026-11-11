"""重算指标并重新加载建议模型核验。实际实现：review/preprocessing_ablation/verify_results.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('verify-preprocessing', argv)


if __name__ == "__main__":
    raise SystemExit(main())
