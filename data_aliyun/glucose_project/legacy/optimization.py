"""早期1561条验证或仅原train的A/B协议。实际实现：paper_optimized.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('legacy-optimization', argv)


if __name__ == "__main__":
    raise SystemExit(main())
