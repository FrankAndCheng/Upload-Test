"""检查base导入与小样本训练。实际实现：review/check_environment.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('environment', argv)


if __name__ == "__main__":
    raise SystemExit(main())
