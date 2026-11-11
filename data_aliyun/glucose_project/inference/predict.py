"""加载已冻结四模型，只预测不训练。实际实现：review/sequential_experiments/predict_selected.py。"""
from ..runner import entrypoint


def main(argv=None):
    return entrypoint('predict', argv)


if __name__ == "__main__":
    raise SystemExit(main())
