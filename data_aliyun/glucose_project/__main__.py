"""List categorized workflows or dispatch one by its short name."""
import argparse

from .catalog import WORKFLOWS
from .runner import entrypoint


def main(argv=None):
    parser = argparse.ArgumentParser(description="血糖预测：按功能分类的运行入口")
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("list", help="列出入口、实际源码和功能")
    run = sub.add_parser("run", help="运行一个入口；入口参数写在名称后")
    run.add_argument("workflow", choices=WORKFLOWS)
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "run":
        return entrypoint(args.workflow, args.arguments)
    if args.action == "list":
        for name, item in WORKFLOWS.items():
            print(f"{name}\n  glucose_project.{item.module}\n  {item.description}\n  {item.source}")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
