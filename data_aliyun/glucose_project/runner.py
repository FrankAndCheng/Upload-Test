"""Run historical code in an isolated process without changing its file path."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .catalog import PROJECT_ROOT, WORKFLOWS


def entrypoint(name, argv=None):
    workflow = WORKFLOWS[name]
    parser = argparse.ArgumentParser(
        prog=f"python -m glucose_project.{workflow.module}",
        description=workflow.description,
        epilog=f"实际源码：{workflow.source}\n输出：{workflow.artifacts}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="只显示命令和输出位置，不执行源码")
    if workflow.arguments == "predict":
        parser.add_argument("--input", type=Path, help="原格式特征CSV")
        parser.add_argument("--output", type=Path, help="预测输出CSV")
        parser.add_argument("--scope", choices=["full", "development"], default="full")
    elif workflow.arguments == "legacy":
        parser.add_argument("--protocol", choices=["same", "external", "all"], default="all")
    args = parser.parse_args(argv)
    forwarded = []
    if workflow.arguments == "predict":
        if not args.dry_run and (args.input is None or args.output is None):
            parser.error("预测需要同时指定 --input 和 --output")
        for flag in ["input", "output"]:
            value = getattr(args, flag)
            if value is not None:
                # Resolve in the caller's directory before the child switches
                # to the project root. Paths with spaces stay separate args.
                forwarded.extend([f"--{flag}", str(value.expanduser().resolve())])
        forwarded.extend(["--scope", args.scope])
    elif workflow.arguments == "legacy":
        forwarded.extend(["--protocol", args.protocol])
    source = PROJECT_ROOT / workflow.source
    if not source.is_file():
        parser.error(f"找不到实际源码：{source}")
    command = [sys.executable, str(source), *forwarded]
    if args.dry_run:
        print(json.dumps({
            "workflow": name, "module": f"glucose_project.{workflow.module}",
            "cwd": str(PROJECT_ROOT), "command": command,
            "artifacts": workflow.artifacts,
        }, ensure_ascii=False, indent=2))
        return 0
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
