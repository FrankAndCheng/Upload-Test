"""Routing checks only: no training, report overwrites or B-label access."""
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from glucose_project.catalog import PROJECT_ROOT, WORKFLOWS
from glucose_project.runner import entrypoint


class WorkflowEntrypoints(unittest.TestCase):
    def test_all_help_and_dry_runs_are_nonexecuting(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name), patch("glucose_project.runner.subprocess.run") as execute:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as stopped:
                        entrypoint(name, ["--help"])
                self.assertEqual(stopped.exception.code, 0)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(entrypoint(name, ["--dry-run"]), 0)
                plan = json.loads(output.getvalue())
                self.assertEqual(plan["cwd"], str(PROJECT_ROOT))
                self.assertEqual(plan["command"][0], sys.executable)
                self.assertEqual(Path(plan["command"][1]), PROJECT_ROOT / WORKFLOWS[name].source)
                execute.assert_not_called()

    def test_importing_categories_does_not_import_training_libraries(self):
        program = """
import importlib, sys
from glucose_project.catalog import WORKFLOWS
for item in WORKFLOWS.values():
    importlib.import_module('glucose_project.' + item.module)
assert not {'tensorflow', 'pandas', 'sklearn', 'xgboost', 'lightgbm'} & set(sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", program], cwd=PROJECT_ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_predict_paths_keep_the_callers_directory_and_spaces(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="glucose routing ") as folder:
            try:
                os.chdir(folder)
                with patch("glucose_project.runner.subprocess.run") as execute:
                    execute.return_value.returncode = 0
                    self.assertEqual(entrypoint("predict", ["--input", "input file.csv",
                                     "--output", "result file.csv", "--scope", "development"]), 0)
                    argv = execute.call_args.args[0]
                    self.assertEqual(argv[argv.index("--input") + 1], str((Path(folder) / "input file.csv").resolve()))
                    self.assertEqual(argv[argv.index("--output") + 1], str((Path(folder) / "result file.csv").resolve()))
                    self.assertEqual(execute.call_args.kwargs["cwd"], PROJECT_ROOT)
            finally:
                os.chdir(old)

    def test_missing_prediction_arguments_cannot_launch_training_or_prediction(self):
        with patch("glucose_project.runner.subprocess.run") as execute, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                entrypoint("predict", [])
            self.assertEqual(stopped.exception.code, 2)
            execute.assert_not_called()

    def test_child_failure_is_returned_to_the_caller(self):
        with patch("glucose_project.runner.subprocess.run") as execute:
            execute.return_value.returncode = 7
            self.assertEqual(entrypoint("preprocessing", []), 7)

    def test_historical_sources_match_the_recorded_organization_snapshot(self):
        catalog = json.loads((PROJECT_ROOT / "glucose_project/source_catalog.json").read_text())
        self.assertEqual(set(catalog["sources"]), {w.source for w in WORKFLOWS.values()})
        for source, record in catalog["sources"].items():
            self.assertEqual(hashlib.sha256((PROJECT_ROOT / source).read_bytes()).hexdigest(), record["sha256"], source)


if __name__ == "__main__":
    unittest.main()
