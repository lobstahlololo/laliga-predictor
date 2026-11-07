"""Cross-platform command entrypoint for the La Liga predictor.

Examples:
    python main.py data
    python main.py features
    python main.py train
    python main.py predict
    python main.py backfill
    python main.py reports
    python main.py experiments
    python main.py phase3
    python main.py update
    python main.py registry
    python main.py api
    python main.py dashboard
    python main.py test
    python main.py all
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_module(module_name: str, function: str = "main") -> int:
    module = importlib.import_module(module_name)
    runner = getattr(module, function)
    # Runners that accept argv must not receive the stage name (for example
    # ``phase3``) as their own CLI argument. Functions without parameters are
    # called normally.
    if inspect.signature(runner).parameters:
        return int(runner([]))
    return int(runner())


def _predict() -> int:
    import predict_upcoming

    return predict_upcoming.main(["predict_upcoming.py"])


def _api() -> int:
    """Serve the FastAPI app with uvicorn using environment configuration."""
    import uvicorn

    host = os.environ.get("LALIGA_API_HOST", "127.0.0.1")
    port = int(os.environ.get("LALIGA_API_PORT", "8000"))
    uvicorn.run("src.api.main:app", host=host, port=port)
    return 0


STAGES = {
    "data": lambda: _run_module("src.data.make_dataset"),
    "features": lambda: _run_module("src.features.build_features"),
    "train": lambda: _run_module("src.models.train_model"),
    "predict": _predict,
    "backfill": lambda: _run_module("backfill_results"),
    "reports": lambda: _run_module("src.predictions.reports"),
    "experiments": lambda: _run_module("src.experiments.runner"),
    "phase3": lambda: _run_module("src.experiments.phase3_runner"),
    "update": lambda: _run_module("src.pipeline.automation"),
    "registry": lambda: _run_module("src.models.registry"),
    "api": _api,
    "test": lambda: subprocess.call([sys.executable, "-m", "pytest"]),
    "dashboard": lambda: subprocess.call(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
    ),
}

ORDER = ["data", "features", "train", "predict", "backfill", "reports"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="La Liga predictor entrypoint")
    parser.add_argument(
        "stage",
        choices=sorted(STAGES) + ["all"],
        help="Pipeline stage to run",
    )
    args = parser.parse_args(argv)

    if args.stage == "all":
        for stage in ORDER:
            print(f"\n>>> {stage}")
            code = STAGES[stage]()
            if code != 0 and stage != "predict":
                print(f"Stage {stage} failed with exit code {code}.")
                return code
        return 0

    return STAGES[args.stage]()


if __name__ == "__main__":
    sys.exit(main())
