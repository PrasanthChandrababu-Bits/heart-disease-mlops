"""Train, compare, track, and package heart-disease classifiers."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import train_all  # noqa: E402


def main() -> None:
    """Run the complete model-development workflow."""
    selected, results = train_all()
    payload = {
        "selected_model": selected.name,
        "models": {
            result.name: {
                "cross_validation_roc_auc": (
                    result.cross_validation_roc_auc
                ),
                "test_metrics": vars(result.test_metrics),
            }
            for result in results
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
