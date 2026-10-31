from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

HORIZONS = (24, 48, 72)
# Walk-forward test windows (same as the DE pipeline's FOLDS); train = everything whose target ends before the window.
FOLDS = (("2025-04-01", "2025-06-15"), ("2025-06-15", "2025-08-15"), ("2025-08-15", "2025-11-01"))
QUANTILES = (0.1, 0.9)
SATURATION = 99.9
