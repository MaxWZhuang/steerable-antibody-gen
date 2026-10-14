"""CLI for explicit low-resource trastuzumab training and benchmark exports."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smallAntibodyGen.experiments.trastuzumab_poc.cli import main


if __name__ == "__main__":
    main()
