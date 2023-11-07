"""
This could for example be a the script for running a simulation

This could for example take input from the `processed_data` folder
and output results in the `results` folder.
"""
from pathlib import Path
from typing import Sequence
import argparse

import run_simulation

here = Path(__file__).absolute().parent


parameter_sets = [
    {"a": -0.2, "b": 1.1},
    {"a": -0.3, "b": 1.1},
    {"a": -0.2, "b": 1.15},
    {"a": -0.3, "b": 1.15},
    {"a": -0.2, "b": 1.2},
    {"a": -0.3, "b": 1.2},
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-r",
        "--resultdir",
        type=Path,
        default=here / "results",
        help="Directory where the results are stored",
    )
    args = vars(parser.parse_args(argv))
    result_directory: Path = args["resultdir"]

    for p in parameter_sets:
        # This would by the equivalent list of
        # arguments passe from the command line
        run_simulation.main(
            [
                "-o",
                str(result_directory.as_posix()),
                "-a",
                str(p["a"]),
                "-b",
                str(p["b"]),
            ],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
