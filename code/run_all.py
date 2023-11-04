"""
This could for example be a the script for running a simulation

This could for example take input from the `processed_data` folder
and output results in the `results` folder.
"""
from pathlib import Path

from run_simulation import main

here = Path(__file__).absolute().parent
result_directory = here / "results"

parameter_sets = [
    {"a": -0.2, "b": 1.1},
    {"a": -0.3, "b": 1.1},
    {"a": -0.2, "b": 1.15},
    {"a": -0.3, "b": 1.15},
    {"a": -0.2, "b": 1.2},
    {"a": -0.3, "b": 1.2},
]

if __name__ == "__main__":
    for p in parameter_sets:
        # This would by the equivalent list of arguments passe from the command line
        args = ["-o", result_directory.as_posix(), "-a", str(p["a"]), "-b", str(p["b"])]
        main(args)
