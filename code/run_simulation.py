"""
This could for example be the main script for running a simulation

Here we have implemented a argument parser that than take arguments from
the command line, i.e
python main -T
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp


def fitzhugh_nagumo(
    t: float,
    x: np.ndarray,
    a: float,
    b: float,
    tau: float = 20.0,
    Iext: float = 0.23,
) -> np.ndarray:
    """Time derivative of the Fitzhugh-Nagumo neural model.

    Parameters
    ----------
    t : float
        Time (not used)
    x : np.ndarray
        State of size 2 - (Membrane potential, Recovery variable)
    a : float
        Parameter in the model
    b : float
        Parameter in the model
    tau : float
        Time scale
    Iext : float
        Constant stimulus current
    Returns
    -------
    np.ndarray
        dx/dt - size 2
    """
    return np.array([x[0] - x[0] ** 3 - x[1] + Iext, (x[0] - a - b * x[1]) / tau])


def main(argv: Sequence[str] | None = None) -> int:
    desc = "Solve Fitzhugh-Nagumo neural model and output the results to file"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "-T",
        "--end-time",
        dest="end_time",
        help="End time of simulation",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "-a",
        dest="a",
        help="Value of parameter a",
        type=float,
        required=True,
    )
    parser.add_argument(
        "-b",
        dest="b",
        help="Value of parameter a",
        type=float,
        required=True,
    )
    args = vars(parser.parse_args(argv))

    time = np.arange(0, args["end_time"], 1.0)
    res = solve_ivp(
        fitzhugh_nagumo,
        [0, args["end_time"]],
        [0, 0],
        args=(args["a"], args["b"]),
        t_eval=time,
    )

    outdir = Path(args["output"])
    outdir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.datetime.now().isoformat()
    # Just make some unique signature of the input
    signature = hashlib.sha256(repr(args).encode()).hexdigest()

    fname = f"results_{signature}.json"
    data = {"y": res.y.tolist(), "timestamp": timestamp, "time": time.tolist(), **args}

    path = outdir / fname
    path.write_text(json.dumps(data))
    print(f"Saved results to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
