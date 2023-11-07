"""
Script for performing pre-processing of the data.

This could for example be scripts for generating meshes,
transforming data to the correct format, cleaning data, etc.
Typically this could take input from `../data/raw` and output
some data to `processed_data`
"""
import json
from pathlib import Path
import argparse
from typing import Sequence

import ap_features as apf
import numpy as np
from run_simulation import fitzhugh_nagumo
from scipy.integrate import solve_ivp


here = Path(__file__).absolute().parent


def generate_syntetic_data(datapath: Path) -> None:
    a = -0.22
    b = 1.17
    time = np.arange(0, 1000.0, 1.0)

    res = solve_ivp(
        fitzhugh_nagumo,
        [0, 1000.0],
        [0, 0],
        args=(a, b),
        t_eval=time,
    )

    v_all = apf.Beats(y=res.y[0, :], t=time)
    w_all = apf.Beats(y=res.y[1, :], t=time)

    v = v_all.average_beat()
    w = w_all.average_beat()
    datapath.parent.mkdir(exist_ok=True, parents=True)
    datapath.write_text(
        json.dumps(
            {
                "t_v": v.t.tolist(),
                "v": v.y.tolist(),
                "t_w": w.t.tolist(),
                "w": w.y.tolist(),
            },
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--datapath",
        type=Path,
        default=here / ".." / "data" / "data.json",
        help="Directory where to dump the data",
    )
    generate_syntetic_data(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
