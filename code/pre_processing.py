"""
Script for performing pre-processing of the data.

This could for example be scripts for generating meshes,
transforming data to the correct format, cleaning data, etc.
Typically this could take input from `../data/raw` and output
some data to `processed_data`
"""
import json
from pathlib import Path

import ap_features as apf
import numpy as np
from run_simulation import fitzhugh_nagumo
from scipy.integrate import solve_ivp


here = Path(__file__).absolute().parent
datadir = here / ".." / "data"
datapath = datadir / "data.json"


def generate_syntetic_data():
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


if __name__ == "__main__":
    generate_syntetic_data()
