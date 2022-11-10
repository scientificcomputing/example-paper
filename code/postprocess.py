"""
Script for recreating figures and tables in the paper

This should take files in the `results` folder as input
and output figures in a folder. It could also
print tables to the console. The important part here is
that you should not have to re-run any simulations to
recreate the figures and tables.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import ap_features as apf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pre_processing import datapath
from run_all import result_directory


@dataclass
class Result:
    y: np.ndarray
    time: np.ndarray
    a: float
    b: float
    end_time: float
    timestamp: datetime

    def __post_init__(self):
        self._v = apf.Beats(y=self.v, t=self.time)
        self._w = apf.Beats(y=self.w, t=self.time)

    @property
    def v(self) -> np.ndarray:
        return self.y[0, :]

    @property
    def w(self) -> np.ndarray:
        return self.y[1, :]

    @cached_property
    def v_mean(self) -> apf.Beat:
        return self._v.average_beat()

    @cached_property
    def w_mean(self) -> apf.Beat:
        return self._w.average_beat()

    @property
    def apd50_v(self) -> float:
        return np.median([beat.apd(50) for beat in self._v.beats])

    @property
    def beatrate_v(self) -> float:
        return self._v.beat_rate

    @property
    def apd50_w(self) -> float:
        return np.median([beat.apd(50) for beat in self._w.beats])

    @property
    def beatrate_w(self) -> float:
        return self._w.beat_rate


def load_result(path: Path) -> Result:
    """Load result from json file into a Result object"""
    data = json.loads(path.read_text())
    data["y"] = np.array(data["y"])
    data["time"] = np.array(data["time"])
    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
    data.pop("output")
    return Result(**data)


def load_results() -> list[Result]:
    """Load all results from the result directory"""
    results: list[Result] = []
    for path in result_directory.iterdir():
        if not path.suffix == ".json":
            continue
        results.append(load_result(path))
    return results


def load_data():
    return json.loads(datapath.read_text())


def align_at_peak(
    v: np.ndarray,
    w: np.ndarray,
    t_v: np.ndarray,
    t_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v_idxmax = np.argmax(v)
    w_idxmax = np.argmax(w)

    new_t_v = t_v - t_v[v_idxmax]
    new_t_w = t_w - t_w[w_idxmax]
    return new_t_v, new_t_w


def figure1(results: list[Result], data: dict[str, list[float]], fname: str) -> None:
    """Reproducing Figure 1 in the paper"""
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    lines = []
    labels = []

    for result in results:
        # Align beats at peak
        t_v, t_w = align_at_peak(
            result.v_mean.y,
            result.w_mean.y,
            result.v_mean.t,
            result.w_mean.t,
        )

        (l,) = ax[0].plot(t_v, result.v_mean.y)
        ax[1].plot(t_w, result.w_mean.y)

        lines.append(l)
        labels.append(f"a={result.a:.2f}, a={result.b:.2f}")

    t_v, t_w = align_at_peak(
        data["v"],
        data["w"],
        np.array(data["t_v"]),
        np.array(data["t_w"]),
    )
    (l,) = ax[0].plot(t_v, data["v"], color="k")
    ax[1].plot(t_w, data["w"], color="k")
    lines.append(l)
    labels.append("data")

    ax[0].set_ylabel("v")
    ax[1].set_ylabel("w")
    ax[1].set_xlabel("Time [ms]")
    for axi in ax:
        axi.grid()
    fig.subplots_adjust(right=0.85)
    fig.legend(
        lines,
        labels,
        bbox_to_anchor=(1.0, 0.5),
        loc="center right",
        title="Parameters",
    )
    fig.savefig(fname, bbox_inches="tight", dpi=500)


def table1(results: list[Result]):
    """Reproduce table 1 in the paper by printing the Latex table"""
    data = []
    for result in results:
        data.append(
            {
                "a": result.a,
                "b": result.b,
                "APD50 (V)": result.apd50_v,
                "Beatrate (V)": result.beatrate_v,
                "APD50 (W)": result.apd50_w,
                "Beatrate (W)": result.beatrate_w,
            },
        )
    df = pd.DataFrame(data)
    print(df.style.to_latex())


def main():
    figdir = Path("figures")
    figdir.mkdir(exist_ok=True)
    results = load_results()
    data = load_data()

    figure1(results, data, figdir / "figure1.png")
    table1(results)


if __name__ == "__main__":
    main()
