# # Demo
#
# This notebook contains a simple demo on how to work with the code.
#

import sys
import numpy as np
from scipy.integrate import solve_ivp
import ap_features as apf

sys.path.insert(0, "../code")
from run_simulation import fitzhugh_nagumo  # noqa: E402

# +
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
# -

v_all.plot()

w_all.plot()

v = v_all.average_beat()
print(v.apd(50))
v.plot()

w = w_all.average_beat()
print(w.apd(50))
w.plot()
