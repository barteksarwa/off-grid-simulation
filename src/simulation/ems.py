# src/simulation/ems.py
from ..utils.sim_constants import SOC_min, SOC_max
import math

def ems(p_l, p_pv, SOC, p_bess, hydrogen_SOC, previous_s, ub, n_cells=1, a_cell=0.01, p_max_fc=1000):
    p_l = -abs(p_l)
    net = p_l + p_pv
    s = previous_s
    P_sofc_base = min(300 * n_cells * a_cell / 0.01, p_max_fc)  # 300W baseload, capped at p_max_fc
    P_soec_min = 200  # 200W minimum for SOEC (unscaled to match tests)

    if net >= 0:  # Energy surplus
        if hydrogen_SOC < 1.0 and net >= P_soec_min:
            s = 2  # SOEC produces hydrogen (significant surplus)
        elif SOC < SOC_max:
            s = 1  # Charge battery (small surplus or H2 full, SOFC may contribute)
        else:
            s = 8  # Curtailment (H2 tanks and battery full)
    else:  # Energy deficit
        if hydrogen_SOC > 0:
            if abs(net) <= P_sofc_base:
                s = 4  # SOFC covers load at baseload
            elif SOC > SOC_min:  # Battery can contribute
                if min(p_bess, (SOC - SOC_min) * ub) >= abs(net) - P_sofc_base:
                    s = 5  # SOFC at baseload + battery fully covers peak
                else:
                    s = 9  # SOFC at baseload, battery partially supports
            else:
                s = 4  # SOFC at baseload (no battery support)
        else:  # No hydrogen
            if SOC > SOC_min and min(p_bess, (SOC - SOC_min) * ub) >= abs(net):
                s = 3  # Battery covers load
            else:
                s = 7  # Deficit (no hydrogen, insufficient battery)

    if SOC >= SOC_max and s not in [2, 8]:
        SOC = SOC_max
    return s