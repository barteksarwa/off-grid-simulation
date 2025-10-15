from ..utils.sim_constants import *

class Battery:
    def __init__(self):
        pass

    def bess_discharge_capacity(self, qb, bd, ub, g, g_min, t):
        pb = qb * bd
        discharge_capacity = min(pb * t, ub * (g - g_min))
        return discharge_capacity

    def bess_charge_capacity(self, qb, bc, ub, g, g_max, t):
        pb = -qb * bc
        charge_capacity = max(pb * t, -ub * (g_max - g))
        return charge_capacity

# qb - power rating
# bd, bc battery charge discharge efficiency
# ub - energy rating
# g - SoC
# t - time

