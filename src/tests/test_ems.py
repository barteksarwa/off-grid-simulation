# src/tests/test_ems.py
import unittest
import numpy as np
from src.simulation.ems import ems
from src.utils.sim_constants import SOC_min, SOC_max, ub_default

class TestEMS(unittest.TestCase):
    def setUp(self):
        self.P_sofc_base = 300  # 300W baseload for n_cells=1, a_cell=0.01
        self.P_soec_min = 200   # 200W minimum for SOEC (unscaled)
        self.n_cells = 27       # Example from designs
        self.a_cell = 0.01
        self.p_max_fc = 1000 * self.n_cells * self.a_cell / 0.01  # Scaled max power
        self.ub = 2560          # Battery capacity from core.py designs
        self.p_bess = 2560      # Battery power rating

    # --- Surplus energy tests ---
    def test_small_surplus_charges_battery(self):
        """Test small surplus charges battery without switching to SOEC."""
        net = self.P_soec_min * 0.5  # Surplus < P_soec_min
        decision = ems(p_l=0, p_pv=net, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=0.5, previous_s=4, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 1, "Should charge battery for small surplus")

    def test_significant_surplus_soec(self):
        """Test significant surplus activates SOEC."""
        net = self.P_soec_min * 1.5  # Surplus > P_soec_min
        decision = ems(p_l=0, p_pv=net, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=0.5, previous_s=4, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 2, "Should activate SOEC for significant surplus")

    def test_surplus_h2_full_charges_battery(self):
        """Test surplus with full H2 tank charges battery via SOFC."""
        decision = ems(p_l=0, p_pv=50, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=1.0, previous_s=4, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 1, "Should charge battery when H2 tank is full")

    def test_curtailment(self):
        """Test surplus with full battery and H2 tank leads to curtailment."""
        decision = ems(p_l=0, p_pv=50, SOC=SOC_max, p_bess=self.p_bess, 
                      hydrogen_SOC=1.0, previous_s=4, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 8, "Should curtail when both storages are full")

    # --- Deficit energy tests ---
    def test_sofc_baseload_small_load(self):
        """Test SOFC covers small load at baseload."""
        load = self.P_sofc_base * self.n_cells * self.a_cell / 0.01 * 0.5  # Load < P_sofc_base
        decision = ems(p_l=-load, p_pv=0, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=0.5, previous_s=0, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 4, "SOFC should cover small load at baseload")

    def test_sofc_and_battery_peak_load(self):
        """Test SOFC at baseload with battery covering peak load."""
        load = self.P_sofc_base * self.n_cells * self.a_cell / 0.01 * 1.5  # Load > P_sofc_base
        decision = ems(p_l=-load, p_pv=0, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=0.5, previous_s=0, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 5, "SOFC at baseload, battery covers peak")

    def test_battery_supports_sofc(self):
        """Test battery supports SOFC baseload when load exceeds baseload."""
        load = self.P_sofc_base * self.n_cells * self.a_cell / 0.01 * 2  # Large load
        decision = ems(p_l=-load, p_pv=0, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=0.1, previous_s=0, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 9, "Battery should support SOFC baseload")

    def test_battery_only_no_hydrogen(self):
        """Test battery covers load when no hydrogen is available."""
        decision = ems(p_l=-200, p_pv=0, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=0.0, previous_s=0, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 3, "Battery should cover load when no hydrogen")

    def test_deficit_no_hydrogen_low_battery(self):
        """Test deficit when no hydrogen and battery is insufficient."""
        decision = ems(p_l=-10000, p_pv=0, SOC=SOC_min, p_bess=self.p_bess, 
                      hydrogen_SOC=0.0, previous_s=0, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 7, "Should enter deficit state")

    # --- SOFC max power constraint tests ---
    def test_baseload_capped_at_p_max_fc(self):
        """Test that baseload is capped at p_max_fc."""
        low_p_max_fc = 100 * self.n_cells * self.a_cell / 0.01  # Lower than 300W scaled
        decision = ems(p_l=-50, p_pv=0, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=0.5, previous_s=0, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=low_p_max_fc)
        self.assertEqual(decision, 4, "SOFC should run at capped baseload")

    # --- Edge case tests ---
    def test_low_hydrogen_sofc(self):
        """Test SOFC runs with low hydrogen, supported by battery."""
        decision = ems(p_l=-(self.P_sofc_base * self.n_cells * self.a_cell / 0.01), 
                      p_pv=0, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=0.01, previous_s=0, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 4, "SOFC should run with low hydrogen")

    def test_zero_hydrogen_shutdown(self):
        """Test SOFC shuts down when hydrogen_SOC = 0."""
        decision = ems(p_l=-(self.P_sofc_base * self.n_cells * self.a_cell / 0.01), 
                      p_pv=0, SOC=0.5, p_bess=self.p_bess, 
                      hydrogen_SOC=0.0, previous_s=4, ub=self.ub, 
                      n_cells=self.n_cells, a_cell=self.a_cell, p_max_fc=self.p_max_fc)
        self.assertEqual(decision, 3, "SOFC should shut down, battery takes over")

if __name__ == '__main__':
    unittest.main()