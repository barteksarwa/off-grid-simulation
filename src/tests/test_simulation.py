# src/tests/test_simulation.py
import unittest
import numpy as np
from src.simulation.core import f

class TestSimulation(unittest.TestCase):
    def test_simulation_runs(self):
        # Test that simulation runs without errors for a simple case
        time_csv, result = f(10, 5, 20, 2)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[1], 11)  # Should have 11 columns
        
if __name__ == '__main__':
    unittest.main()