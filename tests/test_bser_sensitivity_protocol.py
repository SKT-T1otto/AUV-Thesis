import unittest
from chapter3_bser.config import load_bser_phase1a1_config

class SensitivityProtocolTest(unittest.TestCase):
    def test_seven_fixed_ofat_variants(self):
        variants=load_bser_phase1a1_config()["sensitivity"]["variants"]; self.assertEqual(len(variants),7); self.assertEqual(variants[0],{"id":"baseline","p_scale":1.0,"sigma_multiplier":1.0,"tau_executor":18.0})
