import math
import unittest

import numpy as np

from grain_burner.model import (
    FORM_CYLINDER,
    FORM_END,
    FORM_SLOT,
    FORM_STAR,
    INSTALL_BONDED,
    INSTALL_FREE,
    GrainParameters,
    burning_area_components,
    calculate_curve,
)


class GrainModelTests(unittest.TestCase):
    def test_end_burning_area_is_constant(self):
        p = GrainParameters(form=FORM_END, diameter_mm=200, length_mm=500)
        expected = math.pi * 0.2**2 / 4
        self.assertAlmostEqual(sum(burning_area_components(0, p).values()), expected, places=9)
        self.assertAlmostEqual(sum(burning_area_components(400, p).values()), expected, places=9)

    def test_bonded_cylindrical_channel_is_progressive(self):
        p = GrainParameters(
            form=FORM_CYLINDER,
            installation=INSTALL_BONDED,
            diameter_mm=240,
            length_mm=1000,
            web_mm=50,
        )
        start = sum(burning_area_components(0, p).values())
        finish = sum(burning_area_components(50, p).values())
        self.assertGreater(finish, start)
        self.assertAlmostEqual(start, math.pi * 0.140, places=9)

    def test_free_cylindrical_channel_is_neutral(self):
        p = GrainParameters(
            form=FORM_CYLINDER,
            installation=INSTALL_FREE,
            diameter_mm=240,
            length_mm=1000,
            web_mm=50,
        )
        curve = calculate_curve(p, samples=21)
        self.assertEqual(curve.burn_limit_mm, 25)
        self.assertTrue(np.allclose(curve.area_m2, curve.area_m2[0], rtol=0, atol=1e-12))

    def test_manual_cylindrical_channel_diameter_overrides_web(self):
        p = GrainParameters(
            form=FORM_CYLINDER,
            installation=INSTALL_BONDED,
            diameter_mm=240,
            length_mm=1000,
            web_mm=10,
            channel_diameter_override_mm=100,
        )
        curve = calculate_curve(p, samples=21)
        self.assertEqual(p.channel_diameter_mm, 100)
        self.assertEqual(p.physical_web_mm, 70)
        self.assertEqual(curve.burn_limit_mm, 70)
        self.assertAlmostEqual(curve.area_m2[0], math.pi * 0.100, places=9)

    def test_star_matches_manual_example(self):
        p = GrainParameters(
            form=FORM_STAR,
            installation=INSTALL_BONDED,
            diameter_mm=240,
            length_mm=1466.916,
            web_mm=50,
            star_points=8,
            star_half_angle_deg=32,
            star_root_radius_mm=2.5,
        )
        curve = calculate_curve(p, samples=10001)
        self.assertAlmostEqual(p.star_stage1_end_mm(), 27.9595, places=3)
        self.assertAlmostEqual(p.star_complete_burn_mm(), 60.6617, places=3)
        self.assertAlmostEqual(curve.area_m2[0], 1.132625, places=5)
        self.assertAlmostEqual(curve.average_area_m2, 0.941992, places=4)

    def test_slot_matches_manual_average_area(self):
        p = GrainParameters(
            form=FORM_SLOT,
            installation=INSTALL_BONDED,
            diameter_mm=240,
            length_mm=1385.585,
            web_mm=90,
            slot_count=4,
            slot_depth_mm=450,
            slot_width_mm=10,
        )
        curve = calculate_curve(p, samples=10001)
        self.assertAlmostEqual(curve.area_m2[0], 0.610639, places=5)
        self.assertAlmostEqual(curve.average_area_m2, 0.634921, places=5)


if __name__ == "__main__":
    unittest.main()
