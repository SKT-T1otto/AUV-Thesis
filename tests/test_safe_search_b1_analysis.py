"""Paired metric accounting and unavailable-rate handling; no simulation."""
import unittest

from docs.chapter3.search_diagnostics.safe_search_v1.analyze_b1_development import comparison
from chapter3_bser.experiments.safe_search_v1.run_development import INDICES


class PairedTransferTests(unittest.TestCase):
    def test_matched_discordance_and_direction(self):
        values={(i,v):dict(found=int((i==INDICES[0] and v=='V5') or (i==INDICES[1] and v=='V3')))
                for i in INDICES for v in ('V3','V5')}
        result=comparison(values,'V5','V3')['found']
        self.assertEqual(result['mean_delta'],0)
        self.assertEqual(result['gained_indices'],[INDICES[0]])
        self.assertEqual(result['lost_indices'],[INDICES[1]])
        self.assertEqual(result['n_pairs'],20)
        self.assertEqual(result['direction'],'V5 minus V3')

    def test_zero_exposure_is_excluded_not_imputed(self):
        values={(i,v):dict(rate=None if i==INDICES[0] and v=='V0' else 0.5)
                for i in INDICES for v in ('V0','V5')}
        result=comparison(values,'V5','V0')['rate']
        self.assertEqual(result['n_pairs'],19)
        self.assertEqual(result['excluded_zero_exposure_indices'],[INDICES[0]])
        self.assertEqual(result['mean_delta'],0)

    def test_missing_scene_does_not_shrink_denominator_silently(self):
        values={(i,v):dict(found=0) for i in INDICES[:-1] for v in ('V0','V5')}
        with self.assertRaises(KeyError):
            comparison(values,'V5','V0')


if __name__=='__main__':
    unittest.main()
