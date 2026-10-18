import json
from pathlib import Path
import unittest
from tests.d2_v2_chain_probe import probe
from tests.d2_v2_test_utils import evidence_dir

class RealTrainingChain(unittest.TestCase):
    def test_real_b2_b3_and_successor_with_bounded_updates(self):
        results=[]
        for repair in (False,True):
            for label in ('b2','b3'):
                results.append(probe(label,repair))
        (evidence_dir()/'runtime_chain.json').write_bytes((json.dumps(results,indent=2)+'\n').encode())
