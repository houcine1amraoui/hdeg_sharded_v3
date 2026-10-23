"""Small contract tests; no model or data writes."""
import unittest
from run_v3_annotation_coverage_audit import components,select_windows,partitions,seconds,DAY
class ContractTests(unittest.TestCase):
    def test_touching_not_overlapping(self):
        self.assertEqual(len(components([dict(start=0,end=30,row=1),dict(start=30,end=60,row=2)])),2)
    def test_transitive_overlap(self):
        self.assertEqual(len(components([dict(start=0,end=30,row=1),dict(start=20,end=50,row=2),dict(start=45,end=60,row=3)])),1)
    def test_windows_bounded_disjoint(self):
        starts=select_windows(10,1010,0)
        self.assertEqual(len(starts),8)
        self.assertTrue(all(b-a>=30 for a,b in zip(starts,starts[1:])))
        self.assertTrue(all(s>=10 and s+30<=1010 for s in starts))
        self.assertEqual(select_windows(0,29,0),[])
    def test_complete_days_and_gaps(self):
        spans,plan=partitions(0,18*DAY+123)
        self.assertEqual(plan['complete_days'],18)
        self.assertEqual(spans['selection'][0]-spans['fit'][1],DAY)
        self.assertEqual(spans['assessment'][0]-spans['selection'][1],DAY)
        self.assertEqual(spans['assessment'][1],18*DAY)
    def test_no_silent_subsecond_rounding(self):
        with self.assertRaises(ValueError):seconds('2022-10-20 00:00:00.5')
if __name__=='__main__':unittest.main()
