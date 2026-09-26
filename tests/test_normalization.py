import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import unittest
from src.preprocessing.name_normalizer import normalize_name
from src.preprocessing.address_normalizer import normalize_address

class TestNormalization(unittest.TestCase):
    def test_name_normalizer(self):
        self.assertEqual(normalize_name("Amazon.com, Inc."), "amazon com inc")
        self.assertEqual(normalize_name("  Test   Corp! "), "test corp")
        self.assertEqual(normalize_name(None), "")
        
    def test_address_normalizer(self):
        self.assertEqual(normalize_address("123 Fake St."), "123 fake street")
        self.assertEqual(normalize_address("456 Blvd, Apt 1"), "456 boulevard apt 1")
        self.assertEqual(normalize_address("  10  Main Ave  "), "10 main avenue")
        self.assertEqual(normalize_address(None), "")

if __name__ == '__main__':
    unittest.main()
