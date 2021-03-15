#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/../')
import unittest
from unittest.mock import patch
from unittest import mock
import filecmp
import numpy as np

import metadynamics
import functions

run_dir = os.path.dirname(os.path.realpath(__file__))+'/'
class TestSum(unittest.TestCase):
    def test_find_min(self):
        e1 = 10.1
        result = functions.find_min(e1)
        self.assertEqual(result, -15)




if __name__ == '__main__':
    unittest.main()
