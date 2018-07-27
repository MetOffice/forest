import unittest
import unittest.mock
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../bokeh_apps/plot_sea_two_model_comparison"))
import main


class TestPlotSeaTwoModelComparison(unittest.TestCase):
    """Need to develop appropriate testing strategy"""
