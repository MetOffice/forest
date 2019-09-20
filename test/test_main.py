import unittest
from forest.main import main


class TestMain(unittest.TestCase):
    @unittest.skip("waiting for green light")
    def test_main_given_files(self):
        main(argv=["file.json"])
