import unittest
import server


class TestParseArgs(unittest.TestCase):
    def test_show(self):
        args = server.parse_args([])
        result = args.show
        expect = False
        self.assertEqual(expect, result)
