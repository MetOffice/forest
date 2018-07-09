"""Unit test runner to call npm test"""
import unittest
import os
import subprocess


class TestForestJS(unittest.TestCase):
    def test_forestjs(self):
        os.chdir("../forest_lib/forestjs")
        process = subprocess.Popen(["npm", "test"],
                                   stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        message = stdout.decode("utf-8", errors="ignore")
        os.chdir("../../test")
        print(message)
        if process.returncode != 0:
            assert False
