import unittest
import subprocess
import signal
import time
try:
    from selenium import webdriver
    from selenium.common.exceptions import WebDriverException
    from selenium.webdriver.firefox.options import Options
except ImportError:
    pass


@unittest.skip("integration test")
class TestIntegration(unittest.TestCase):
    def setUp(self):
        options = Options()
        options.headless = True
        self.driver = webdriver.Firefox(options=options)

    def tearDown(self):
        self.driver.quit()

    def test_command_line_forest(self):
        process = subprocess.Popen(["forest", "file.json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        time.sleep(5)
        try:
            self.driver.get("http://localhost:5006")
        except WebDriverException as e:
            print(e.msg)
        process.terminate()
        o, e = process.communicate()
        print(o.decode())
        print(e.decode())
        self.assertTrue(False)
