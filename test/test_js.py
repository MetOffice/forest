import os
import subprocess


JS_DIR = os.path.join(os.path.dirname(__file__), "../forest/js")


def test_forestjs():
    cwd = os.getcwd()
    os.chdir(JS_DIR)
    subprocess.check_call(["npm", "test"])
    os.chdir(cwd)
