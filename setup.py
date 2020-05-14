"""
   FOREST - Forecast and Observation Research and Evaluation Survey Tool
"""
import os
import re
import subprocess
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.install


NAME = "forest"
JS_DIR = os.path.join(os.path.dirname(__file__), NAME, r"js")


def find_version():
    path = os.path.join(os.path.dirname(__file__), NAME, "__init__.py")
    with open(path) as stream:
        contents = stream.read()
    match = re.search(r"^__version__ = ['\"']([0-9\.]*)['\"']", contents, re.MULTILINE)
    if match:
        return match.group(1)
    else:
        raise RuntimeError("Unable to find version number")


def load(fname):
    result = []
    with open(fname, 'r') as fi:
        result = [package.strip() for package in fi.readlines()]
    return result


def build_js(command_subclass):
    """Decorator to call npm install and npm run build"""
    subclass_run = command_subclass.run
    def run(self):
        self.run_command("build_js")
        subclass_run(self)
    command_subclass.run = run
    return command_subclass


@build_js
class InstallCommand(setuptools.command.install.install):
    """Python and JS code"""


@build_js
class DevelopCommand(setuptools.command.develop.develop):
    """Python and JS code"""


@build_js
class BuildPyCommand(setuptools.command.build_py.build_py):
    """Python and JS code"""


class BuildJSCommand(setuptools.command.build_py.build_py):
    """Use nodejs and npm commands to browserify forest.js

    .. note:: Assume current working directory is package ROOT
    """
    def run(self):
        cwd = os.getcwd()
        os.chdir(JS_DIR)
        if not os.path.exists("node_modules"):
            subprocess.check_call(["npm", "install"])
        subprocess.check_call(["npm", "run", "build"])
        os.chdir(cwd)
        super().run()


setuptools.setup(
        name=NAME,
        version=find_version(),
        author="Andrew Ryan",
        author_email="andrew.ryan@metoffice.gov.uk",
        cmdclass={
            "install": InstallCommand,
            "develop": DevelopCommand,
            "build_py": BuildPyCommand,
            "build_js": BuildJSCommand,
        },
        description="Forecast visualisation and survey tool",
        packages=setuptools.find_packages(),
        package_data={
            "forest": [
                "templates/index.html",
                "static/*",
                "tutorial/*.json",
                "tutorial/*.nc"
            ]
        },
        test_suite="test",
        tests_require=load("requirements-dev.txt"),
        extras_require={
            ':python_version == "3.6"': [
                "dataclasses"
            ]
        },
        entry_points={
            'console_scripts': [
                'forest=forest.cli.main:main',
                'forestdb=forest.db.main:main',
                'forest-tutorial=forest.tutorial.main:main'
            ]
        })
