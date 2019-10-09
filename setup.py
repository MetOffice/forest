import os
import re
import setuptools


NAME = "forest"


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


setuptools.setup(
        name=NAME,
        version=find_version(),
        author="Andrew Ryan",
        author_email="andrew.ryan@metoffice.gov.uk",
        description="Forecast visualisation and survey tool",
        packages=setuptools.find_packages(),
        package_data={
            "forest": [
                "templates/index.html",
                "tutorial/*.json",
                "tutorial/*.nc"
            ]
        },
        test_suite="test",
        tests_require=load("requirements-dev.txt"),
        entry_points={
            'console_scripts': [
                'forest=forest.cli.main:main',
                'forestdb=forest.db.main:main',
                'forest-tutorial=forest.tutorial.main:main'
            ]
        })
