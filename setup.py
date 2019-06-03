import setuptools


setuptools.setup(
        name="forest",
        version="0.0.1",
        author="Andrew Ryan",
        author_email="andrew.ryan@metoffice.gov.uk",
        description="Forecast visualisation and survey tool",
        packages=setuptools.find_packages(),
        entry_points={
            'console_scripts': [
                'forest=forest.cli.main:main'
            ]
        })
