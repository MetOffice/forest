import yaml


def test_load_given_str():
    assert yaml.load("""
    labels:
        - hello
        - world
    """) == {'labels': ['hello', 'world']}
