import argparse


def test_parse_known_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo")
    args, extra = parser.parse_known_args(["--other", "arg", "--foo", "bar"])
    assert args.foo == "bar"
    assert extra == ["--other", "arg"]
