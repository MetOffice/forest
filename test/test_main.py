import os
from forest.main import main


def test_main_given_rdt_files(tmp_path):
    rdt_file = tmp_path / "file.json"
    with rdt_file.open("w"):
        pass
    main(argv=["--file-type", "rdt", str(rdt_file)])
