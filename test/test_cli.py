import pytest
import forest.cli.main


@pytest.mark.parametrize("argv,expect", [
    (["--allow-websocket-origin", "eld388:8080", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--allow-websocket-origin", "eld388:8080",
                "--args", "file.nc"]),
    (["--dev", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--dev",
                "--args", "file.nc"]),
    (["--port", "5006", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--port", "5006",
                "--args", "file.nc"]),
    (["--show", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--show",
                "--args", "file.nc"]),
    (["--show", "file.nc"],
     ["bokeh", "serve", "/app/path",
                "--show",
                "--args", "file.nc"]),
    ])
def test_bokeh_command(argv, expect):
    result = forest.cli.main.bokeh_command("/app/path", argv)
    assert expect == result
