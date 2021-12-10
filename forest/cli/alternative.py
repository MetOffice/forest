"""
FOREST command line application
"""
from pathlib import Path
import typer
import subprocess
from typing import List

APP_NAME = "forest"
BOKEH_APP_PATH = Path(__file__).resolve().parent.parent


app = typer.Typer(help="Command line interface for FOREST web application")


@app.command()
def edit():
    """Edit catalogues"""
    app_path = Path(typer.get_app_dir(APP_NAME))
    if not app_path.is_dir():
        app_path.mkdir()

    config_path: Path = app_path / "config.yaml"
    if not config_path.is_file():
        config_path.write_text('{"version": "1.0.0"}')

    print(config_path)
    typer.launch(str(config_path), locate=True)


@app.command()
def view(files: List[Path], driver: str = "gridded_forecast"):
    """Quickly browse file(s)"""
    typer.echo(files)
    typer.secho("Launching Bokeh...", fg=typer.colors.MAGENTA)
    forest_args = ["--file-type", driver] + [str(f) for f in files]
    command = ["bokeh", "serve", str(BOKEH_APP_PATH), "--args"] + forest_args
    typer.secho(" ".join(command), fg=typer.colors.MAGENTA)
    subprocess.call(command)


@app.command()
def ctl():
    """Control a collection of data sources"""
    typer.secho("Launching Bokeh...", fg=typer.colors.MAGENTA)
    subprocess.call(["bokeh", "-h"])
