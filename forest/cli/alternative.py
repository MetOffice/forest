"""
FOREST command line application
"""
from pathlib import Path
import typer
import subprocess
from typing import List
import forest.db.main
import forest.tutorial.main

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


PORT_OPTION = typer.Option(None, help="bokeh server port")


@app.command()
def view(
    files: List[Path],
    driver: str = "gridded_forecast",
    open_tab: bool = True,
    # Bokeh specific arguments
    port: int = PORT_OPTION,
):
    """Quickly browse file(s)"""
    typer.secho("Launching Bokeh...", fg=typer.colors.MAGENTA)
    forest_args = ["--file-type", driver] + [str(f) for f in files]
    bokeh_args = ["bokeh", "serve", str(BOKEH_APP_PATH)]
    if port:
        bokeh_args += ["--port", str(port)]
    if open_tab:
        bokeh_args.append("--show")
    command = bokeh_args + ["--args"] + forest_args
    typer.secho(" ".join(command), fg=typer.colors.CYAN)
    return subprocess.call(command)


@app.command()
def ctl(config_file: Path, open_tab: bool = True, port: int = PORT_OPTION):
    """Control a collection of data sources"""
    typer.secho("Launching Bokeh...", fg=typer.colors.MAGENTA)
    forest_args = ["--config-file", str(config_file)]
    bokeh_args = ["bokeh", "serve", str(BOKEH_APP_PATH)]
    if port:
        bokeh_args += ["--port", str(port)]
    if open_tab:
        bokeh_args.append("--show")
    command = bokeh_args + ["--args"] + forest_args
    typer.secho(" ".join(command), fg=typer.colors.CYAN)
    return subprocess.call(command)


app.command(name="tutorial")(forest.tutorial.main.main)


app.command(name="db")(forest.db.main.main)
