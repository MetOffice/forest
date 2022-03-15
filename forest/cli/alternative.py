"""
FOREST command line application
"""
from pathlib import Path
import typer
import subprocess
from typing import List, Optional
import forest.db.main
import forest.tutorial.main

APP_NAME = "forest"
BOKEH_APP_PATH = Path(__file__).resolve().parent.parent


app = typer.Typer(help="Command line interface for FOREST web application")

# Driver sub-command
driver_app = typer.Typer(help="Describe available drivers")
app.add_typer(driver_app, name="driver")


@driver_app.command()
def list():
    """List keywords associated with drivers"""
    import forest.drivers

    typer.secho("Listing builtin drivers...\n", fg=typer.colors.CYAN)
    for name in sorted(forest.drivers.iter_drivers()):
        typer.echo(name)


def version_callback(value: bool):
    if value:
        typer.secho(f"🌲 Version {forest.__version__} ✨", fg=typer.colors.CYAN)
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        is_eager=True,
        help="Print version number",
        callback=version_callback,
    )
):
    pass


OPTION_PORT = typer.Option(None, help="Bokeh --port")
OPTION_WEBSOCKET = typer.Option(None, help="Bokeh --allow-websocket-origin")
OPTION_DEV = typer.Option(None, help="Bokeh --dev")


@app.command(hidden=True)
def edit():
    """Edit catalogues

    Work in progress
    """
    app_path = Path(typer.get_app_dir(APP_NAME))
    if not app_path.is_dir():
        app_path.mkdir()

    config_path: Path = app_path / "config.yaml"
    if not config_path.is_file():
        config_path.write_text('{"version": "1.0.0"}')

    print(config_path)
    typer.launch(str(config_path), locate=True)


@app.command()
def view(
    files: List[Path],
    driver: str = "gridded_forecast",
    open_tab: bool = True,
    # Bokeh specific arguments
    port: int = OPTION_PORT,
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
def ctl(
    config_file: Path,
    open_tab: bool = True,
    dev: bool = OPTION_DEV,
    port: int = OPTION_PORT,
    allow_websocket_origin: str = OPTION_WEBSOCKET,
):
    """Control a collection of data sources"""
    typer.secho("Launching Bokeh...", fg=typer.colors.MAGENTA)
    forest_args = ["--config-file", str(config_file)]
    bokeh_args = ["bokeh", "serve", str(BOKEH_APP_PATH)]
    if port:
        bokeh_args += ["--port", str(port)]
    if dev:
        bokeh_args += ["--dev"]
    if allow_websocket_origin:
        bokeh_args += ["--allow-websocket-origin", allow_websocket_origin]
    if open_tab:
        bokeh_args.append("--show")
    command = bokeh_args + ["--args"] + forest_args
    typer.secho(" ".join(command), fg=typer.colors.CYAN)
    return subprocess.call(command)


app.command(name="tutorial")(forest.tutorial.main.main)


app.command(name="db")(forest.db.main.main)


@app.command()
def init(
    config_file: Path = "forest.config.yaml",
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing file"
    ),
):
    """Quickly initialise a template"""
    if config_file.exists() and not force:
        name = typer.style(f"{config_file}", fg=typer.colors.CYAN)
        typer.secho(f"\n{name} already exists")
        flag = typer.style(f"--force", bold=True)
        typer.echo(f"use {flag} to overwrite it or")
        flag = typer.style(f"--config-file", bold=True)
        typer.echo(f"use {flag} to specify a different file\n")
        raise typer.Exit()

    # Build a template configuration file
    typer.secho("\nFile name:", fg=typer.colors.BLUE)
    typer.secho(f"{config_file}", fg=typer.colors.CYAN)
    from forest.config import Edition2022, HighLevelDataset, HighLevelDriver
    from dataclasses import asdict
    import yaml

    driver = HighLevelDriver("", {})
    dataset = HighLevelDataset("", "", driver)
    config = Edition2022(datasets=[dataset])
    with config_file.open("w") as stream:
        yaml.dump(asdict(config), stream, sort_keys=False)

    # Summary
    typer.secho("\nContents:", fg=typer.colors.BLUE)
    with config_file.open() as stream:
        text = stream.read()
        typer.secho(text, fg=typer.colors.CYAN)

    # Instructions
    typer.secho("\nNext steps:", fg=typer.colors.BLUE)
    typer.secho("Use your favourite editor to modify", fg=typer.colors.CYAN)
    typer.secho(f"{config_file}\n", bold=True)
    typer.secho("Then visualise your data using:", fg=typer.colors.CYAN)
    typer.secho(f"forest ctl {config_file}\n", bold=True)
