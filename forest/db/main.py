import typer
from typing import List
from pathlib import Path


def main(
    database: Path = typer.Argument(..., help="database file to write/extend"),
    files: List[Path] = typer.Argument(..., help="unified model netcdf files"),
):
    """Generate database to accelerate big data navigation"""
    import forest.db.database

    typer.secho(f"open: {database}", fg=typer.colors.CYAN)
    with forest.db.database.Database.connect(str(database)) as handle:
        for path in files:
            typer.secho(f"insert records: {path}", fg=typer.colors.YELLOW)
            handle.insert_netcdf(str(path))
    typer.secho(f"close: {database}", fg=typer.colors.CYAN)
