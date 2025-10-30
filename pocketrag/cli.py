from __future__ import annotations
import json, pathlib, sys
import typer
from rich.console import Console
from rich import box
from rich.table import Table
from importlib.metadata import version as dist_version, PackageNotFoundError

from .utils.hw import get_system_report

app = typer.Typer(add_completion=False)
console = Console()

def _pkg_version(name: str) -> str:
    try:
        return dist_version(name)
    except PackageNotFoundError:
        return "unknown"

@app.command("version")
def _show_version():
    """Show PocketRAG and key dependency versions."""
    rows = {
        "pocketrag": _pkg_version("pocketrag"),
        "python": sys.version.split()[0],
    }
    for lib in ["torch", "transformers", "sentence-transformers", "faiss-cpu"]:
        rows[lib] = _pkg_version(lib)

    table = Table(title="Versions", box=box.SIMPLE)
    table.add_column("Package")
    table.add_column("Version")
    for k, v in rows.items():
        table.add_row(k, v)
    console.print("\n")
    console.print(table)

@app.command()
def hw():
    """Print hardware & framework capabilities (for result reproducibility)."""
    info = get_system_report()
    console.print_json(data=info)

@app.command()
def init(
    data_dir: str = typer.Option("./data", help="Directory for datasets & indexes"),
    results_dir: str = typer.Option("./results", help="Directory to write benchmark outputs"),
):
    """Create standard folders and a default config file."""
    dd = pathlib.Path(data_dir); rd = pathlib.Path(results_dir)
    dd.mkdir(parents=True, exist_ok=True); rd.mkdir(parents=True, exist_ok=True)
    cfg = {
        "data_dir": str(dd.resolve()),
        "results_dir": str(rd.resolve()),
        "seed": 42,
        "device": "auto"  # auto|cpu|cuda|mps
    }
    (pathlib.Path("pocketrag.yaml")).write_text(json.dumps(cfg, indent=2))
    console.print(f"[green]Initialized.[/green] data_dir={dd} results_dir={rd} config=pocketrag.yaml")