"""Rich terminal interface for LLM Server."""

from __future__ import annotations

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .catalog import cached_models, delete, download, models, search
from .runtime import ServiceManager

app = typer.Typer(
    help="⚡ Manage local MLX language models on Apple Silicon.", no_args_is_help=True
)
models_app, services_app = typer.Typer(no_args_is_help=True), typer.Typer(no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(services_app, name="services")
console, manager = Console(), ServiceManager()


def show(items: list[dict], title: str) -> None:
    if not items:
        console.print("[yellow]No entries yet.[/yellow]")
        return
    table = Table(title=title, header_style="bold cyan")
    for key in items[0]:
        table.add_column(key.replace("_", " ").title())
    for item in items:
        table.add_row(*(str(value) for value in item.values()))
    console.print(table)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    """Start the localhost-only control-plane API."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise typer.BadParameter("Remote binding is unsupported; use a secure reverse proxy.")
    console.print(
        f"[bold green]● CONTROL PLANE[/bold green] http://{host}:{port}  [dim]docs: /docs[/dim]"
    )
    uvicorn.run("llm_server.api:app", host=host, port=port)


@models_app.command("list")
def model_list() -> None:
    show(models(), "✨ Curated MLX aliases")
    show(cached_models(), "📦 Downloaded models")


@models_app.command("search")
def model_search(query: str, limit: int = typer.Option(10, min=1, max=50)) -> None:
    show(search(query, limit), f"🔎 Hub search: {query}")


@models_app.command("download")
def model_download(identifier: str, revision: str | None = None) -> None:
    console.print(f"[cyan]⇣ Downloading[/cyan] {identifier}")
    console.print(f"[green]✓ Cached at[/green] {download(identifier, revision)}")


@models_app.command("delete")
def model_delete(identifier: str) -> None:
    delete(identifier)
    console.print(f"[green]✓ Deleted cached entry:[/green] {identifier}")


@services_app.command("status")
def service_status() -> None:
    show([s.model_dump() for s in manager.list()], "⚡ Managed services")


@services_app.command("start")
def service_start(
    model: str,
    name: str | None = None,
    port: int = 8080,
    max_kv_size: int | None = None,
    wait: bool = True,
) -> None:
    service = manager.start(model, name or model.replace("/", "--"), port, max_kv_size)
    console.print(f"[cyan]◌ STARTING[/cyan] {service.name} → http://127.0.0.1:{port}")
    if wait:
        service = manager.mark_ready(service.name)
        color = "green" if service.status == "ready" else "red"
        console.print(f"[{color}]● {service.status.upper()}[/{color}] {service.repository}")


@services_app.command("stop")
def service_stop(name: str) -> None:
    console.print(f"[yellow]■ STOPPED[/yellow] {manager.stop(name).name}")


@services_app.command("restart")
def service_restart(name: str) -> None:
    console.print(f"[cyan]◌ RESTARTING[/cyan] {name}")
    console.print(
        f"[green]● {manager.mark_ready(manager.restart(name).name).status.upper()}[/green]"
    )


@services_app.command("logs")
def service_logs(name: str, lines: int = typer.Option(80, min=1, max=500)) -> None:
    console.print(manager.logs(name, lines))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
