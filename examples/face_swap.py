#!/usr/bin/env python3
"""Example client for the FaceSwap API.

Demonstrates async (queue), sync, and model comparison modes with Rich
terminal formatting: progress bars, tables, panels, and live status.
"""

import io
import sys
import time
import uuid
import base64

import requests
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

console = Console()

URL = "http://127.0.0.1:8090"

SOURCE_IMAGE = "../data/src.jpg"
TARGET_IMAGE = "../data/target.jpg"
SOURCE_INDEXES = "-1"
TARGET_INDEXES = "-1"
BACKGROUND_ENHANCE = True
FACE_RESTORE = True
FACE_UPSAMPLE = True
UPSCALE = 1
CODEFORMER_FIDELITY = 0.5
OUTPUT_FORMAT = "JPEG"

# Model: pick one from the 13 available models
FACE_SWAPPER_MODEL = "inswapper_128"
# FACE_SWAPPER_MODEL = "simswap_256"
# FACE_SWAPPER_MODEL = "ghost_1_256"
# FACE_SWAPPER_MODEL = "blendswap_256"
# FACE_SWAPPER_MODEL = "hififace_unofficial_256"

# Resolution override (None = auto-detect default per model)
FACE_SWAPPER_RESOLUTION = None
# FACE_SWAPPER_RESOLUTION = "1024x1024"  # higher quality, slower

# Identity blend (0.0 = more target identity, 1.0 = more source identity)
FACE_SWAPPER_WEIGHT = 1.0

# Mask controls
FACE_MASK_BLUR = 0.3
FACE_MASK_PADDING = "0,0,0,0"

# Face selector (target face filtering)
FACE_SELECTOR_MODE = "many"
FACE_SELECTOR_ORDER = "left-right"
# FACE_SELECTOR_GENDER = "female"
# FACE_SELECTOR_AGE_START = 20
# FACE_SELECTOR_AGE_END = 50


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and encode to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_result_image(image_b64: str, prefix: str = "result") -> str:
    """Decode base64 image and save to disk."""
    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    output_file = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    img.save(output_file, format="JPEG")
    return output_file


def build_payload():
    """Build the request payload from module-level settings."""
    return {
        "source_image": encode_image_to_base64(SOURCE_IMAGE),
        "target_image": encode_image_to_base64(TARGET_IMAGE),
        "source_indexes": SOURCE_INDEXES,
        "target_indexes": TARGET_INDEXES,
        "background_enhance": BACKGROUND_ENHANCE,
        "face_restore": FACE_RESTORE,
        "face_upsample": FACE_UPSAMPLE,
        "upscale": UPSCALE,
        "codeformer_fidelity": CODEFORMER_FIDELITY,
        "output_format": OUTPUT_FORMAT,
        "face_swapper_model": FACE_SWAPPER_MODEL,
        "face_swapper_resolution": FACE_SWAPPER_RESOLUTION,
        "face_swapper_weight": FACE_SWAPPER_WEIGHT,
        "face_mask_blur": FACE_MASK_BLUR,
        "face_mask_padding": FACE_MASK_PADDING,
        "face_selector_mode": FACE_SELECTOR_MODE,
        "face_selector_order": FACE_SELECTOR_ORDER,
    }


def _show_config():
    """Print current configuration in a panel."""
    lines = [
        f"[bold]Model:[/] {FACE_SWAPPER_MODEL}",
        f"[bold]Weight:[/] {FACE_SWAPPER_WEIGHT}",
        f"[bold]Selector:[/] {FACE_SELECTOR_MODE} / {FACE_SELECTOR_ORDER}",
        f"[bold]Resolution:[/] {FACE_SWAPPER_RESOLUTION or 'auto'}",
        f"[bold]Mask blur:[/] {FACE_MASK_BLUR}  [bold]Padding:[/] {FACE_MASK_PADDING}",
    ]
    console.print(Panel("\n".join(lines), title="Configuration", border_style="blue"))


# ---------------------------------------------------------------------------
# Async mode
# ---------------------------------------------------------------------------

STATUS_ICONS = {
    "queued": "🕐",
    "processing": "🔄",
    "completed": "✅",
    "failed": "❌",
}

STATUS_STYLES = {
    "queued": "yellow",
    "processing": "cyan",
    "completed": "green",
    "failed": "red",
}


def async_mode():
    """Submit a job and poll for the result with a live progress bar."""
    console.print(Panel("Async Mode", style="bold blue"))
    _show_config()

    payload = build_payload()
    start_time = time.time()

    # Submit
    r = requests.post(f"{URL}/faceswap", json=payload)
    resp = r.json()

    if r.status_code != 202:
        console.print(f"[red]Error ({r.status_code}):[/] {resp}")
        return

    job_id = resp["job_id"]
    status_url = resp["status_url"]
    console.print(f"Job ID: [bold cyan]{job_id}[/]")

    # Poll with progress bar
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[yellow]queued...", total=None,
        )

        while True:
            r = requests.get(f"{URL}{status_url}")
            data = r.json()
            status = data["status"]
            elapsed = time.time() - start_time

            # Update spinner description
            icon = STATUS_ICONS.get(status, "")
            style = STATUS_STYLES.get(status, "")
            progress.update(task, description=f"[{style}]{icon} {status}")

            if status == "completed":
                output = save_result_image(data["result"]["image"], "async")
                progress.update(task, completed=True, description="[green]✅ completed")
                console.print(f"[green]Saved:[/] [bold]{output}[/]")
                console.print(
                    f"[green]Total time:[/] [bold]{elapsed:.1f}s[/]"
                )
                break
            elif status == "failed":
                progress.update(task, description="[red]❌ failed")
                console.print(
                    f"[red]Failed:[/] {data.get('error', 'unknown error')}"
                )
                break

            time.sleep(1)


# ---------------------------------------------------------------------------
# Sync mode
# ---------------------------------------------------------------------------


def sync_mode():
    """Submit a synchronous face swap request."""
    console.print(Panel("Sync Mode", style="bold yellow"))
    _show_config()

    payload = build_payload()

    with console.status("[cyan]Processing face swap...", spinner="dots"):
        start_time = time.time()
        r = requests.post(f"{URL}/faceswap/sync", json=payload)
        elapsed = time.time() - start_time
    resp = r.json()

    if resp["status"] == "ok":
        output = save_result_image(resp["image"], "sync")
        console.print(f"[green]Saved:[/] [bold]{output}[/]")
        console.print(f"[green]Total time:[/] [bold]{elapsed:.1f}s[/]")
    else:
        console.print(f"[red]Error:[/] {resp.get('msg', 'unknown')}")
        if resp.get("detail"):
            console.print(f"[dim]{resp['detail']}[/]")


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_models(models=None):
    """Run the same swap across multiple models with live progress."""
    if models is None:
        models = [
            "inswapper_128",
            "simswap_256",
            "blendswap_256",
            "ghost_1_256",
        ]

    console.print(
        Panel(f"Comparing {len(models)} models", style="bold magenta")
    )
    console.print(f"Models: [italic]{', '.join(models)}[/]\n")

    results = {}

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for model in models:
            desc = f"[cyan]{model}"
            task = progress.add_task(desc, total=None)

            payload = build_payload()
            payload["face_swapper_model"] = model

            start = time.time()
            r = requests.post(f"{URL}/faceswap", json=payload)

            if r.status_code != 202:
                results[model] = {"status": "error", "time": time.time() - start}
                progress.update(
                    task, description=f"[red]{model} ✗ submit failed",
                    completed=True,
                )
                continue

            job = r.json()
            status_url = job["status_url"]

            while True:
                r = requests.get(f"{URL}{status_url}")
                data = r.json()
                if data["status"] == "completed":
                    elapsed = time.time() - start
                    results[model] = {"status": "ok", "time": elapsed}
                    output = save_result_image(
                        data["result"]["image"], f"compare_{model}",
                    )
                    progress.update(
                        task,
                        description=f"[green]{model} ✓ {elapsed:.1f}s → {output}",
                        completed=True,
                    )
                    break
                elif data["status"] == "failed":
                    elapsed = time.time() - start
                    results[model] = {"status": "failed", "time": elapsed}
                    progress.update(
                        task,
                        description=f"[red]{model} ✗ {data.get('error', 'unknown')}",
                        completed=True,
                    )
                    break
                time.sleep(1)

    # Summary table
    table = Table(title="Model Comparison Results", border_style="blue")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right", style="green")

    fastest = min(
        (i for i in results.values() if i["status"] == "ok"),
        key=lambda x: x["time"],
        default=None,
    )

    for model, info in results.items():
        status_icon = "✅" if info["status"] == "ok" else "❌"
        time_str = f"{info['time']:.1f}s"
        if fastest and info is fastest:
            time_str = f"[bold]{time_str} ⭐[/]"

        table.add_row(model, status_icon, time_str)

    console.print()
    console.print(table)

    if fastest:
        console.print(
            "\n[bold]⭐ Fastest:[/] "
            f"[cyan]{list(results.keys())[list(results.values()).index(fastest)]}[/]"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "async"

    if mode == "sync":
        sync_mode()
    elif mode == "compare":
        models = sys.argv[2:] if len(sys.argv) > 2 else None
        compare_models(models)
    else:
        async_mode()
