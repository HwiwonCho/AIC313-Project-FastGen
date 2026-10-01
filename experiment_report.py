"""Turn an experiment's losses.csv into a plot and a Notion-ready record.

Usage: python experiment_report.py runs/soflow_v2
       python experiment_report.py runs/conditional_v4_losses.csv

Only Pillow (already in the course requirements.txt) is needed.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_history(csv_path: Path) -> tuple[list[dict[str, str]], dict[str, list[tuple[float, float]]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ValueError("CSV needs a header with an 'epoch' column")
        rows = list(reader)
        metric_names = [
            name for name in reader.fieldnames
            if any(word in name.lower() for word in ("loss", "mse", "score"))
        ]

    if not rows:
        raise ValueError("CSV has no completed epochs")

    series: dict[str, list[tuple[float, float]]] = {}
    for name in metric_names:
        points = []
        for row in rows:
            try:
                epoch, value = float(row["epoch"]), float(row[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(epoch) and math.isfinite(value):
                points.append((epoch, value))
        if points:
            series[name] = points

    if not series:
        raise ValueError("CSV has no finite loss, MSE, or score values")
    return rows, series


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_curves(series: dict[str, list[tuple[float, float]]], output: Path) -> None:
    columns, panel_width, panel_height, gap = 2, 570, 250, 18
    rows = math.ceil(len(series) / columns)
    width = columns * panel_width + (columns + 1) * gap
    height = rows * panel_height + (rows + 1) * gap + 48
    image = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(image)
    draw.text((gap, 13), "Training curves", fill="#172b4d", font=font(24))
    colors = ("#2563eb", "#ea580c", "#059669", "#9333ea", "#be123c")

    for index, (name, points) in enumerate(series.items()):
        col, row = index % columns, index // columns
        x0 = gap + col * (panel_width + gap)
        y0 = 48 + gap + row * (panel_height + gap)
        x1, y1 = x0 + panel_width, y0 + panel_height
        draw.rounded_rectangle((x0, y0, x1, y1), radius=9, fill="white", outline="#dbe2ea")
        draw.text((x0 + 18, y0 + 12), name, fill="#172b4d", font=font(17))

        left, top, right, bottom = x0 + 72, y0 + 47, x1 - 20, y1 - 42
        epochs = [epoch for epoch, _ in points]
        values = [value for _, value in points]
        epoch_min, epoch_max = min(epochs), max(epochs)
        value_min, value_max = min(values), max(values)
        spread = value_max - value_min
        padding = spread * 0.08 if spread else max(abs(value_min) * 0.08, 0.01)
        value_min -= padding
        value_max += padding

        for tick in range(5):
            y = round(bottom - tick * (bottom - top) / 4)
            value = value_min + tick * (value_max - value_min) / 4
            draw.line((left, y, right, y), fill="#e7ecf2", width=1)
            draw.text((x0 + 9, y - 8), f"{value:.3g}", fill="#65758b", font=font(12))
        draw.line((left, top, left, bottom), fill="#94a3b8", width=1)
        draw.line((left, bottom, right, bottom), fill="#94a3b8", width=1)
        for tick in range(5):
            x = round(left + tick * (right - left) / 4)
            epoch = epoch_min + tick * (epoch_max - epoch_min) / 4
            draw.text((x - 12, bottom + 8), f"{epoch:.0f}", fill="#65758b", font=font(12))

        epoch_range = epoch_max - epoch_min or 1
        value_range = value_max - value_min
        pixels = [
            (
                round(left + (epoch - epoch_min) / epoch_range * (right - left)),
                round(bottom - (value - value_min) / value_range * (bottom - top)),
            )
            for epoch, value in points
        ]
        color = colors[index % len(colors)]
        if len(pixels) > 1:
            draw.line(pixels, fill=color, width=3, joint="curve")
        x, y = pixels[-1]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((x1 - 150, y0 + 14), f"last {values[-1]:.4g}", fill=color, font=font(13))

    image.save(output)


def log_facts(directory: Path) -> dict[str, str]:
    log_path = directory / "train.log"
    if not log_path.exists():
        return {}
    facts = {}
    labels = {"device", "gpu", "train images", "batches per epoch", "parameters", "sample rows"}
    with log_path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            key, separator, value = line.partition(":")
            if separator and key in labels and key not in facts:
                facts[key] = value.strip()
    return facts


def sample_range(directory: Path) -> tuple[str, str]:
    sample_dir = directory / "samples"
    if not sample_dir.is_dir():
        return "not found", "not found"
    samples = []
    for path in sample_dir.glob("epoch_*.png"):
        match = re.search(r"epoch_(\d+)", path.name)
        if match and ("ema" in path.stem or "raw" not in path.stem):
            samples.append((int(match.group(1)), path))
    if not samples:
        return "not found", "not found"
    samples.sort()
    return samples[0][1].relative_to(directory).as_posix(), samples[-1][1].relative_to(directory).as_posix()


def write_summary(
    csv_path: Path,
    rows: list[dict[str, str]],
    series: dict[str, list[tuple[float, float]]],
    plot_path: Path,
    output: Path,
) -> None:
    first_epoch = rows[0]["epoch"]
    last_epoch = rows[-1]["epoch"]
    facts = log_facts(csv_path.parent)
    first_sample, last_sample = sample_range(csv_path.parent)
    checkpoint = csv_path.parent / "latest.ckpt"
    vram_values = []
    for row in rows:
        try:
            value = float(row.get("peak_vram_gb", ""))
            if math.isfinite(value):
                vram_values.append(value)
        except (TypeError, ValueError):
            pass

    lines = [
        f"# Experiment record: {csv_path.parent.name}",
        "",
        "> Generated from the loss CSV; rerunning this tool overwrites this file. Fill the TODO fields in Notion.",
        "",
        "## Setup and provenance",
        "",
        "- Goal / hypothesis: TODO",
        "- Method and paper citations: TODO",
        "- GitHub URL and commit used for this run: TODO (the current checkout may differ)",
        "- Training command and configuration: TODO",
        "- Dataset split used: TODO (verify training split only)",
        f"- Logged epochs: {first_epoch}–{last_epoch} ({len(rows)} rows)",
        "- Exact training time: TODO (not recorded in this CSV)",
        f"- Peak allocated VRAM: {max(vram_values):.2f} GiB" if vram_values else "- Peak allocated VRAM: TODO",
    ]
    for key, value in facts.items():
        lines.append(f"- {key.capitalize()}: {value}")
    lines += [
        "",
        "## Results",
        "",
        f"- Loss curve: {plot_path.name}",
        f"- Early sample: {first_sample}",
        f"- Latest sample: {last_sample}",
        f"- Checkpoint file: {checkpoint.name if checkpoint.exists() else 'not found'} (saved epoch must be checked)",
        "- Quantitative evaluation / FID: TODO (do not infer from training loss)",
        "- Qualitative observations: TODO",
        "- Next experiment / decision: TODO",
        "",
        "| Metric | First | Latest | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, points in series.items():
        values = [value for _, value in points]
        lines.append(
            f"| {name} | {values[0]:.5g} | {values[-1]:.5g} | "
            f"{min(values):.5g} | {max(values):.5g} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="run directory or a loss CSV file")
    args = parser.parse_args()
    csv_path = args.path / "losses.csv" if args.path.is_dir() else args.path
    if not csv_path.is_file():
        parser.error(f"loss CSV not found: {csv_path}")
    prefix = "" if csv_path.name == "losses.csv" else csv_path.stem + "_"
    plot_path = csv_path.with_name(prefix + "loss_curves.png")
    summary_path = csv_path.with_name(prefix + "experiment_summary.generated.md")
    try:
        rows, series = read_history(csv_path)
        draw_curves(series, plot_path)
        write_summary(csv_path, rows, series, plot_path, summary_path)
    except ValueError as error:
        parser.error(str(error))
    print(f"Plot: {plot_path}")
    print(f"Notion draft: {summary_path}")


if __name__ == "__main__":
    main()
