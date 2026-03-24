#!/usr/bin/env python3
"""
Graficas para reporte academico: metricas del modo 'experiment' de proyecto.cpp.

Entrada: CSV con columnas
  dataset, n_points, dimension, k, repetitions, threads,
  avg_time_serial, avg_time_parallel, avg_speedup, avg_efficiency,
  avg_improvement_percent, avg_parallel_cost,
  avg_wcss_serial, avg_wcss_parallel,
  avg_iterations_serial, avg_iterations_parallel

Salida: speedup.png, tiempo.png, eficiencia.png

Uso:
  python visualizar_metricas_experimento.py --csv metricas.csv
  python visualizar_metricas_experimento.py --csv metricas.csv --outdir figuras/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
except ImportError:
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})


EXPECTED_COLUMNS = [
    "dataset",
    "n_points",
    "dimension",
    "k",
    "repetitions",
    "threads",
    "avg_time_serial",
    "avg_time_parallel",
    "avg_speedup",
    "avg_efficiency",
    "avg_improvement_percent",
    "avg_parallel_cost",
    "avg_wcss_serial",
    "avg_wcss_parallel",
    "avg_iterations_serial",
    "avg_iterations_parallel",
]


def load_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas en el CSV: {missing}. Esperadas: {EXPECTED_COLUMNS}"
        )
    df = df[EXPECTED_COLUMNS].copy()
    for col in (
        "threads",
        "dimension",
        "k",
        "repetitions",
        "n_points",
    ):
        df[col] = df[col].astype(int)
    numeric = [
        "avg_time_serial",
        "avg_time_parallel",
        "avg_speedup",
        "avg_efficiency",
        "avg_improvement_percent",
        "avg_parallel_cost",
        "avg_wcss_serial",
        "avg_wcss_parallel",
        "avg_iterations_serial",
        "avg_iterations_parallel",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[numeric].isna().any().any():
        bad = df[df[numeric].isna().any(axis=1)]
        raise ValueError(f"Valores no numericos o faltantes en filas:\n{bad}")
    df["series_label"] = (
        df["dataset"].astype(str)
        + " | "
        + df["dimension"].astype(str)
        + "D | k="
        + df["k"].astype(str)
    )
    df = df.sort_values(["series_label", "threads"], kind="mergesort")
    return df


def _series_groups(df: pd.DataFrame):
    return [(label, g.copy()) for label, g in df.groupby("series_label", sort=False)]


def _color_cycle(n: int):
    cmap = plt.colormaps["tab10"]
    if n <= 0:
        return []
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_speedup(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = _series_groups(df)
    colors = _color_cycle(len(groups))
    for (label, g), color in zip(groups, colors, strict=True):
        ax.plot(
            g["threads"],
            g["avg_speedup"],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=7,
            color=color,
            label=label,
        )
    ax.set_title("Speedup frente al numero de hilos")
    ax.set_xlabel("Numero de hilos (threads)")
    ax.set_ylabel("Speedup promedio (avg_speedup)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.set_xticks(sorted(df["threads"].unique()))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = _series_groups(df)
    colors = _color_cycle(len(groups))
    for (label, g), color in zip(groups, colors, strict=True):
        ax.plot(
            g["threads"],
            g["avg_time_parallel"],
            marker="s",
            linestyle="-",
            linewidth=2,
            markersize=6,
            color=color,
            label=f"{label} (paralelo)",
        )
        # proyecto.cpp vuelve a medir serial en cada config. de hilos; suele variar levemente.
        t_serial_ref = float(g["avg_time_serial"].mean())
        ax.axhline(
            t_serial_ref,
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"{label} (serial, promedio)",
        )
    ax.set_title("Tiempo de ejecucion frente al numero de hilos")
    ax.set_xlabel("Numero de hilos (threads)")
    ax.set_ylabel("Tiempo (s)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.set_xticks(sorted(df["threads"].unique()))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = _series_groups(df)
    colors = _color_cycle(len(groups))
    for (label, g), color in zip(groups, colors, strict=True):
        ax.plot(
            g["threads"],
            g["avg_efficiency"],
            marker="^",
            linestyle="-",
            linewidth=2,
            markersize=7,
            color=color,
            label=label,
        )
    ax.set_title("Eficiencia paralela frente al numero de hilos")
    ax.set_xlabel("Numero de hilos (threads)")
    ax.set_ylabel("Eficiencia promedio (avg_efficiency)")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.set_xticks(sorted(df["threads"].unique()))
    ax.axhline(
        1.0,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label="Eficiencia ideal = 1",
        alpha=0.8,
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graficas academicas desde CSV del modo experiment de K-means."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("metricas_experimento.csv"),
        help="Ruta al CSV de metricas (modo experiment)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directorio donde guardar los PNG",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        sys.exit(f"No se encontro el archivo: {args.csv}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_metrics_csv(args.csv)

    plot_speedup(df, args.outdir / "speedup.png")
    plot_time(df, args.outdir / "tiempo.png")
    plot_efficiency(df, args.outdir / "eficiencia.png")

    print(f"Leido: {args.csv} ({len(df)} filas)")
    for name in ("speedup.png", "tiempo.png", "eficiencia.png"):
        print(f"Guardado: {args.outdir / name}")


if __name__ == "__main__":
    main()
