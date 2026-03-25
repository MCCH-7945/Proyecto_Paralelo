#!/usr/bin/env python3
"""
Visualiza salidas CSV de proyecto.cpp (modo serial y paralelo).
Formato esperado por fila: x[, y[, z]], cluster (sin encabezado).

Por que a veces hay 100% de coincidencia fila-a-fila: los dos CSV deben ser
salidas del mismo dataset, mismo k/semilla e iteraciones comparables; el
paralelo (OpenMP) sigue la misma logica y a menudo reproduce exactamente las
mismas asignaciones. Si hubiera diferencias (p. ej. otra semilla, distinto
orden numerico o runs independientes), aparecerian celdas fuera de la diagonal
en la matriz de confusion y el mapa espacial de discordancias.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _tab10():
    return plt.colormaps["tab10"]


def load_assignments_csv(path: Path) -> tuple[pd.DataFrame, int]:
    """Carga CSV; ultima columna es cluster."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{path}: se necesitan al menos 2 columnas (coords + cluster).")
    dim = df.shape[1] - 1
    df = df.rename(columns={dim: "cluster"})
    df["cluster"] = df["cluster"].astype(int)
    return df, dim


def subsample(df: pd.DataFrame, dim: int, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def plot_2d_comparison(
    serial: pd.DataFrame,
    parallel: pd.DataFrame,
    dim: int,
    max_points: int,
    out_path: Path | None,
) -> None:
    if dim != 2:
        raise ValueError("plot_2d_comparison solo aplica a dim=2.")

    s = subsample(serial, dim, max_points)
    p = subsample(parallel, dim, max_points)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharex=True, sharey=True, layout="constrained"
    )
    cmap = _tab10()
    vmax = max(9, int(max(s["cluster"].max(), p["cluster"].max())))

    for ax, df, title in (
        (axes[0], s, "Serial"),
        (axes[1], p, "Paralelo (OpenMP)"),
    ):
        clusters = df["cluster"].to_numpy()
        x = df[0].to_numpy()
        y = df[1].to_numpy()
        sc = ax.scatter(
            x,
            y,
            c=clusters,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            s=4,
            alpha=0.35,
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes, label="cluster", shrink=0.85)
    fig.suptitle("K-Means: asignaciones (muestra si hay muchos puntos)")

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def rowwise_agreement_stats(serial: pd.DataFrame, parallel: pd.DataFrame) -> tuple[int, int, np.ndarray]:
    """Retorna (n_coincidentes, n_distintos, mask_diff). Requiere mismo orden de filas."""
    if len(serial) != len(parallel):
        raise ValueError("CSV con distinto numero de filas: no hay comparacion fila-a-fila valida.")
    s_lab = serial["cluster"].to_numpy()
    p_lab = parallel["cluster"].to_numpy()
    same = s_lab == p_lab
    n_diff = int((~same).sum())
    return int(same.sum()), n_diff, ~same


def plot_comparison_labels(
    serial: pd.DataFrame,
    parallel: pd.DataFrame,
    out_path: Path | None,
) -> None:
    """
    Matriz de confusion fila-a-fila: filas=cluster serial, columnas=cluster paralelo.
    Con acuerdo total solo hay diagonal; con desacuerdos aparecen celdas fuera de la diagonal.
    """
    s_lab = serial["cluster"].to_numpy()
    p_lab = parallel["cluster"].to_numpy()
    k = int(max(s_lab.max(), p_lab.max())) + 1
    cm = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(s_lab, p_lab, strict=True):
        cm[a, b] += 1

    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    im = ax_cm.imshow(cm, aspect="auto", cmap="Blues")
    ax_cm.set_title("Matriz serial (filas) vs paralelo (columnas)")
    ax_cm.set_xlabel("cluster paralelo")
    ax_cm.set_ylabel("cluster serial")
    ax_cm.set_xticks(range(k))
    ax_cm.set_yticks(range(k))
    for i in range(k):
        for j in range(k):
            if cm[i, j] > 0:
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax_cm, label="conteo de puntos")

    idx = range(k)
    counts_s = serial["cluster"].value_counts().reindex(idx, fill_value=0).to_numpy()
    counts_p = parallel["cluster"].value_counts().reindex(idx, fill_value=0).to_numpy()
    w = 0.35
    ax_bar.bar([i - w / 2 for i in idx], counts_s, width=w, label="serial")
    ax_bar.bar([i + w / 2 for i in idx], counts_p, width=w, label="paralelo")
    ax_bar.set_xticks(list(idx))
    ax_bar.set_xlabel("cluster")
    ax_bar.set_ylabel("numero de puntos")
    ax_bar.set_title("Tamanos de cluster")
    ax_bar.legend()

    if out_path:
        p = out_path.with_stem(out_path.stem + "_comparacion_etiquetas")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {p}")
    else:
        plt.show()
    plt.close(fig)


def plot_spatial_mismatches_2d(
    serial: pd.DataFrame,
    parallel: pd.DataFrame,
    diff_mask: np.ndarray,
    max_points: int,
    out_path: Path | None,
) -> None:
    """Solo puntos donde serial y paralelo difieren; color = etiqueta serial, borde = paralelo."""
    idx = np.where(diff_mask)[0]
    if len(idx) == 0:
        return
    if len(idx) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, size=max_points, replace=False)

    x = serial.loc[idx, 0].to_numpy()
    y = serial.loc[idx, 1].to_numpy()
    c_s = serial.loc[idx, "cluster"].to_numpy()
    c_p = parallel.loc[idx, "cluster"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    cmap_tab = _tab10()
    vmax_e = max(9, int(max(c_s.max(), c_p.max())))
    norm_e = plt.Normalize(vmin=0, vmax=vmax_e)
    edge_rgba = cmap_tab(norm_e(c_p))
    sc = ax.scatter(
        x,
        y,
        c=c_s,
        cmap=cmap_tab,
        vmin=0,
        vmax=vmax_e,
        s=36,
        edgecolors=edge_rgba,
        linewidths=0.8,
        alpha=0.85,
        rasterized=True,
    )
    ax.set_title(
        f"Puntos con etiqueta distinta (relleno=serial, borde≈paralelo; muestra {len(idx)})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="cluster serial")
    if out_path:
        p = out_path.with_stem(out_path.stem + "_diferencias_espaciales")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {p}")
    else:
        plt.show()
    plt.close(fig)


def print_and_plot_agreement(
    serial: pd.DataFrame,
    parallel: pd.DataFrame,
    dim: int,
    max_points: int,
    out_path: Path | None,
) -> None:
    """Imprime estadisticas; si hay discordancias y es 2D, grafica posiciones discordantes."""
    try:
        n_ok, n_diff, diff_mask = rowwise_agreement_stats(serial, parallel)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return

    n = len(serial)
    pct_ok = 100.0 * n_ok / n if n else 0.0
    pct_diff = 100.0 * n_diff / n if n else 0.0
    print(f"Coincidencia de cluster (misma fila): {n_ok} / {n} ({pct_ok:.4f} %)")
    print(f"Filas con etiqueta distinta: {n_diff} ({pct_diff:.4f} %)")

    plot_comparison_labels(serial, parallel, out_path)

    if dim != 2:
        if n_diff > 0:
            print("Mapa espacial de diferencias solo disponible en 2D.")
        return

    if n_diff == 0:
        print(
            "No hay discordancias fila-a-fila: el mapa espacial de 'diferencias' estaria vacio. "
            "La matriz de confusion muestra solo la diagonal (acuerdo total)."
        )
        return

    plot_spatial_mismatches_2d(serial, parallel, diff_mask, max_points, out_path)


def plot_3d_panel(
    serial: pd.DataFrame,
    parallel: pd.DataFrame,
    dim: int,
    max_points: int,
    out_path: Path | None,
) -> None:
    if dim != 3:
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    s = subsample(serial, dim, max_points)
    p = subsample(parallel, dim, max_points)
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    cmap = _tab10()
    vmax = max(9, int(max(s["cluster"].max(), p["cluster"].max())))

    for i, (df, title) in enumerate(((s, "Serial"), (p, "Paralelo (OpenMP)")), start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        clusters = df["cluster"].to_numpy()
        sc = ax.scatter(
            df[0],
            df[1],
            df[2],
            c=clusters,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            s=2,
            alpha=0.35,
            depthshade=False,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.colorbar(sc, ax=fig.axes, label="cluster", shrink=0.6, pad=0.1)
    fig.suptitle("K-Means 3D: asignaciones (muestra)")

    if out_path:
        p3 = out_path.with_stem(out_path.stem + "_3d")
        fig.savefig(p3, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {p3}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizar CSV serial vs paralelo de K-Means.")
    parser.add_argument(
        "--serial",
        type=Path,
        default=Path("salida_run_test.csv"),
        help="CSV salida modo serial",
    )
    parser.add_argument(
        "--paralelo",
        type=Path,
        default=Path("salida_parallel_run.csv"),
        help="CSV salida modo paralelo",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("visualizacion_kmeans_serial_vs_paralelo.png"),
        help="Ruta base para PNG(s) de salida",
    )
    parser.add_argument(
        "--max-puntos",
        type=int,
        default=50_000,
        help="Max puntos a dibujar por panel (muestreo aleatorio si hay mas)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar ventanas en lugar de solo guardar",
    )
    args = parser.parse_args()

    if not args.serial.is_file():
        sys.exit(f"No existe: {args.serial}")
    if not args.paralelo.is_file():
        sys.exit(f"No existe: {args.paralelo}")

    s_df, dim_s = load_assignments_csv(args.serial)
    p_df, dim_p = load_assignments_csv(args.paralelo)
    if dim_s != dim_p:
        sys.exit(f"Dimension distinta: serial={dim_s}, paralelo={dim_p}")

    dim = dim_s
    print(f"Dimension: {dim}D | puntos serial: {len(s_df)} | puntos paralelo: {len(p_df)}")

    out = None if args.show else args.out

    if dim == 2:
        plot_2d_comparison(s_df, p_df, dim, args.max_puntos, out)
        print_and_plot_agreement(
            s_df, p_df, dim, min(args.max_puntos, 10_000), out
        )
    elif dim == 3:
        plot_3d_panel(s_df, p_df, dim, args.max_puntos, out)
        print_and_plot_agreement(
            s_df, p_df, dim, min(args.max_puntos, 10_000), out
        )
    else:
        print("Solo se grafican 2D y 3D; resumen numerico:")
        print(s_df["cluster"].value_counts().sort_index())
        print(p_df["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
