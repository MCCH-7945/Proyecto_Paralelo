#!/usr/bin/env python3
"""
Genera los 14 CSV sinteticos para experimentos de rendimiento con proyecto.cpp.

Formato: sin encabezado, coma, una fila por punto (x,y o x,y,z).
Nombres: {n}_data_2d.csv y {n}_data_3d.csv para n en SIZES.

Dependencia: scikit-learn, numpy
  pip install scikit-learn numpy

Recursos: son 14 archivos (no 1M archivos); el mas grande tiene 1e6 filas.
  - Disco total tipico del orden de **~150–250 MiB** en texto CSV (depende de los numeros).
  - Tiempo: varios minutos en laptop normal; el paso de 1M puntos es el mas costoso.
  - Usa `--dry-run` para ver el plan y la estimacion sin escribir disco.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Tamaños de muestra (requisito del experimento)
SIZES: tuple[int, ...] = (
    100_000,
    200_000,
    300_000,
    400_000,
    600_000,
    800_000,
    1_000_000,
)

# Hiperparámetros de blobs (alineados con synthetic_clusters_kmeans.ipynb)
N_CLUSTERS = 8
CLUSTER_STD = 0.04
CENTER_BOX = (0.0, 1.0)
BASE_RANDOM_STATE = 7

# Estimacion conservadora de bytes por fila en CSV (coordenadas con 3 decimales + comas + \\n)
_BYTES_PER_ROW_2D = 24
_BYTES_PER_ROW_3D = 36


def estimate_batch_disk_bytes() -> int:
    """Suma aproximada del tamano de los 14 CSV en disco."""
    total = 0
    for n in SIZES:
        total += n * _BYTES_PER_ROW_2D
        total += n * _BYTES_PER_ROW_3D
    return total


def _human_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def print_resource_notice(out_dir: Path, *, dry_run: bool) -> None:
    n_files = 2 * len(SIZES)
    approx = estimate_batch_disk_bytes()
    total_points_2d = sum(SIZES)
    total_points_3d = sum(SIZES)
    print("---", flush=True)
    print("AVISO: generacion de datasets grandes", flush=True)
    print(
        f"  Archivos: {n_files} (7 tamanos x 2D y 3D), en: {out_dir.resolve()}",
        flush=True,
    )
    print(
        f"  Puntos en total: ~{total_points_2d + total_points_3d:,} filas CSV "
        f"({total_points_2d:,} en 2D + {total_points_3d:,} en 3D)",
        flush=True,
    )
    print(
        f"  Disco aproximado (CSV): ~{_human_mib(approx)} (orden de magnitud; puede variar)",
        flush=True,
    )
    print(
        "  Tiempo: varios minutos; 1M filas por archivo es el paso mas pesado.",
        flush=True,
    )
    if dry_run:
        print("  Modo --dry-run: no se escribira ningun archivo.", flush=True)
    print("---\n", flush=True)


def build_filename(n_points: int, n_features: int) -> str:
    dim = "2d" if n_features == 2 else "3d"
    return f"{n_points}_data_{dim}.csv"


def generate_clustered_points(
    n_samples: int,
    n_features: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Puntos bien agrupados en N_CLUSTERS blobs; reproducibles con random_state."""
    from sklearn.datasets import make_blobs

    points, _ = make_blobs(
        n_samples=n_samples,
        centers=N_CLUSTERS,
        n_features=n_features,
        cluster_std=CLUSTER_STD,
        random_state=random_state,
        center_box=CENTER_BOX,
    )
    # Mismo post-proceso que el notebook: valores no negativos y 3 decimales
    points = np.round(np.abs(points), 3)
    if points.shape[0] != n_samples:
        raise RuntimeError("make_blobs no devolvio n_samples filas")
    return points


def dataset_seed(n_samples: int, n_features: int) -> int:
    """Semilla distinta por (tamaño, dim) pero fija entre ejecuciones."""
    return BASE_RANDOM_STATE + n_features * 1_000_003 + (n_samples % 1_009_981)


def write_csv_no_header(path: Path, points: np.ndarray) -> None:
    np.savetxt(path, points, delimiter=",", fmt="%.3f")


def generate_all(out_dir: Path, *, dry_run: bool = False) -> None:
    out_dir = out_dir.resolve()
    print_resource_notice(out_dir, dry_run=dry_run)
    if dry_run:
        for n in SIZES:
            for dim in (2, 3):
                name = build_filename(n, dim)
                rows = n
                br = rows * (_BYTES_PER_ROW_2D if dim == 2 else _BYTES_PER_ROW_3D)
                print(
                    f"[dry-run] {name}  ~{_human_mib(br)}  (n={rows}, dim={dim})",
                    flush=True,
                )
        print("\nDry-run terminado. Quita --dry-run para generar los CSV.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for n in SIZES:
        for dim in (2, 3):
            name = build_filename(n, dim)
            dest = out_dir / name
            seed = dataset_seed(n, dim)
            print(f"[generar] {name}  (n={n}, dim={dim}, seed={seed}) ...", flush=True)
            pts = generate_clustered_points(n, dim, random_state=seed)
            write_csv_no_header(dest, pts)
            print(f"[ok]     guardado: {dest}  ({pts.shape[0]} filas x {pts.shape[1]} cols)", flush=True)

    expected = 2 * len(SIZES)
    print(f"\nListo: {expected} archivos en {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera CSV 2D/3D para experimentos K-means (proyecto.cpp)."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Carpeta de salida (por defecto: directorio actual)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra plan, estimacion de disco y tamanos; no crea CSV",
    )
    args = parser.parse_args()
    try:
        generate_all(args.outdir, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
