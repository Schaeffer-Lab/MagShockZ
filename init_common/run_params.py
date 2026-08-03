"""Shared, dimension-agnostic run-parameter helpers.

These pure-arithmetic pieces were duplicated (with matching formulas) in both
initialization entrypoints.  They are yt-free and unit-testable.

Tile-number selection is included here, resolved against the authoritative
OSIRIS CUDA docs (source/cuda/README.md).  Previously the two writers disagreed:
FLASH_OSIRIS_define used the (correct) shared-memory formula but with a
guard-cell sign bug and a broken 1D branch, while simplified_magshockz used an
unrelated n_cells/1024 heuristic.  Both are replaced by ``tile_numbers`` below.
"""

from dataclasses import dataclass

import numpy as np

# Interpolation order used in the guard-cell term of the tile-size formula.
INTERP_ORDER = {"linear": 1, "quadratic": 2, "cubic": 3, "quartic": 4}

# Max shared memory per block for NVIDIA A100 (compute capability 8.0): 163 KiB.
# See source/cuda/README.md.  Change this for a different GPU.
A100_SHMEM_BYTES = 163 * 1024


def cfl_dt(dx: float, dims: int, safety: float = 0.95) -> float:
    """CFL-limited timestep: dx * safety / sqrt(dims).

    Matches both writers (dims=1 -> dx*0.95; dims=2 -> dx*0.95/sqrt(2)).
    Returns a raw float; callers format/round it as they did before.
    """
    return dx * safety / np.sqrt(dims)


def ndump(tmax: float, dt: float, ndump_tot: int) -> int:
    """Dump cadence so that ~ndump_tot dumps are written over tmax.

    Equivalent to both prior forms: int(tmax / (ndump_tot * dt)).
    Pass whatever dt value the caller uses (raw or rounded) to preserve behavior.
    """
    return int(tmax / (ndump_tot * dt))


@dataclass
class ParticleLoad:
    n_particles: float
    n_gpus: float
    n_nodes: float


def estimate_particle_load(
    n_cells_total: float,
    ppc_per_cell: float,
    n_species: int = 3,
    bytes_per_particle: int = 70,
    overalloc: int = 2,
    mem_per_gpu: float = 40e9,
    gpu_fill: float = 0.8,
    gpus_per_node: int = 4,
) -> ParticleLoad:
    """Estimate particle count and recommended GPU/node count.

    ``ppc_per_cell`` is the total particles-per-cell across all sub-dimensions:
    e.g. ppc**dims for a scalar ppc, or product(num_par_x) for a per-dimension
    list.  ``n_cells_total`` is the product of cell counts over all dimensions.

    ~70 bytes/particle (per M. Trantham); the factor ``overalloc`` accounts for
    OSIRIS allocating headroom beyond the initial particle count.
    """
    n_particles = n_cells_total * ppc_per_cell * n_species
    n_bytes = n_particles * overalloc * bytes_per_particle
    max_bytes_per_gpu = mem_per_gpu * gpu_fill
    n_gpus = np.ceil(n_bytes / max_bytes_per_gpu)
    n_nodes = np.ceil(n_bytes / max_bytes_per_gpu / gpus_per_node)
    return ParticleLoad(n_particles=n_particles, n_gpus=n_gpus, n_nodes=n_nodes)


def osiris_quoted_list(items, suffix: str = "") -> str:
    """Format a list of diagnostic names as an OSIRIS comma-separated quoted list.

    >>> osiris_quoted_list(["charge", "j1"], ", savg")
    '"charge, savg", "j1, savg"'
    >>> osiris_quoted_list(["uth1", "ufl1"])
    '"uth1", "ufl1"'

    Used for ``reports``/``emf_reports`` (with ``suffix=", savg"``) and for
    ``rep_udist``/``phasespaces`` (no suffix).  Replaces the bespoke string
    builders that lived in each writer.
    """
    return ", ".join(f'"{item}{suffix}"' for item in items)


def max_tile_cells(
    dims: int,
    interpolation: str,
    shmem_bytes: float = A100_SHMEM_BYTES,
    precision: int = 8,
    safety_frac: float = 0.8,
) -> int:
    """Maximum tile size, in cells per dimension, that fits in CUDA shared memory.

    From source/cuda/README.md (assuming square/cubic tiles):

        n_x = (shmemsize / (2 * 3 * precision))^(1/dims) - (2 * interp + 1)

    where the 2 is for the E and B field arrays, the 3 for their components, and
    the trailing term is the guard cells.  ``precision`` is bytes per float (8 =
    double, 4 = single).  ``safety_frac`` is a conservative margin below the
    hardware shared-memory limit (the README formula itself uses the full limit;
    0.8 keeps tiles a touch smaller to avoid edge-case overflows).
    """
    if interpolation not in INTERP_ORDER:
        raise ValueError(f"Unsupported interpolation '{interpolation}'")
    interp = INTERP_ORDER[interpolation]
    usable = shmem_bytes * safety_frac
    side = (usable / (2 * 3 * precision)) ** (1.0 / dims)
    return int(side - (2 * interp + 1))


def tile_numbers(
    n_cells_per_dim,
    dims: int,
    interpolation: str,
    shmem_bytes: float = A100_SHMEM_BYTES,
    precision: int = 8,
    safety_frac: float = 0.8,
):
    """Number of tiles per dimension: smallest power of two keeping every tile
    within the shared-memory cap (i.e. the largest allowed tiles, fewest tiles).

    Parameters
    ----------
    n_cells_per_dim : sequence of int
        Cell count along each simulation dimension (length must equal ``dims``).

    Returns
    -------
    list[int]
        Power-of-two tile count for each dimension.
    """
    if len(n_cells_per_dim) != dims:
        raise ValueError(f"expected {dims} cell counts, got {len(n_cells_per_dim)}")
    cap = max_tile_cells(dims, interpolation, shmem_bytes, precision, safety_frac)
    if cap < 1:
        raise ValueError(
            f"shared-memory cap gives max tile size {cap} < 1 cell "
            f"(dims={dims}, interpolation={interpolation}, precision={precision})"
        )
    tiles = []
    for n_cells in n_cells_per_dim:
        i = 0
        while n_cells / 2**i > cap:
            i += 1
        tiles.append(2**i)
    return tiles
