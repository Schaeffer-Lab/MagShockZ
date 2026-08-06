#!/bin/bash -l
# Build the 2D CUDA WarpX *app* that init_warpx/run_heater_2d.sbatch runs.
#
# WHY THIS EXISTS. ParticleHeater + TargetInjector live on the lab fork's
# `feature/particle-heater` branch, and every pre-existing GPU build in that tree
# predates those commits (build_pm_gpu_py_2d/bin/ has no binary at all, and the older
# build_pm_gpu/ binaries contain no `particle_heater` symbol). A GPU run therefore needs
# its own build; the CPU 2D app is the only pre-built binary that has the operators.
#
# APP, NOT PYTHON. The heater/injector are ParmParse-only — no PICMI binding — so the
# deck is a text file and the python bindings buy nothing. -DWarpX_PYTHON=OFF with
# -DWarpX_APP=ON is both faster to build and avoids the pybind11+NVCC toolchain pin.
#
# QED is off (unused here) and openPMD is left at its default: the analysis reads AMReX
# plotfiles via yt either way.
#
# Usage (login node is fine; ~15-30 min with ccache cold):
#     bash init_warpx/build_warpx_gpu_2d.sh
# Then check the binary carries the operator:
#     strings /pscratch/sd/d/dschnei/warpx/build_pm_gpu_2d/bin/warpx.2d | grep particle_heater
set -euo pipefail

WARPX_DIR=${WARPX_DIR:-/pscratch/sd/d/dschnei/warpx}
BUILD_DIR=${BUILD_DIR:-$WARPX_DIR/build_pm_gpu_2d}
JOBS=${JOBS:-16}

cd "$WARPX_DIR"

# Two builds sharing one build dir interleave their object writes and relinks, and the
# verification below then inspects whichever binary happened to be linked last -- which
# is how a good build got reported as missing the operator. Refuse to start instead.
if pgrep -f "cmake --build $BUILD_DIR" >/dev/null; then
  echo "ERROR: another 'cmake --build $BUILD_DIR' is already running."
  echo "       Wait for it (or kill it) rather than racing it in the same build dir."
  exit 1
fi

source "$HOME/perlmutter_gpu_warpx.profile"

branch=$(git rev-parse --abbrev-ref HEAD)
echo "=== warpx $WARPX_DIR @ $branch ($(git rev-parse --short HEAD)) ==="
if ! git ls-files --error-unmatch Source/Particles/ParticleHeater/ParticleHeater.cpp >/dev/null 2>&1; then
  echo "ERROR: ParticleHeater sources are not in this checkout. Expected the"
  echo "       feature/particle-heater branch; got '$branch'."
  exit 1
fi
echo "=== nvcc: $(which nvcc)   CUDA arch: ${AMREX_CUDA_ARCH:-unset} ==="

cmake -S . -B "$BUILD_DIR" \
  -DWarpX_DIMS=2 \
  -DWarpX_COMPUTE=CUDA \
  -DWarpX_APP=ON \
  -DWarpX_PYTHON=OFF \
  -DWarpX_QED=OFF \
  -DCMAKE_BUILD_TYPE=Release
echo "=== CONFIG DONE ==="

cmake --build "$BUILD_DIR" -j "$JOBS" --target app_2d
echo "=== BUILD DONE ==="

# -type f skips the warpx.2d symlink this script creates, which would otherwise be
# picked up ahead of the real binary on a rebuild and verified in its place.
exe=$(find "$BUILD_DIR/bin" -maxdepth 1 -type f -name 'warpx.2d*' ! -name '*.so' \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [[ -z "$exe" ]]; then
  echo "ERROR: no warpx.2d* binary under $BUILD_DIR/bin"
  exit 1
fi
# A binary without the operator runs the deck happily and silently produces a
# physics-free result, so fail loudly here rather than after burning GPU hours.
#
# grep -c, not grep -q: with `set -o pipefail` a short-circuiting `grep -q` kills
# `strings` with SIGPIPE partway through a 594 MB binary, and pipefail then reports the
# whole pipeline as failed -- i.e. a perfectly good build looks like it lacks the
# operator. `grep -c` reads to EOF, and `|| true` absorbs its exit-1-on-zero-matches.
heater_hits=$(strings "$exe" | grep -c particle_heater || true)
if [[ "${heater_hits:-0}" -eq 0 ]]; then
  echo "ERROR: $exe contains no 'particle_heater' symbol."
  exit 1
fi
echo "=== operator present: $heater_hits 'particle_heater' strings ==="
ln -sfn "$(basename "$exe")" "$BUILD_DIR/bin/warpx.2d"
echo "=== GPU_2D_APP_BUILD_SUCCESS: $exe ==="
echo "run_heater_2d.sbatch picks this up as $BUILD_DIR/bin/warpx.2d"
