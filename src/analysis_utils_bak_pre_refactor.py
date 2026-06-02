from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import fnmatch
import re

import astropy
import h5py
import numpy as np
import osiris_utils
import osh5def
import osh5vis
import plasmapy
import scipy.integrate


@dataclass
class AxisDescriptor:
    name: str
    grid: np.ndarray
    units_hint: str = "normalized"


@dataclass
class DiagnosticDescriptor:
    key: str
    kind: str
    ndim: int
    axes: List[AxisDescriptor]
    shape: Tuple[int, ...]
    source: Any
    label: str
    units: str = ""


class MomentDiagnostic:
    """Diagnostic adapter with osiris-like indexing for lazy phase-space moments."""

    def __init__(
        self,
        run: "MagShockZRun",
        species: str,
        momentum_component: str,
        order: int,
        cache: bool = True,
    ):
        self.run = run
        self.species = species
        self.momentum_component = momentum_component
        self.order = order
        self.cache = cache

        moment_name = self.run._moment_name(order, momentum_component)
        self.name = f"{species}/{moment_name}-from-{momentum_component}"

        phase_key = self.run._resolve_phase_field(species, momentum_component)
        self.phase = self.run._get_native_field(phase_key)

        p_axis_idx = self.run._find_momentum_axis(self.phase, momentum_component)
        phase_grids = self.phase.grid if isinstance(self.phase.grid, list) else [self.phase.grid]
        phase_nx = self.phase.nx if isinstance(self.phase.nx, list) else [self.phase.nx]

        self.grid = [phase_grids[i] for i in range(len(phase_grids)) if i != p_axis_idx]
        self.nx = [phase_nx[i] for i in range(len(phase_nx)) if i != p_axis_idx]

        axis_meta = getattr(self.phase, "axis", None)
        if axis_meta is None:
            self.axis = []
        else:
            self.axis = [axis_meta[i] for i in range(len(axis_meta)) if i != p_axis_idx]

    def __getitem__(self, timestep: int) -> np.ndarray:
        return self.run.calculate_moment(
            self.species,
            timestep=timestep,
            order=self.order,
            momentum_component=self.momentum_component,
            cache=self.cache,
        )

    def __len__(self) -> int:
        return len(self.phase)

    def time(self, timestep: int):
        return self.phase.time(timestep)


class DiagnosticIndex:
    """Discover and store diagnostics with normalized metadata for 1D/2D plotting."""

    def __init__(self, run: "MagShockZRun"):
        self.run = run
        self._registry: Dict[str, DiagnosticDescriptor] = {}

    def build(self, refresh: bool = False) -> Dict[str, DiagnosticDescriptor]:
        if self._registry and not refresh:
            return self._registry

        self._registry = {}

        for key in self.run._discover_native_keys():
            try:
                field = self.run._get_native_field(key)
                self._registry[key] = self._make_descriptor(key, field, self._infer_kind(key))
            except Exception:
                # Ignore broken or non-standard entries while indexing.
                continue

        for key, moment in self.run._moment_diags.items():
            self._registry[key] = self._make_descriptor(key, moment, "moment")

        return self._registry

    def register_moment(self, key: str, moment_obj: MomentDiagnostic) -> None:
        self._registry[key] = self._make_descriptor(key, moment_obj, "moment")

    def get(self, key: str) -> DiagnosticDescriptor:
        self.build(refresh=False)
        if key in self._registry:
            return self._registry[key]

        # Fallback for paths that were not discovered by index() but are valid.
        if key in self.run._moment_diags:
            field = self.run._moment_diags[key]
        else:
            field = self.run._get_native_field(key)
        kind = "moment" if key in self.run._moment_diags else self._infer_kind(key)
        descriptor = self._make_descriptor(key, field, kind)
        self._registry[key] = descriptor
        return descriptor

    def list(
        self,
        pattern: Optional[str] = None,
        kind: Optional[str] = None,
        ndim: Optional[int] = None,
    ) -> List[str]:
        self.build(refresh=False)
        keys = []
        for key, descriptor in self._registry.items():
            if pattern and not fnmatch.fnmatch(key, pattern):
                continue
            if kind and descriptor.kind != kind:
                continue
            if ndim is not None and descriptor.ndim != ndim:
                continue
            keys.append(key)
        return sorted(keys)

    def _infer_kind(self, key: str) -> str:
        lower = key.lower()
        if "-from-p" in lower:
            return "moment"
        if "p1" in lower or "p2" in lower or "p3" in lower:
            return "phase"
        if "dens" in lower or "charge" in lower:
            return "density"
        return "field"

    def _make_descriptor(self, key: str, field: Any, kind: str) -> DiagnosticDescriptor:
        sample = np.asarray(field[0])
        squeezed = np.squeeze(sample)
        if squeezed.ndim == 0:
            squeezed = squeezed.reshape(1)
        ndim = squeezed.ndim
        shape = tuple(squeezed.shape)

        coords, labels = self.run._coords_for(field, timestep=0, spatial_units="cells")
        axes = [AxisDescriptor(name=label, grid=np.asarray(coord), units_hint="cells") for coord, label in zip(coords, labels)]

        return DiagnosticDescriptor(
            key=key,
            kind=kind,
            ndim=ndim,
            axes=axes,
            shape=shape,
            source=field,
            label=key,
            units="",
        )


class Plotter:
    """Unified plotting entry point for line/image/lineout/streak modes."""

    def __init__(self, run: "MagShockZRun"):
        self.run = run

    def plot(
        self,
        name: str,
        t: int = 0,
        mode: str = "auto",
        ax=None,
        spatial_units: str = "ion",
        time_units: str = "ion gyrotime",
        axis: str = "x",
        position: Optional[float] = None,
        slab: Optional[Tuple[float, float]] = None,
        timesteps: Optional[Sequence[int]] = None,
        slice_pos: Optional[Dict[str, float]] = None,
        log: bool = False,
        norm=None,
        vmin=None,
        vmax=None,
        cmap: str = "RdBu_r",
        colorbar: bool = True,
        **kwargs,
    ):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        descriptor = self.run.diags.get(name)
        quantity_label = self.run._quantity_label(name)

        if mode == "auto":
            mode = "line" if descriptor.ndim == 1 else "image"

        if mode == "line":
            data, coords, labels, time_label, time_value = self._data_coords_time(
                descriptor.source, t, spatial_units, time_units
            )
            if data.ndim > 1:
                return self.plot(
                    name,
                    t=t,
                    mode="lineout",
                    ax=ax,
                    spatial_units=spatial_units,
                    time_units=time_units,
                    axis=axis,
                    position=position,
                    slab=slab,
                    slice_pos=slice_pos,
                    **kwargs,
                )

            if ax is None:
                _, ax = plt.subplots(figsize=(8, 5))

            ax.plot(coords[0], data, **kwargs)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(quantity_label)
            ax.set_title(f"{name} at t = {np.round(time_value, 3)} {time_label}")
            ax.grid(alpha=0.3)
            if log:
                ax.set_yscale("log")
            if vmin is not None or vmax is not None:
                ax.set_ylim(vmin, vmax)
            return ax

        if mode == "image":
            data, coords, labels, time_label, time_value = self._data_coords_time(
                descriptor.source, t, spatial_units, time_units
            )
            if data.ndim == 1:
                return self.plot(
                    name,
                    t=t,
                    mode="line",
                    ax=ax,
                    spatial_units=spatial_units,
                    time_units=time_units,
                    log=log,
                    vmin=vmin,
                    vmax=vmax,
                    **kwargs,
                )

            if data.ndim >= 3:
                data, coords, labels, title_extra = self._slice_to_2d(data, coords, labels, slice_pos)
            else:
                title_extra = ""

            if norm is None:
                if log:
                    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                else:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            if ax is None:
                _, ax = plt.subplots(figsize=(7, 5))

            im = ax.imshow(
                data.T,
                origin="lower",
                extent=[coords[0][0], coords[0][-1], coords[1][0], coords[1][-1]],
                norm=norm,
                cmap=cmap,
                aspect="auto",
                **kwargs,
            )
            if colorbar:
                plt.colorbar(im, ax=ax, label=quantity_label)

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_title(f"{name} at t = {np.round(time_value, 3)} {time_label}{title_extra}")
            return ax

        if mode == "lineout":
            field = descriptor.source
            data = np.squeeze(np.asarray(field[t]))
            coords, labels = self.run._coords_for(field, timestep=t, spatial_units=spatial_units)
            time_raw, _ = field.time(t)
            time_value, time_label = self.run._convert_time(time_raw, time_units)

            if data.ndim == 1:
                lineout = data
                lineout_coord = coords[0]
                lineout_label = labels[0]
                transverse_desc = "1D"
            else:
                if data.ndim >= 3:
                    data, coords, labels, slice_desc = self._slice_to_2d(data, coords, labels, slice_pos)
                else:
                    slice_desc = ""

                lineout, lineout_coord, lineout_label, transverse_desc = self._extract_lineout(
                    data=data,
                    coords=coords,
                    labels=labels,
                    axis=axis,
                    position=position,
                    slab=slab,
                )
                if slice_desc:
                    transverse_desc = f"{slice_desc.lstrip(', ')}, {transverse_desc}"

            if ax is None:
                _, ax = plt.subplots(figsize=(8, 5))
            ax.plot(lineout_coord, lineout, **kwargs)
            ax.set_xlabel(lineout_label)
            ax.set_ylabel(quantity_label)
            ax.set_title(f"{name} lineout at t = {np.round(time_value, 3)} {time_label}\n{transverse_desc}")
            ax.grid(alpha=0.3)
            return ax

        if mode == "streak":
            streak_data, transverse_desc = self.make_streak_h5data(
                name,
                timesteps=timesteps,
                spatial_units=spatial_units,
                time_units=time_units,
                axis=axis,
                position=position,
                slab=slab,
                slice_pos=slice_pos,
            )

            if ax is not None:
                plt.sca(ax)
            else:
                _, ax = plt.subplots(figsize=(10, 6))

            osh5vis.osplot(streak_data, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=colorbar, **kwargs)
            ax.set_title(f"{name} streak plot\n{transverse_desc}")
            return ax

        if mode == "phase":
            return self.plot(
                name,
                t=t,
                mode="image",
                ax=ax,
                spatial_units=spatial_units,
                time_units=time_units,
                slice_pos=slice_pos,
                log=log,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                colorbar=colorbar,
                **kwargs,
            )

        raise ValueError("mode must be one of: auto, line, image, lineout, streak, phase")

    def _resolve_timesteps(self, field_obj: Any, timesteps: Optional[Sequence[int]]) -> List[int]:
        n_times_available = len(field_obj)
        if timesteps is None:
            return list(range(n_times_available))
        if isinstance(timesteps, tuple) and len(timesteps) == 2:
            return list(range(timesteps[0], min(timesteps[1], n_times_available)))
        return [int(t) for t in timesteps]

    def make_streak_h5data(
        self,
        name: str,
        timesteps: Optional[Sequence[int]] = None,
        spatial_units: str = "ion",
        time_units: str = "ion gyrotime",
        axis: str = "x",
        position: Optional[float] = None,
        slab: Optional[Tuple[float, float]] = None,
        slice_pos: Optional[Dict[str, float]] = None,
    ) -> Tuple[osh5def.H5Data, str]:
        field = self.run._get_field(name)
        timestep_list = self._resolve_timesteps(field, timesteps)

        profiles = []
        converted_times = []
        spatial_coords = None
        spatial_label = ""
        transverse_desc = ""

        for timestep in timestep_list:
            data = np.squeeze(np.asarray(field[timestep]))
            coords, labels = self.run._coords_for(field, timestep=timestep, spatial_units=spatial_units)

            if data.ndim == 1:
                profile = data
                profile_coord = coords[0]
                profile_label = labels[0]
                this_desc = "1D"
            else:
                if data.ndim >= 3:
                    data, coords, labels, slice_desc = self._slice_to_2d(data, coords, labels, slice_pos)
                else:
                    slice_desc = ""
                profile, profile_coord, profile_label, this_desc = self._extract_lineout(
                    data=data,
                    coords=coords,
                    labels=labels,
                    axis=axis,
                    position=position,
                    slab=slab,
                )
                if slice_desc:
                    this_desc = f"{slice_desc.lstrip(', ')}, {this_desc}"

            profiles.append(np.asarray(profile))
            spatial_coords = np.asarray(profile_coord)
            spatial_label = profile_label
            transverse_desc = this_desc

            time_raw, _ = field.time(timestep)
            t_conv, _ = self.run._convert_time(time_raw, time_units)
            converted_times.append(t_conv)

        if spatial_coords is None:
            raise ValueError(f"No data found for streak plot '{name}'")

        streak_data = np.stack(profiles, axis=0)
        time_values = np.asarray(converted_times, dtype=float)

        time_axis = osh5def.DataAxis(
            float(np.min(time_values)),
            float(np.max(time_values)),
            len(time_values),
            attrs={"NAME": "time", "LONG_NAME": "time", "UNITS": self._time_unit_label(time_units)},
        )
        space_axis = osh5def.DataAxis(
            float(spatial_coords[0]),
            float(spatial_coords[-1]),
            len(spatial_coords),
            attrs={
                "NAME": self._axis_name(axis, profile_label=spatial_label),
                "LONG_NAME": self._axis_name(axis, profile_label=spatial_label),
                "UNITS": self._spatial_unit_label(spatial_units),
            },
        )

        data_attrs = {
            "NAME": self._sanitize_label_text(name, default="data"),
            "LONG_NAME": self._sanitize_label_text(f"{name} streak plot", default="streak data"),
            "UNITS": self._sanitize_units(getattr(field, "data_attrs", {}).get("UNITS", "")),
        }
        run_attrs = dict(getattr(field, "run_attrs", {}))
        run_attrs["TIME"] = [float(time_values[0])]

        return osh5def.H5Data(streak_data, data_attrs=data_attrs, run_attrs=run_attrs, axes=[time_axis, space_axis]), transverse_desc

    def _spatial_unit_label(self, spatial_units: str) -> str:
        if spatial_units in ("ion", "ion inertial length"):
            return "c/omega_pi"
        if spatial_units in ("electron", "electron inertial length"):
            return "c/omega_pe"
        if spatial_units == "physical":
            return "cm"
        if spatial_units == "cells":
            return "cell"
        return spatial_units

    def _time_unit_label(self, time_units: str) -> str:
        if time_units in ("ion gyrotime", "1 / omega_ci", "1/omega_ci", "omega_ci^-1"):
            return "omega_ci^-1"
        if time_units == "electron":
            return "omega_pe^-1"
        if time_units == "physical":
            return "s"
        return time_units

    def _axis_name(self, axis: str, profile_label: str = "") -> str:
        axis_lower = axis.lower().strip()
        if axis_lower in ("x", "x1", "y", "x2", "z", "x3"):
            return axis_lower[0]
        if profile_label:
            return profile_label.split()[0].strip("$")
        return "x"

    def _sanitize_label_text(self, text: Any, default: str = "") -> str:
        if text is None:
            return default
        cleaned = str(text).replace("$", "").strip()
        return cleaned if cleaned else default

    def _sanitize_units(self, units: Any) -> str:
        cleaned = self._sanitize_label_text(units, default="")
        return cleaned if cleaned else "arb."

    def _data_coords_time(
        self,
        field_obj: Any,
        timestep: int,
        spatial_units: str,
        time_units: str,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[str], str, float]:
        data = np.squeeze(np.asarray(field_obj[timestep]))
        coords, labels = self.run._coords_for(field_obj, timestep=timestep, spatial_units=spatial_units)
        time_raw, _ = field_obj.time(timestep)
        time_value, time_label = self.run._convert_time(time_raw, time_units)
        return data, coords, labels, time_label, time_value

    def _slice_to_2d(
        self,
        data: np.ndarray,
        coords: List[np.ndarray],
        labels: List[str],
        slice_pos: Optional[Dict[str, float]],
    ) -> Tuple[np.ndarray, List[np.ndarray], List[str], str]:
        axis_map = {"x": 0, "y": 1, "z": 2}
        if slice_pos is None:
            slice_axis = 0
            slice_idx = data.shape[0] // 2
        else:
            slice_axis = axis_map.get(slice_pos.get("axis", "x").lower(), 0)
            target = slice_pos.get("value", coords[slice_axis][len(coords[slice_axis]) // 2])
            slice_idx = int(np.argmin(np.abs(coords[slice_axis] - target)))

        if slice_axis == 0:
            slice_data = data[slice_idx, :, :]
            slice_coords = [coords[1], coords[2]]
            slice_labels = [labels[1], labels[2]]
        elif slice_axis == 1:
            slice_data = data[:, slice_idx, :]
            slice_coords = [coords[0], coords[2]]
            slice_labels = [labels[0], labels[2]]
        else:
            slice_data = data[:, :, slice_idx]
            slice_coords = [coords[0], coords[1]]
            slice_labels = [labels[0], labels[1]]

        axis_name = ["x", "y", "z"][slice_axis]
        axis_value = coords[slice_axis][slice_idx]
        title_extra = f", {axis_name}={axis_value:.3f}"
        return slice_data, slice_coords, slice_labels, title_extra

    def _extract_lineout(
        self,
        data: np.ndarray,
        coords: List[np.ndarray],
        labels: List[str],
        axis: str,
        position: Optional[float],
        slab: Optional[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray, str, str]:
        axis_alias = {"x": 0, "x1": 0, "y": 1, "x2": 1, "z": 2, "x3": 2}
        axis_idx = axis_alias.get(axis.lower(), 0)
        if axis_idx >= data.ndim:
            axis_idx = 0

        if data.ndim == 1:
            return data, coords[0], labels[0], "1D"

        line_coords = coords[axis_idx]
        line_label = labels[axis_idx]

        transverse_axis = 1 if axis_idx == 0 else 0
        transverse_coords = coords[transverse_axis]
        transverse_label = labels[transverse_axis]

        # Orient data so first index is lineout axis.
        profile_data = data if axis_idx == 0 else data.T

        if slab is not None:
            slab_min, slab_max = slab
            idx_min = int(np.argmin(np.abs(transverse_coords - slab_min)))
            idx_max = int(np.argmin(np.abs(transverse_coords - slab_max)))
            if idx_min > idx_max:
                idx_min, idx_max = idx_max, idx_min
            lineout = np.mean(profile_data[:, idx_min : idx_max + 1], axis=1)
            desc = f"avg {transverse_label} in [{slab_min:.3f}, {slab_max:.3f}]"
            return lineout, line_coords, line_label, desc

        if position is None:
            position = 0.5 * (transverse_coords[0] + transverse_coords[-1])
        idx = int(np.argmin(np.abs(transverse_coords - position)))
        lineout = profile_data[:, idx]
        actual = transverse_coords[idx]
        desc = f"{transverse_label} = {actual:.3f}"
        return lineout, line_coords, line_label, desc


class MagShockZRun:
    """Run context with indexed diagnostics and unified plotting for OSIRIS data."""

    def __init__(
        self,
        input_deck: str,
        norm_density: float,
        B0: astropy.units.Gauss = None,
        Z: int = None,
        m_i: astropy.units.g = None,
    ):
        self.sim = osiris_utils.Simulation(input_deck_path=input_deck)
        self.deck = self.sim._input_deck

        self.norm_density = norm_density
        self.B0 = B0
        self.Z = Z
        self.m_i = m_i

        self._moment_diags: Dict[str, MomentDiagnostic] = {}
        self.diags = DiagnosticIndex(self)
        self.plotter = Plotter(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, refresh: bool = False) -> Dict[str, DiagnosticDescriptor]:
        return self.diags.build(refresh=refresh)

    def list(
        self,
        pattern: Optional[str] = None,
        kind: Optional[str] = None,
        ndim: Optional[int] = None,
    ) -> List[str]:
        return self.diags.list(pattern=pattern, kind=kind, ndim=ndim)

    def get(self, name: str) -> DiagnosticDescriptor:
        return self.diags.get(name)

    def diagnostics_table(
        self,
        refresh: bool = False,
        pattern: Optional[str] = None,
        kind: Optional[str] = None,
        ndim: Optional[int] = None,
        max_rows: Optional[int] = 40,
        print_table: bool = True,
    ) -> str:
        """Return a compact diagnostics table with key metadata.

        Parameters
        ----------
        refresh : bool
            Rebuild index before rendering table.
        pattern : str, optional
            fnmatch pattern filter on diagnostic key.
        kind : str, optional
            Filter by kind: field, density, phase, moment.
        ndim : int, optional
            Filter by dimensionality.
        max_rows : int, optional
            Maximum rows shown in table. None shows all.
        print_table : bool
            Print table to stdout and return it.
        """
        self.index(refresh=refresh)
        keys = self.list(pattern=pattern, kind=kind, ndim=ndim)

        if max_rows is not None:
            keys = keys[:max_rows]

        rows = []
        for key in keys:
            descriptor = self.get(key)
            shape_str = "x".join(str(s) for s in descriptor.shape)
            rows.append(
                {
                    "key": key,
                    "kind": descriptor.kind,
                    "ndim": str(descriptor.ndim),
                    "shape": shape_str,
                }
            )

        if not rows:
            table = "No diagnostics matched the requested filters."
            if print_table:
                print(table)
            return table

        headers = ["key", "kind", "ndim", "shape"]
        widths = {h: max(len(h), max(len(r[h]) for r in rows)) for h in headers}

        header = " | ".join(h.ljust(widths[h]) for h in headers)
        sep = "-+-".join("-" * widths[h] for h in headers)
        body = [" | ".join(r[h].ljust(widths[h]) for h in headers) for r in rows]

        table = "\n".join([header, sep] + body)
        if print_table:
            print(table)
        return table

    def add_moment(
        self,
        species: str,
        p: str = "p1",
        order: int = 0,
        cache: bool = True,
    ) -> str:
        if order not in (0, 1, 2):
            raise ValueError("order must be 0, 1, or 2")
        moment_name = self._moment_name(order, p)
        key = f"{species}/{moment_name}-from-{p}"

        moment_diag = MomentDiagnostic(
            run=self,
            species=species,
            momentum_component=p,
            order=order,
            cache=cache,
        )
        self._moment_diags[key] = moment_diag
        self.diags.register_moment(key, moment_diag)
        return key

    def plot(self, name: str, t: int = 0, mode: str = "auto", **kwargs):
        return self.plotter.plot(name=name, t=t, mode=mode, **kwargs)

    # ------------------------------------------------------------------
    # Backward-compatible wrappers
    # ------------------------------------------------------------------

    def add_moment_diagnostic(self, species: str, momentum_component: str = "p1", order: int = 0):
        return self.add_moment(species=species, p=momentum_component, order=order, cache=True)

    def plot_field(self, field_name: str, timestep: int = 0, **kwargs):
        return self.plot(name=field_name, t=timestep, mode="auto", **kwargs)

    def plot_lineout(self, field_name: str, timestep: int = 0, **kwargs):
        return self.plot(name=field_name, t=timestep, mode="lineout", **kwargs)

    def plot_streak(self, field_name: str, **kwargs):
        return self.plot(name=field_name, mode="streak", **kwargs)

    def make_streak_h5data(self, field_name: str, **kwargs):
        return self.plotter.make_streak_h5data(field_name, **kwargs)

    def __getitem__(self, key):
        return self._get_field(key)

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def _resolve_params(self, B_real=None, Z=None, m_i=None):
        resolved = {}
        if B_real is not None or self.B0 is not None:
            resolved["B_real"] = B_real if B_real is not None else self.B0
        if Z is not None or self.Z is not None:
            resolved["Z"] = Z if Z is not None else self.Z
        if m_i is not None or self.m_i is not None:
            resolved["m_i"] = m_i if m_i is not None else self.m_i
        return resolved

    def _require(self, params, *required_keys):
        missing = [k for k in required_keys if k not in params]
        if missing:
            raise ValueError(
                f"Missing required parameter(s): {', '.join(missing)}. "
                "Provide them as arguments or set during initialization."
            )
        return [params[k] for k in required_keys]

    @property
    def omega_p_real(self):
        return plasmapy.formulary.plasma_frequency(self.norm_density, particle="e-").to("rad/s")

    def omega_pi_real(self, Z: int = None, m_i: astropy.units.g = None):
        params = self._resolve_params(Z=Z, m_i=m_i)
        Z, m_i = self._require(params, "Z", "m_i")
        return plasmapy.formulary.plasma_frequency(
            self.norm_density / Z,
            particle=self._ion_particle(Z, m_i),
        ).to("rad/s")

    @property
    def rqm(self):
        if "al" in self.deck.species:
            return self.deck.species["al"].rqm
        raise KeyError("Ion species 'al' not found in input deck")

    def _ion_particle(self, Z: int, m_i: astropy.units.g):
        return plasmapy.particles.particle_class.CustomParticle(mass=m_i, Z=Z)

    def omega_ce_real(self, B_real: astropy.units.Gauss = None):
        params = self._resolve_params(B_real=B_real)
        (B_real,) = self._require(params, "B_real")
        return plasmapy.formulary.gyrofrequency(B_real, particle="e-").to("rad/s")

    def B_osiris(self, B_real: astropy.units.Gauss = None):
        params = self._resolve_params(B_real=B_real)
        (B_real,) = self._require(params, "B_real")
        B_norm = (
            astropy.constants.m_e
            * self.omega_p_real
            / astropy.units.rad
            / astropy.constants.e.si
        ).to(astropy.units.Gauss)
        return (B_real / B_norm).to(astropy.units.dimensionless_unscaled)

    def omega_ci_real(
        self,
        B_real: astropy.units.Gauss = None,
        Z: int = None,
        m_i: astropy.units.g = None,
    ):
        params = self._resolve_params(B_real=B_real, Z=Z, m_i=m_i)
        B_real, Z, m_i = self._require(params, "B_real", "Z", "m_i")
        return plasmapy.formulary.gyrofrequency(
            B_real,
            particle=self._ion_particle(Z=Z, m_i=m_i),
        ).to("rad/s")

    def omega_ci(self, B_real: astropy.units.Gauss = None):
        params = self._resolve_params(B_real=B_real)
        (B_real,) = self._require(params, "B_real")
        return self.B_osiris(B_real) / self.rqm

    def lambda_D_real(self, T_e: astropy.units.eV):
        return plasmapy.formulary.Debye_length(n_e=self.norm_density, T_e=T_e).to("cm")

    def lambda_D(self, T_e: astropy.units.eV):
        return (self.lambda_D_real(T_e) / self.electron_inertial_length_real()).to(
            astropy.units.dimensionless_unscaled
        )

    def vA_real(
        self,
        B_real: astropy.units.Gauss = None,
        Z: int = None,
        m_i: astropy.units.g = None,
    ):
        params = self._resolve_params(B_real=B_real, Z=Z, m_i=m_i)
        B_real, Z, m_i = self._require(params, "B_real", "Z", "m_i")
        return (B_real / np.sqrt(astropy.constants.mu0 * self.norm_density / Z * m_i)).to("cm/s")

    def vA(self, B_real: astropy.units.Gauss = None):
        params = self._resolve_params(B_real=B_real)
        (B_real,) = self._require(params, "B_real")
        return self.B_osiris(B_real) / np.sqrt(self.rqm)

    def electron_inertial_length_real(self):
        return (astropy.constants.c.si / (self.omega_p_real / astropy.units.rad)).to("cm")

    def ion_inertial_length_real(self, Z: int = None, m_i: astropy.units.g = None):
        params = self._resolve_params(Z=Z, m_i=m_i)
        Z, m_i = self._require(params, "Z", "m_i")
        return (astropy.constants.c.si / (self.omega_pi_real(Z, m_i) / astropy.units.rad)).to("cm")

    def ion_inertial_length(self):
        return np.sqrt(self.rqm)

    def ion_sound_speed_real(
        self,
        T_e: astropy.units.eV,
        adiabatic_index: float = 5 / 3,
        Z: int = None,
        m_i: astropy.units.g = None,
    ):
        params = self._resolve_params(Z=Z, m_i=m_i)
        Z, m_i = self._require(params, "Z", "m_i")
        return (np.sqrt(adiabatic_index * Z * T_e / m_i)).to("cm/s")

    # ------------------------------------------------------------------
    # Coordinates and time conversion
    # ------------------------------------------------------------------

    def _convert_axis(self, axis_values: np.ndarray, units: str, direction: str = "x"):
        direction_l = direction.lower()

        if direction_l in ("p1", "p2", "p3"):
            p_idx = direction_l[1]
            return axis_values, rf"$p_{{{p_idx}}} [m_e c]$"

        allowed_units = [
            "ion",
            "ion inertial length",
            "electron",
            "electron inertial length",
            "physical",
            "cells",
        ]

        if units in ("ion", "ion inertial length"):
            scale = 1.0 / self.ion_inertial_length()
            return axis_values * scale, rf"${direction} [c/\omega_{{pi}}]$"

        if units in ("electron", "electron inertial length"):
            scale = 1.0
            return axis_values * scale, rf"${direction} [c/\omega_{{pe}}]$"

        if units == "physical":
            scale = self.electron_inertial_length_real().to("cm").value
            return axis_values * scale, rf"${direction}$ [cm]"

        if units == "cells":
            return axis_values, rf"{direction} [cell]"

        raise ValueError(f"Unknown units '{units}'. Choose from {allowed_units}.")

    def _convert_time(self, time_value: float, units: str):
        allowed_units = ["ion gyrotime", "electron", "physical"]

        if units in ("ion gyrotime", "1 / omega_ci", "1/omega_ci", "omega_ci^-1"):
            if self.B0 is None:
                raise ValueError("B0 must be set during initialization to use ion gyrotime")
            scale = self.omega_ci().value if hasattr(self.omega_ci(), "value") else self.omega_ci()
            return time_value * scale, r"$[\omega_{ci}^{-1}]$"

        if units == "electron":
            return time_value, r"$[\omega_{pe}^{-1}]$"

        if units == "physical":
            omega_pe = self.omega_p_real.value
            return time_value / omega_pe, "[s]"

        raise ValueError(f"Unknown time units '{units}'. Choose from {allowed_units}.")

    def _coords_for(self, field_obj: Any, timestep: int = 0, spatial_units: str = "ion"):
        data = np.squeeze(np.asarray(field_obj[timestep]))
        if data.ndim == 0:
            data = data.reshape(1)

        grids_raw = getattr(field_obj, "grid", None)
        if grids_raw is None:
            coords = [np.arange(data.shape[0])]
            labels = ["x [idx]"]
            return coords, labels

        grids = grids_raw if isinstance(grids_raw, list) else [grids_raw]

        # Align grid count with squeezed-data dimensionality.
        if len(grids) > data.ndim:
            grids = grids[-data.ndim :]
        elif len(grids) < data.ndim:
            missing = data.ndim - len(grids)
            grids = [np.array([0, n - 1]) for n in data.shape[:missing]] + list(grids)

        axis_names = ["x", "y", "z"]
        axis_meta = getattr(field_obj, "axis", None)
        coords = []
        labels = []

        for i, grid in enumerate(grids):
            grid_arr = np.asarray(grid)
            if grid_arr.ndim == 0 or grid_arr.size == 1:
                axis_vals = np.linspace(0, data.shape[i] - 1, data.shape[i])
            elif grid_arr.size == data.shape[i]:
                axis_vals = grid_arr
            else:
                axis_vals = np.linspace(grid_arr[0], grid_arr[-1], data.shape[i])

            if axis_meta is not None and i < len(axis_meta):
                meta_name = str(axis_meta[i].get("name", "")).lower().strip()
            else:
                meta_name = ""

            direction = axis_names[i] if i < len(axis_names) else f"x{i + 1}"
            if meta_name in ("x1", "x2", "x3"):
                direction = {"x1": "x", "x2": "y", "x3": "z"}[meta_name]
            elif meta_name in ("p1", "p2", "p3"):
                direction = meta_name

            conv_axis, label = self._convert_axis(axis_vals, spatial_units, direction=direction)
            coords.append(np.asarray(conv_axis))
            labels.append(label)

        return coords, labels

    # ------------------------------------------------------------------
    # Diagnostics discovery and access
    # ------------------------------------------------------------------

    def _discover_native_keys(self) -> List[str]:
        keys: List[str] = []

        top_keys = self._keys_from_node(self.sim)
        for top in top_keys:
            try:
                node = self.sim[top]
            except Exception:
                continue

            child_keys = self._keys_from_node(node)
            if child_keys:
                for child in child_keys:
                    keys.append(f"{top}/{child}")
            else:
                keys.append(top)

        return sorted(set(keys))

    def _keys_from_node(self, node: Any) -> List[str]:
        if hasattr(node, "keys"):
            try:
                return list(node.keys())
            except Exception:
                return []
        if hasattr(node, "_quantities") and isinstance(node._quantities, dict):
            return list(node._quantities.keys())
        return []

    def _get_field(self, field_name: str):
        if field_name in self._moment_diags:
            return self._moment_diags[field_name]
        return self.diags.get(field_name).source

    def _species_alias_candidates(self, species: str) -> List[str]:
        """Return likely species key variants used by osiris_utils datasets."""
        lower = species.lower().strip()
        alias_map = {
            "al": ["al", "aluminum"],
            "aluminum": ["aluminum", "al"],
            "si": ["si", "silicon"],
            "silicon": ["silicon", "si"],
            "e": ["e", "electron", "electrons"],
            "electron": ["electron", "electrons", "e"],
            "electrons": ["electrons", "electron", "e"],
        }

        candidates = alias_map.get(lower, [lower])
        if lower not in candidates:
            candidates.append(lower)

        # Include discovered top-level simulation keys that look like species aliases.
        top_keys = [str(k).lower() for k in self._keys_from_node(self.sim)]
        for key in top_keys:
            if key in candidates:
                continue
            if key.startswith(lower) or lower.startswith(key):
                candidates.append(key)

        # Preserve order while deduplicating.
        return list(dict.fromkeys(candidates))

    def _get_native_field(self, field_name: str):
        for sep in ("/", "."):
            if sep in field_name:
                parts = field_name.split(sep)
                if len(parts) != 2:
                    raise ValueError(
                        f"Field name '{field_name}' has too many separators. "
                        f"Expected 'species{sep}quantity' or plain 'quantity'."
                    )
                left, right = parts

                # Try exact species/quantity form first.
                try:
                    return self.sim[left][right]
                except Exception:
                    pass

                # Then try quantity/species ordering used by some datasets.
                try:
                    return self.sim[right][left]
                except Exception:
                    pass

                # Finally, attempt common species aliases in either order.
                for species_alias in self._species_alias_candidates(left):
                    try:
                        return self.sim[species_alias][right]
                    except Exception:
                        continue
                for species_alias in self._species_alias_candidates(right):
                    try:
                        return self.sim[left][species_alias]
                    except Exception:
                        continue

                raise KeyError(f"Could not resolve diagnostic field '{field_name}'")
        return self.sim[field_name]

    def _native_exists(self, field_name: str) -> bool:
        try:
            _ = self._get_native_field(field_name)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Phase-space moments
    # ------------------------------------------------------------------

    def _moment_name(self, order: int, momentum_component: str) -> str:
        names = {0: "n", 1: f"v{momentum_component[1]}", 2: "vth2"}
        if order not in names:
            raise ValueError("order must be 0, 1, or 2")
        return names[order]

    def _quantity_label(self, name: str) -> str:
        """Generate human-readable plot label with basic normalized units."""
        if "-from-p" in name:
            # Example: al/n-from-p1, al/v1-from-p1, al/vth2-from-p1
            match = re.match(r"([^/]+)/([^-]+)-from-(p[123])", name)
            if match:
                _, quantity, pcomp = match.groups()
                if quantity == "n":
                    return rf"$n$ (from {pcomp})"
                if quantity.startswith("v") and quantity != "vth2":
                    component = quantity[1:]
                    return rf"$v_{{{component}}}/c$ (from {pcomp})"
                if quantity == "vth2":
                    return rf"$v_{{th}}^2/c^2$ (from {pcomp})"

        lower = name.lower()
        if any(token in lower for token in ("/p1", "/p2", "/p3", "p1x", "p2x", "p3x")):
            return r"$f(x, p)$"

        return name

    def _resolve_phase_field(self, species: str, momentum_component: str) -> str:
        dim = int(getattr(self.deck, "dim", 1))
        candidates = []
        species_candidates = self._species_alias_candidates(species)

        phase_quantity_candidates = [
            f"{momentum_component}x1",
            f"{momentum_component}x1x2",
            f"{momentum_component}x1x2x3",
            momentum_component,
        ]

        # Prefer quantity suffixes based on dimensionality.
        if dim == 1:
            preferred_quantities = [f"{momentum_component}x1", f"{momentum_component}x1x2"]
        else:
            preferred_quantities = [f"{momentum_component}x1x2", f"{momentum_component}x1"]

        ordered_quantities = list(dict.fromkeys(preferred_quantities + phase_quantity_candidates))

        for sp in species_candidates:
            for q in ordered_quantities:
                candidates.extend(
                    [
                        f"{sp}/{q}",
                        f"{q}/{sp}",
                        f"{sp}.{q}",
                        f"{q}.{sp}",
                    ]
                )

        for candidate in candidates:
            if self._native_exists(candidate):
                return candidate

        # Last resort: search indexed keys.
        for key in self.list(kind="phase"):
            key_l = key.lower()
            if momentum_component not in key_l:
                continue
            if any(sp in key_l for sp in species_candidates):
                return key

        raise KeyError(
            f"Could not find phase-space diagnostic for species='{species}', momentum_component='{momentum_component}'. "
            f"Tried species aliases: {species_candidates}"
        )

    @staticmethod
    def _find_momentum_axis(phase_obj: Any, momentum_component: str) -> int:
        axis_meta = getattr(phase_obj, "axis", None)
        if axis_meta is not None:
            for i, axis_info in enumerate(axis_meta):
                axis_name = str(axis_info.get("name", "")).lower()
                if axis_name == momentum_component.lower():
                    return i
        # Fallback convention: momentum is last axis.
        return len(getattr(phase_obj, "grid", [])) - 1

    @staticmethod
    def _moment_integral(data: np.ndarray, p_axis: np.ndarray, order: int, axis: int) -> np.ndarray:
        weights = p_axis ** order
        reshape = [1] * data.ndim
        reshape[axis] = -1
        return scipy.integrate.simpson(data * weights.reshape(reshape), x=p_axis, axis=axis)

    def _moment_cache_path(self, diag_name: str, timestep: int) -> Path:
        h5_name = diag_name.replace("/", "_")
        sim_path = Path(self.sim._simulation_folder)
        return sim_path / "moments" / h5_name / f"{h5_name}-{timestep:06d}.h5"

    def _read_moment_cache(self, cache_path: Path) -> Optional[np.ndarray]:
        if not cache_path.exists():
            return None
        with h5py.File(cache_path, "r") as handle:
            return handle["data"][()]

    def _write_moment_cache(
        self,
        cache_path: Path,
        data: np.ndarray,
        phase_obj: Any,
        timestep: int,
        p_axis_idx: int,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(cache_path, "w") as handle:
            handle.create_dataset("data", data=data)
            time_val, _ = phase_obj.time(timestep)
            handle.attrs["time"] = float(time_val)

            axis_group = handle.create_group("axes")
            raw_grid = phase_obj.grid if isinstance(phase_obj.grid, list) else [phase_obj.grid]
            axis_counter = 0
            for i, grid in enumerate(raw_grid):
                if i == p_axis_idx:
                    continue
                arr = np.asarray(grid)
                axis_group.create_dataset(f"axis_{axis_counter}", data=arr)
                axis_counter += 1

    def calculate_moment(
        self,
        species: str,
        timestep: int,
        order: int,
        momentum_component: str = "p1",
        cache: bool = True,
    ) -> np.ndarray:
        if order not in (0, 1, 2):
            raise ValueError("order must be 0, 1, or 2")

        moment_name = self._moment_name(order, momentum_component)
        diag_name = f"{species}/{moment_name}-from-{momentum_component}"
        cache_path = self._moment_cache_path(diag_name, timestep)

        if cache:
            cached = self._read_moment_cache(cache_path)
            if cached is not None:
                return cached

        phase_key = self._resolve_phase_field(species, momentum_component)
        phase_obj = self._get_native_field(phase_key)
        data_t = np.asarray(phase_obj[timestep])

        p_axis_idx = self._find_momentum_axis(phase_obj, momentum_component)
        p_grid = np.asarray(phase_obj.grid[p_axis_idx])
        if p_grid.size == data_t.shape[p_axis_idx]:
            p_axis = p_grid
        else:
            p_axis = np.linspace(p_grid[0], p_grid[-1], data_t.shape[p_axis_idx])

        if order == 0:
            result = self._moment_integral(data_t, p_axis, order=0, axis=p_axis_idx)
        elif order == 1:
            density = self.calculate_moment(
                species,
                timestep=timestep,
                order=0,
                momentum_component=momentum_component,
                cache=cache,
            )
            flux = self._moment_integral(data_t, p_axis, order=1, axis=p_axis_idx)
            result = np.divide(flux, density, out=np.zeros_like(flux), where=np.abs(density) > 0)
        else:
            density = self.calculate_moment(
                species,
                timestep=timestep,
                order=0,
                momentum_component=momentum_component,
                cache=cache,
            )
            velocity = self.calculate_moment(
                species,
                timestep=timestep,
                order=1,
                momentum_component=momentum_component,
                cache=cache,
            )

            reshape_p = [1] * data_t.ndim
            reshape_p[p_axis_idx] = -1
            p_broadcast = p_axis.reshape(reshape_p)

            reshape_v = list(data_t.shape)
            reshape_v[p_axis_idx] = 1
            v_broadcast = velocity.reshape(reshape_v)

            centered_sq = np.square(p_broadcast - v_broadcast)
            numerator = scipy.integrate.simpson(data_t * centered_sq, x=p_axis, axis=p_axis_idx)
            result = np.divide(numerator, density, out=np.zeros_like(numerator), where=np.abs(density) > 0)

        if cache:
            self._write_moment_cache(cache_path, result, phase_obj, timestep, p_axis_idx)

        return result

    # ------------------------------------------------------------------
    # Placeholders for higher-level physics summaries
    # ------------------------------------------------------------------

    def upstream_density(self, timestep: int = 0) -> float:
        raise NotImplementedError

    def upstream_temperature(self, species: str, timestep: int = 0) -> float:
        raise NotImplementedError

    def compression_ratio(self, timestep: int = -1) -> float:
        raise NotImplementedError

    def summary(self, timestep: int = -1) -> dict:
        return {
            "upstream_n [cm^-3]": self.upstream_density(0),
            "upstream_T_e [eV]": self.upstream_temperature("electrons", 0),
            "compression ratio": self.compression_ratio(timestep),
        }
