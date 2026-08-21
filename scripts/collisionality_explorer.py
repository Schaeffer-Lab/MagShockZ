#!/usr/bin/env python
"""Interactive explorer for "is the MagShockZ shock collisionless?".

The answer scales as ``v^4``, so it is extremely sensitive to the upstream parameters --
which the experiment does not pin down tightly.  This puts sliders on every one of them and
plots the stopping length against the two scales a shock transition is measured by, so the
sensitivity can be read off directly rather than inferred from a table.

The physics is entirely ``magshockz.common.collisionality``; this script only draws it.

Two cases, selected by the radio buttons:

``Al -> Al (shock)``
    The primary question.  An upstream Al ion enters the front at the shock-frame inflow
    speed and must be stopped over the transition for collisions to have built the ramp.

``Si -> Al (piston)``
    The secondary case: Si piston ions interpenetrating the Al chamber plasma.

Front-ends
----------
``widget_panel()`` -- **the one to use on Perlmutter.**  ipywidgets, via NERSC JupyterHub
    (https://jupyter.nersc.gov, ``analysis`` kernel); see
    ``notebooks/collisionality_explorer.ipynb``.  Perlmutter login nodes have no Qt and no
    X server, and VS Code Remote SSH does not forward X11, so there is usually no display
    to open a window on.

``build()`` / running this script directly -- matplotlib's own Slider widgets in a real
    window.  Needs an interactive backend (only Tk is installed here) and therefore a live
    ``DISPLAY``: ``ssh -Y`` from a real terminal with XQuartz/VcXsrv running, or a NoMachine
    session.  Unlike the batch scripts in this directory it must NOT set "Agg".

``--save PNG`` -- renders one frame headlessly, for checking the script still runs.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

from magshockz.common import collisionality as coll

CM3_TO_M3 = 1e6
KM_S = 1e3

CASES = {
    "Al -> Al (shock)": ("Al", "Al"),
    "Si -> Al (piston)": ("Si", "Al"),
}

#: (label, unit, min, max, initial, log).  The defaults are the FLASH_MagShockZ3D-corrected
#: los45 upstream at 9 ns -- the column the rest of the analysis quotes.
SLIDERS = [
    ("v", "km/s", 20.0, 2000.0, 292.0, True),
    ("n_i", "cm^-3", 1e16, 1e21, 1e18, True),
    ("T_i", "eV", 1.0, 2000.0, 30.0, True),
    ("T_e", "eV", 1.0, 2000.0, 30.0, True),
    ("Z_test", "", 1.0, 14.0, 5.0, False),
    ("Z_field", "", 1.0, 13.0, 5.0, False),
    ("|B|", "T", 0.1, 200.0, 20.0, True),
]


def evaluate(p: dict, test: str, field: str) -> dict:
    """Every derived quantity the panel shows, from the current slider values."""
    n_i = p["n_i"] * CM3_TO_M3
    n_e = p["Z_field"] * n_i
    v = p["v"] * KM_S

    at = lambda speed: coll.slowing_down(                                    # noqa: E731
        v=speed, n_field=n_i, T_field=p["T_i"], z_test=p["Z_test"],
        z_field=p["Z_field"], test=test, field=field, n_e=n_e, T_e=p["T_e"])

    s = at(v)
    d_i, rho_i = coll.shock_scales(n_e=n_e, T_i=p["T_i"], z=p["Z_field"],
                                   b=p["|B|"], ion_species=field)
    v_ti = np.sqrt(2.0 * p["T_i"] * 1.602176634e-19 / (coll.mass_number(field) * 1.66053907e-27))
    v_te = np.sqrt(2.0 * p["T_e"] * 1.602176634e-19 / 9.1093837139e-31)
    electron = coll.slowing_down(v=v_te, n_field=n_i, T_field=p["T_i"], z_test=1.0,
                                 z_field=p["Z_field"], test="e-", field=field,
                                 n_e=n_e, T_e=p["T_e"])

    sweep_v = np.logspace(np.log10(SLIDERS[0][2] * KM_S), np.log10(SLIDERS[0][3] * KM_S), 220)
    sweep = at(sweep_v)

    return dict(s=s, d_i=d_i, rho_i=rho_i, v=v, v_ti=v_ti, electron=electron,
                thermal=at(v_ti), sweep_v=sweep_v, sweep=sweep)


def verdict(knudsen: float) -> tuple[str, str]:
    """The claim the current numbers support, and the colour to say it in."""
    if not np.isfinite(knudsen):
        return "undefined", "0.5"
    if knudsen > 10.0:
        return "COLLISIONLESS", "#1a7f37"
    if knudsen > 1.0:
        return "marginal", "#bf8700"
    return "COLLISIONAL", "#cf222e"


def readout(r: dict, test: str, field: str) -> str:
    s, um = r["s"], 1e6
    lines = [
        f"{test} -> {field}",
        "",
        f"  x  = {s.x:>10.3g}      ({'beam' if s.x > 1 else 'thermal'} limit)",
        f"  lnLambda = {s.coulomb_log:>6.2f}",
        "",
        f"  stopping range   {s.stopping_range * um:>10.4g} um",
        f"  mfp  (v/nu_s)    {s.mfp * um:>10.4g} um",
        f"  mfp  (v/nu_0)    {s.mfp_lorentz * um:>10.4g} um   <- collisionality.md",
        f"  thermal mfp      {r['thermal'].mfp * um:>10.4g} um",
        "",
        f"  d_i              {r['d_i'] * um:>10.4g} um",
        f"  rho_i            {r['rho_i'] * um:>10.4g} um",
        "",
        f"  range / d_i      {s.stopping_range / r['d_i']:>10.3g}   <- the number",
        f"  mfp   / d_i      {s.mfp / r['d_i']:>10.3g}",
        f"  range / rho_i    {s.stopping_range / r['rho_i']:>10.3g}",
        "",
        f"  electron mfp     {r['electron'].mfp * um:>10.4g} um",
    ]
    if s.n_clamped:
        lines += ["", "  ! lnLambda clamped at 1: strongly coupled,", "    treat as indicative only."]
    return "\n".join(lines)


def _make_panel(figsize=(15.5, 8.5), left=0.29):
    """Build the figure, axes and artists shared by both front-ends.

    Returns ``(fig, ax, update)`` where ``update(p, case_label)`` redraws everything from a
    dict of parameter values.  Both the matplotlib-window and the ipywidgets front-end drive
    the same ``update``, so the physics is drawn identically either way.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([left, 0.12, 0.38, 0.80])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("test-particle speed in the background frame [km/s]")
    ax.set_ylabel("length [$\\mu$m]")
    ax.grid(True, which="both", alpha=0.25)

    text = fig.add_axes([left + 0.41, 0.12, 0.30, 0.80])
    text.axis("off")
    handle = text.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9)
    banner = fig.text(left + 0.555, 0.955, "", ha="center", fontsize=15, weight="bold")

    (line_range,) = ax.plot([], [], lw=2.2, color="#0969da", label="stopping range")
    (line_mfp,) = ax.plot([], [], lw=1.6, color="#0969da", ls="--", alpha=0.7,
                          label="mfp $v/\\nu_s$")
    line_di = ax.axhline(np.nan, color="#cf222e", lw=1.6, label="$d_i$")
    line_rho = ax.axhline(np.nan, color="#8250df", lw=1.6, ls=":", label=r"$\rho_i$")
    marker = ax.axvline(np.nan, color="0.35", lw=1.2)
    (dot,) = ax.plot([], [], "o", ms=9, color="#0969da", zorder=5)
    ax.legend(loc="upper left", framealpha=0.9)

    def update(p: dict, case_label: str):
        test, field = CASES[case_label]
        r = evaluate(p, test, field)
        um = 1e6

        line_range.set_data(r["sweep_v"] / KM_S, r["sweep"].stopping_range * um)
        line_mfp.set_data(r["sweep_v"] / KM_S, r["sweep"].mfp * um)
        line_di.set_ydata([r["d_i"] * um] * 2)
        line_rho.set_ydata([r["rho_i"] * um] * 2)
        marker.set_xdata([p["v"]] * 2)
        dot.set_data([p["v"]], [r["s"].stopping_range * um])

        finite = r["sweep"].stopping_range[np.isfinite(r["sweep"].stopping_range)] * um
        if finite.size:
            lo = min(finite.min(), r["rho_i"] * um) * 0.5
            hi = max(finite.max(), r["d_i"] * um) * 2.0
            ax.set_ylim(lo, hi)
        ax.set_xlim(SLIDERS[0][2], SLIDERS[0][3])
        ax.set_title(f"{test} $\\rightarrow$ {field}", fontsize=12)

        handle.set_text(readout(r, test, field))
        label, colour = verdict(r["s"].stopping_range / r["d_i"])
        banner.set_text(label)
        banner.set_color(colour)
        return r

    return fig, ax, update


def build(preset: dict | None = None):
    """Front-end using matplotlib's own Slider widgets, for a real X11/Tk window."""
    values = {name: (preset or {}).get(name, init)
              for name, _, _, _, init, _ in SLIDERS}
    case = ["Al -> Al (shock)"]

    fig, ax, update = _make_panel()
    fig.canvas.manager.set_window_title("MagShockZ collisionality")

    widgets = {}
    for i, (name, unit, lo, hi, init, log) in enumerate(SLIDERS):
        slider_ax = fig.add_axes([0.075, 0.86 - 0.075 * i, 0.16, 0.03])
        label = f"{name} [{unit}]" if unit else name
        widgets[name] = Slider(
            slider_ax, label, np.log10(lo) if log else lo, np.log10(hi) if log else hi,
            valinit=np.log10(values[name]) if log else values[name],
        )
        if log:  # the slider travels in log10; the label must still show the real value.
            widgets[name].valtext.set_text(f"{values[name]:.3g}")
            widgets[name].on_changed(
                lambda val, w=widgets[name]: w.valtext.set_text(f"{10.0 ** val:.3g}"))
        else:
            widgets[name].valtext.set_text(f"{values[name]:.3g}")
            widgets[name].on_changed(
                lambda val, w=widgets[name]: w.valtext.set_text(f"{val:.3g}"))

    radio_ax = fig.add_axes([0.075, 0.14, 0.16, 0.11])
    radio_ax.set_title("case", fontsize=10, loc="left")
    radio = RadioButtons(radio_ax, list(CASES), active=0)

    def current() -> dict:
        return {name: (10.0 ** widgets[name].val if log else widgets[name].val)
                for name, _, _, _, _, log in SLIDERS}

    def redraw(_=None):
        update(current(), case[0])
        fig.canvas.draw_idle()

    for w in widgets.values():
        w.on_changed(redraw)

    def on_case(label):
        case[0] = label
        test, _ = CASES[label]
        # The piston is a distinct species from the background; the shock is not.
        widgets["Z_test"].set_val(10.0 if test == "Si" else widgets["Z_field"].val)
        redraw()

    radio.on_clicked(on_case)
    redraw()
    return fig, widgets, radio


def widget_panel(preset: dict | None = None):
    """Front-end using ipywidgets, for NERSC JupyterHub -- no X11, no DISPLAY needed.

    Perlmutter login nodes have no Qt and no X server, and VS Code Remote SSH does not
    forward X11, so ``build()`` cannot open a window there.  This renders the identical
    panel into a notebook instead.  Use from a cell as::

        %matplotlib inline
        from collisionality_explorer import widget_panel
        widget_panel()

    Returns the ipywidgets container, which Jupyter displays automatically.
    """
    import ipywidgets as ipw
    from IPython.display import display

    values = {name: (preset or {}).get(name, init)
              for name, _, _, _, init, _ in SLIDERS}

    controls = {}
    for name, unit, lo, hi, init, log in SLIDERS:
        label = f"{name} [{unit}]" if unit else name
        common = dict(value=values[name], description=label, continuous_update=False,
                      readout_format=".3g", style={"description_width": "90px"},
                      layout=ipw.Layout(width="330px"))
        # FloatLogSlider takes min/max as EXPONENTS (value stays a real value);
        # FloatSlider takes them as values.
        controls[name] = (
            ipw.FloatLogSlider(base=10, step=0.01, min=np.log10(lo), max=np.log10(hi),
                               **common)
            if log else
            ipw.FloatSlider(step=(hi - lo) / 200.0, min=lo, max=hi, **common))

    case = ipw.ToggleButtons(options=list(CASES), description="case",
                             style={"description_width": "90px"})
    out = ipw.Output()

    # The figure is built once and its artists updated, so a slider drag does not leak
    # figures -- matplotlib's inline backend would otherwise accumulate one per redraw.
    fig, _, update = _make_panel(figsize=(14.0, 7.5), left=0.07)

    def refresh(_=None):
        update({name: controls[name].value for name in controls}, case.value)
        with out:
            out.clear_output(wait=True)
            display(fig)

    for widget in (*controls.values(), case):
        widget.observe(refresh, names="value")

    def on_case(change):
        # The piston is a distinct species from the background; the shock is not.
        test, _ = CASES[change["new"]]
        controls["Z_test"].value = 10.0 if test == "Si" else controls["Z_field"].value

    case.observe(on_case, names="value")
    refresh()

    sliders = ipw.VBox([controls[name] for name, *_ in SLIDERS])
    return ipw.VBox([case, ipw.HBox([sliders, out])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", metavar="PNG",
                        help="render one frame to this path and exit, instead of opening a "
                             "window (headless smoke test; no sliders are interactive)")
    for name, unit, lo, hi, init, _ in SLIDERS:
        parser.add_argument(f"--{name.strip('|').lower().replace('_', '-')}", type=float,
                            default=None, metavar=unit or "VAL",
                            help=f"initial {name} [{unit}] (default {init:g})")
    args = parser.parse_args()

    preset = {}
    for name, *_ in SLIDERS:
        value = getattr(args, name.strip("|").lower().replace("-", "_"), None)
        if value is not None:
            preset[name] = value

    if args.save:
        import matplotlib
        matplotlib.use("Agg")

    fig, _, _ = build(preset)
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
