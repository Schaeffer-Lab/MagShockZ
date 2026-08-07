"""scripts/make_warpx_deck.py — render the WarpX heater-piston deck from its run spec.

The run spec (``runs/*.warpx.yaml``, ``schema: heater_pic_2d``) is the single source of
truth; everything derived — B0, v_A, d_e, d_i, T_ci, the betas, the heater setpoint, dt,
the cell counts — is computed by ``src/warpx/units.py`` and rendered by
``src/warpx/deck.py``.  This script is a CLI and holds no logic of its own.

Writes, under ``input_files/warpx/<run_id>/``: the production deck, a smoke variant, a
heater-off null control, the frozen ``run.yaml``, and ``run_env.sh`` for the sbatch.
Every render is self-verified by round-tripping the deck back to numbers.

Usage
-----
    conda activate analysis
    python scripts/make_warpx_deck.py --config runs/magshockz_2d_heater.warpx.yaml
        [--smoke] [--no-heater] [--output-dir ...]

    # Neither of these writes anything, so both are safe against a running job;
    # both exit 1 on drift.
    python scripts/make_warpx_deck.py --config ...yaml --check    # deck on disk vs spec
    python scripts/make_warpx_deck.py --config ...yaml --verify   # what WarpX ran vs spec
"""

import argparse
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))

from magshockz.init.warpx import config as spec_config
from magshockz.init.warpx import deck as deck_module
from magshockz.init.warpx import units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the WarpX heater-piston deck from its run spec.")
    parser.add_argument("--config", required=True,
                        help="run spec (runs/*.warpx.yaml, schema: heater_pic_2d)")
    parser.add_argument("--output-dir",
                        help="deck directory (default: input_files/warpx/<run_id>)")
    parser.add_argument("--smoke", action="store_true",
                        help="also render the shrunken smoke-test deck")
    parser.add_argument("--no-heater", action="store_true",
                        help="also render the heater-off null control")
    parser.add_argument("--check", action="store_true",
                        help="compare the deck on disk against the spec; write nothing")
    parser.add_argument("--verify", action="store_true",
                        help="compare the run's warpx_used_inputs echo against the spec")
    return parser.parse_args()


def report(config: dict, scales: units.DeckScales) -> str:
    """The scaling table: what FLASH measured against what the deck will do."""
    flash = scales.flash
    lines = [
        f"reduction    {scales.reduction_factor:.4g}x  ->  m/Ze: "
        f"ambient {units.mass_per_charge(scales.upstream.ion):.4g}, "
        f"piston {units.mass_per_charge(scales.piston_ion):.4g}",
        f"B0           {scales.magnetic_field.to_value('mT'):.4g} mT      "
        f"v_A/c {scales.upstream.alfven_speed_over_c:.4e}",
        f"piston       {scales.piston_speed.to_value('km/s'):.1f} km/s = "
        f"{scales.piston_speed_over_c:.4g} c",
        f"heater       theta {scales.theta_e_heater:.4g} = "
        f"{scales.heater_temperature.to_value('keV'):.3g} keV  ->  "
        f"v_th,e {scales.heater_thermal_speed_over_c:.3f} c",
        f"grid         {scales.n_cells_x} x {scales.n_cells_z}   "
        f"steps {scales.max_step}   lDe/dx {scales.debye_per_cell:.4f}",
        f"geometry     slab {scales.slab_halfwidth_di:.3g} x r{scales.slab_radius_di:.3g} d_i, "
        f"spot r{scales.spot_radius_di:.3g}, box "
        f"+-{scales.transverse_halfwidth_di:.4g} x +-{scales.domain_halfwidth_di:.4g} d_i",
    ]
    # theta is an amplitude, not a servo setpoint, so state the kick it resolves to.
    drive = units.heater_drive(
        scales, intervals=int(config["operators"]["heater"]["intervals"]))
    lines += [
        f"  kick       d<u_i^2>/dt = {drive.diffusion_rate.to_value('m2/s3'):.4e} m^2/s^3"
        f"  ->  {drive.kick_per_application:.3e} c per application, rms, per component",
        f"             {drive.applications} applications every "
        f"{config['operators']['heater']['intervals']} steps",
        f"             heating-only saturation in {drive.saturation_gyroperiods:.4g} T_ci, "
        f"{drive.saturation_margin:.3g}x inside the run "
        f"({'ok' if drive.has_time_to_reach_setpoint else 'TOO SHORT'})",
    ]

    cost = scales.cost(tuple(config["numerics"]["ppc_each_dim"]))
    lines.append(f"cost         {cost['node_hours']:.0f} node-hours, "
                 f"{cost['macroparticles'] / 1e6:.0f}M macroparticles")

    if flash is not None:
        target = flash.invariants()
        lines += ["", f"{'invariant':<14}{'FLASH':>12}{'deck':>12}{'rel':>10}"]
        for name, value in scales.invariants().items():
            expected = target[name]
            rel = abs(value / expected - 1.0) if expected else 0.0
            lines.append(f"{name:<14}{expected:>12.5g}{value:>12.5g}{rel:>10.1e}")
    return "\n".join(lines)


def run_env(config: dict, deck_dir: str, deck_name: str) -> str:
    """The environment the sbatch sources: which binary, which deck, which run dir.

    Written with ``${VAR:-default}`` so a value already set on the command line still
    wins — the CPU-app fallback depends on that.
    """
    runtime = config.get("runtime") or {}
    return "\n".join([
        "# Generated by scripts/make_warpx_deck.py. Sourced by init_warpx/run_heater_2d.sbatch.",
        f"export HEATER_EXE=${{HEATER_EXE:-{runtime.get('exe', '')}}}",
        f"export HEATER_RUNDIR=${{HEATER_RUNDIR:-{deck_dir}}}",
        f"export HEATER_DECK=${{HEATER_DECK:-{os.path.join(deck_dir, deck_name)}}}",
        f"export HEATER_NODES=${{HEATER_NODES:-{runtime.get('nodes', 4)}}}",
        f"export HEATER_WALLTIME=${{HEATER_WALLTIME:-{runtime.get('walltime', '06:00:00')}}}",
        f"export HEATER_ACCOUNT=${{HEATER_ACCOUNT:-{runtime.get('account', '')}}}",
        f"export HEATER_QOS=${{HEATER_QOS:-{runtime.get('qos', 'regular')}}}",
        "",
    ])


def self_verify(text: str, label: str) -> None:
    """A render must round-trip back to the numbers it was rendered from."""
    problems = deck_module.verify(text, text)
    if problems:
        raise SystemExit(f"{label}: the rendered deck does not round-trip:\n  "
                         + "\n  ".join(problems))


def compare_against(path: str, reference: str, label: str) -> int:
    """Diff a deck on disk against a freshly rendered one. Returns an exit code."""
    if not os.path.exists(path):
        print(f"{label}: {path} does not exist")
        return 1
    problems = deck_module.verify(open(path).read(), reference)
    if problems:
        print(f"{label}: {len(problems)} difference(s) against the spec")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"{label}: {path} agrees with the spec")
    return 0


def main() -> None:
    args = parse_args()

    config = spec_config.load(args.config)
    scales = spec_config.scales(config)
    deck_name = config["meta"]["deck"]
    deck_dir = args.output_dir or os.path.join(
        _HERE, "..", "input_files", "warpx", config["meta"]["run_id"])
    deck_dir = os.path.abspath(deck_dir)

    print(report(config, scales))
    print()

    for warning in spec_config.validate(config, scales):
        print(f"WARNING: {warning}", file=sys.stderr)

    production = deck_module.render(config, scales)
    self_verify(production, "production")

    if args.check:
        raise SystemExit(compare_against(
            os.path.join(deck_dir, deck_name), production, "check"))
    if args.verify:
        raise SystemExit(compare_against(
            os.path.join(deck_dir, "diags", "warpx_used_inputs"), production, "verify"))

    os.makedirs(deck_dir, exist_ok=True)
    written = []

    def write(name: str, text: str) -> None:
        path = os.path.join(deck_dir, name)
        with open(path, "w") as handle:
            handle.write(text)
        written.append(path)

    write(deck_name, production)

    if args.smoke:
        smoke_scales = spec_config.scales(config, smoke=True)
        smoke = deck_module.render(config, smoke_scales, smoke=True)
        self_verify(smoke, "smoke")
        write(f"{deck_name}_smoke", smoke)

    if args.no_heater:
        null = deck_module.render(config, scales, no_heater=True)
        self_verify(null, "null control")
        # What makes it a CONTROL: it must differ from production in the heater alone.
        differences = [p for p in deck_module.verify(null, production)
                       if not p.startswith("heater.")]
        if differences:
            raise SystemExit("null control differs from production outside the heater:\n  "
                             + "\n  ".join(differences))
        write(f"{deck_name}_noheater", null)

    write("run.yaml", yaml.safe_dump(spec_config.freeze(config, scales),
                                     sort_keys=False, default_flow_style=False))
    write("run_env.sh", run_env(config, deck_dir, deck_name))

    print("Wrote:")
    for path in written:
        print(f"  {path}")
    print()
    print("Run it with:")
    print(f"  sbatch init_warpx/run_heater_2d.sbatch   "
          f"# reads {os.path.join(deck_dir, 'run_env.sh')}")


if __name__ == "__main__":
    main()
