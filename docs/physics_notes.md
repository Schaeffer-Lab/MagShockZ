# Physics notes and conventions

The unit systems, the code conventions that go with them, and the corrections that were
expensive to find. Everything here is a statement about the physics or the data format,
not about how the repository is laid out.

## OSIRIS normalized units

OSIRIS normalizes to the electron plasma frequency `ω_p` and the reference density `n_0`.
Primed quantities are what live in the HDF5 files (Gaussian-based normalization):

| quantity | normalization | note |
|----------|---------------|------|
| time     | `t' = t·ω_p`                              | frequencies `ω' = ω/ω_p` |
| length   | `x' = ω_p·x / c`                          | i.e. units of `c/ω_p` (electron skin depth) |
| momentum | `u' = p/(m_sp·c) = γv/c`                  | **per-species** mass `m_sp`, so `u' ≈ v/c` for every species |
| E field  | `E' = (e·c/ω_p)/(m_e c²)·E`               | |
| B field  | `B' = (e·c/ω_p)/(m_e c²)·B`               | so `ω_ce = e·B/(m_e c) = B'·ω_p` |
| density  | `n' = n/n_0`                              | |
| energy   | per particle in `m_e c²`; densities in `n_0 m_e c²` | |

Consequences used throughout the analysis (`magshockz/common/energy_partition.py`,
`magshockz/common/temperature_anisotropy.py`):

- Because momentum is per-species (`u' = v/c`), bulk velocities are directly comparable
  across species and to `v_shock` (also in `c`); no per-species rescaling of velocities.
- The 2nd central moment of a phase space is `σ² = uth'² = T/(m_sp c²)`, so temperature in
  `m_e c²` is `T = |rqm|·σ²` (with `|rqm| = m_sp/m_e` for charge state 1).
- Kinetic energy densities (in `n_0 m_e c²`): ram `= ½·n·|rqm|·(⟨u'⟩−v_sh)²`, thermal
  `= ½·n·|rqm|·Σ_d σ_d²` (= `(3/2) n T_iso` isotropic). Both carry the ½ of `½mv²`.
- EM energy densities (in `n_0 m_e c²`) are `B'²/2` and `E'²/2` — the Gaussian
  `B²/(8π)` with `B² = B'²·B_0²`, `B_0 = m_e c ω_p/e`, reduces to exactly `B'²/2`.
  So field and particle energies share the same `n_0 m_e c²` units and are directly
  comparable. Fields look small only because `u_ram/u_B = v²/v_A² = M_A²` (≈100 here).

## FLASH magnetic field (unitsystem = none): the sqrt(4π) is REAL — do not strip it

These FLASH runs set the runtime parameter `unitsystem = "none"`, i.e. the rationalized
MHD convention where the magnetic pressure is `B_code²/2` (the 4π is absorbed into the
field variable). The **physical Gaussian field**, whose `v_A = B/sqrt(4π ρ)` reproduces
the Alfvén speed FLASH actually evolved, is therefore `B_Gauss = sqrt(4π)·B_code ≈
3.545·B_code`. yt's FLASH frontend knows this: for `unitsystem="none"` it sets
`ds.magnetic_unit = sqrt(4π) G`, so a plain `yt.load(...)` + `.to("G")` already returns
the correct physical Gauss. **Do not override `magnetic_unit`.**

History (so it is not re-introduced): a 2026-06-25 change wrongly diagnosed the sqrt(4π)
as a yt bug and overrode `magnetic_unit → 1 G` in `load_for_osiris`, which stripped the
factor and made every B-derived quantity (`v_A`, `M_A`, `β`, `T_ci`, **and the B written
into the OSIRIS deck**) wrong by sqrt(4π)/4π. It was reverted 2026-06-26 after three
independent confirmations: (1) the yt frontend applies sqrt(4π) *by design* only for
`unitsystem="none"`; (2) the dump's `unitsystem` parameter is literally `'none'`; (3) the
measured perpendicular-shock compression (dump 9: r≈3.12, *below* the gas-dynamic ceiling
3.29) matches the RH prediction only with the physical, sqrt(4π)-larger field (M_A≈8.5,
β≈6 → r=3.14), not the stripped one (M_A≈30, β≈78 → r=3.28). The correct numbers are the
original ones: `M_A ≈ 6–8.5`, `β ≈ 2–6`, upstream `|B| ≈ 15–25 T`. **Any OSIRIS deck
regenerated while the override was in place has its B too small by sqrt(4π) and must be
rebuilt.**

## YAML floats: PyYAML is YAML 1.1, so exponents need a dot AND a sign

`5.0e18` and `1e-9` load as **strings**, not floats. PyYAML's 1.1 float resolver wants a
`.` in the mantissa *and* an explicit sign on the exponent — `5.0e+18` and `1.0e-09` are
floats, and `1.0e-9` happens to be fine because the negative exponent already carries its
sign. Nothing errors at load time and most consumers call `float(...)`, so a string sits
in the config until something compares or does arithmetic with it.

`yaml_edit._fmt` therefore pads the mantissa when rendering (`%g`'s `1e-09` → `1.0e-09`;
`%g` always signs the exponent), and `assert_roundtrip` compares numbers numerically
(`rel_tol=1e-5`, just above `%g`'s 6-significant-digit rounding) while still rejecting a
numeric-looking *string* — that check is what caught `tune_flash_shock`'s `t 1` writing
`t_shock_0_s: 1e-9`. `tests/test_config_yaml_scalars.py` walks every `config/*.yaml` and
`runs/*.yaml` and fails on any scalar that is a string Python can parse as a number.
