"""Per-`VesselType` hydrodynamic/specification profiles for heterogeneous fleets.

Only CARGO and USV were ever actually instantiated before this — every other
`VesselType` was a label with no data behind it, and CARGO/USV themselves
shared nearly identical MMG coefficients (only mass/size differed), so
"heterogeneous fleet" produced almost-identical maneuvering behavior
regardless of vessel type. This table gives every type distinct
`length/beam/draft/mass/moment_of_inertia` plus the MMG coefficients that
actually drive `VesselDynamics.compute_derivatives` (added mass
`X_u_dot/Y_v_dot/N_r_dot`, damping `X_u/Y_v/N_r`, and `max_rpm`/
`rudder_area`).

`thrust_coefficient` is *derived*, not estimated: it's the value that makes
`compute_derivatives`' thrust/drag steady-state balance land exactly on this
type's own `max_speed` at full throttle (rpm_fraction=1.0), solving
`thrust_coefficient = (-X_u * max_speed**2) / ((max_rpm/60)**2 *
propeller_diameter**4)` for each type individually — a single shared
constant can't hit every type's own max_speed because their thrust/drag
ratios differ (see `VesselDynamics.thrust_coefficient`'s docstring for why a
global default doesn't work).

`yaw_coefficient` is derived the same way, targeting a steady turning
radius of `turning_circle / 2` at 80%-throttle cruise speed and full
30-degree rudder: `yaw_coefficient = (-N_r_dot * r_target) / (max_rudder_angle
* rudder_area * u_cruise**2)` where `r_target = u_cruise / (turning_circle /
2)` and `u_cruise = 0.8 * max_speed`. This ignores the (much smaller)
quadratic `N_r` damping term, so the resulting turning circle lands in the
right order of magnitude relative to `turning_circle`, not exactly on it.

**CARGO and USV are the exception**: their `N_r`/`yaw_coefficient` are
*not* the formula above. That formula's damping term was weak enough that
the simulated turning circle missed the IMO Res. MSC.137(76) 5*L tactical-
diameter ceiling by a wide margin (see `VesselDynamics.N_r`'s docstring in
`data_classes.py` for the diagnosis). CARGO/USV's values here were instead
found by direct simulation search and verified, on the actual MMG solver
(`MMGDynamicsSolver.run_turning_circle_test`/`run_zigzag_test`), to satisfy
both the 5*L turning-circle ceiling and the 10/10 zig-zag <=25-degree
overshoot criterion simultaneously — checked across 10 consecutive zig-zag
reversal cycles, not just the officially-measured first two overshoots, to
rule out a response that passes the checked overshoots en route to a
slower divergence. CARGO and USV are the only two types any scenario in
this codebase actually instantiates (see `scenarios.py`'s default
CARGO/USV-by-parity fleet); CONTAINER/TANKER/PASSENGER/FERRY/FISHING below
still use the un-reverified formula value and should not be assumed
IMO-compliant if a future scenario starts using them.

**Provenance — read before trusting any single number.** Only `TANKER`'s
`length`/`beam`/`draft`/`mass`/`max_speed`/`propeller_diameter`/`rudder_area`
are grounded in a real, named vessel (see `# CITED` comments on that entry
and the references below); its MMG derivative coefficients and every field
on the other 6 types are engineering estimates — scaled roughly by size/mass
relative to an original CARGO/USV pair, with `max_speed`/`turning_circle`
hand-tuned per type's general maneuvering character (TANKER slow and
wide-turning; FERRY/USV fast and agile; FISHING small, agile, and
slow-topped) — **not measurements or literature values**. Do not cite them
as sourced.

References for the TANKER entry:
- KVLCC2 ("KRISO Very Large Crude Carrier 2"), the vessel TANKER is modeled
  on, is a real, internationally used benchmark tanker hull from KRISO
  (Korea Research Institute of Ships & Ocean Engineering), used in the
  SIMMAN 2008/2014 ship-maneuvering-prediction workshops
  (https://www.simman2008.dk/kvlcc/kvlcc2/). `max_speed` (15.5 kn),
  `propeller_diameter` (9.86 m), and `rudder_area` (273.3 m^2) are drawn
  directly from KVLCC2's published full-scale particulars. `length`/`beam`/
  `draft` (320.0 / 58.0 / 20.8 m) and block coefficient (Cb=0.8098, used
  only to derive `mass` below) are this hull's widely-reproduced principal
  dimensions, appearing consistently across dozens of open ship-maneuvering
  papers — high confidence, but not independently re-verified against a
  fetched primary source in the session that added them (general web access
  was proxy-restricted at the time).
- Yasukawa, H., Yoshimura, Y. (2015). "Introduction of MMG standard method
  for ship maneuvering predictions." Journal of Marine Science and
  Technology, 20, 37-52. The paper defining the MMG standard method this
  codebase's equation *structure* follows, using KVLCC2 as its worked
  example — cited for the vessel identity/methodology, not for any specific
  coefficient value transcribed into this file (its captive-model-test
  derivative table was not accessible to verify in the session that added
  this).
- Clarke, D., Gedling, P., Hine, G. (1983). "The application of manoeuvring
  criteria in hull design using linear theory." Transactions RINA, 125,
  45-68. A real, established regression method for *estimating*
  hydrodynamic derivatives from a ship's principal dimensions when no
  captive-model-test data is available — the natural next step for
  replacing the remaining engineering-estimate coefficients below with a
  documented method, not applied here because its exact regression
  formulas could not be verified in the session that added this file.
"""

from marlin_twin.data_classes import VesselType

VESSEL_PROFILES: dict[VesselType, dict] = {
    VesselType.CARGO: dict(
        length=150.0,
        beam=25.0,
        draft=8.0,
        mass=1.5e7,
        moment_of_inertia=2.0e9,
        max_speed=12.0,
        turning_circle=400.0,
        X_u_dot=-50000.0,
        Y_v_dot=-100000.0,
        N_r_dot=-500000.0,
        X_u=-1000.0,
        Y_v=-5000.0,
        # N_r/yaw_coefficient below are IMO-criteria-verified, not
        # formula-derived (see module docstring's "IMO turning-circle /
        # zig-zag compliance" note and `VesselDynamics.N_r`'s docstring).
        N_r=-1.0e9,
        max_rpm=150.0,
        rudder_area=20.0,
        propeller_diameter=4.0,
        thrust_coefficient=90.0,
        yaw_coefficient=8000.0,
    ),
    VesselType.CONTAINER: dict(
        length=300.0,
        beam=40.0,
        draft=14.0,
        mass=8.0e7,
        moment_of_inertia=9.0e9,
        max_speed=13.5,
        turning_circle=650.0,
        X_u_dot=-180000.0,
        Y_v_dot=-400000.0,
        N_r_dot=-2200000.0,
        X_u=-1500.0,
        Y_v=-7000.0,
        N_r=-30000.0,
        max_rpm=140.0,
        rudder_area=35.0,
        propeller_diameter=4.0,
        thrust_coefficient=196.1396,
        yaw_coefficient=34.201795,
    ),
    VesselType.TANKER: dict(
        # CITED — KVLCC2 principal dimensions (see module docstring).
        length=320.0,  # Lpp, m
        beam=58.0,  # m
        draft=20.8,  # m
        # CITED — mass derived from real principal dimensions above: block
        # coefficient Cb=0.8098 (KVLCC2) x displacement volume x seawater
        # density 1025 kg/m^3 = 320.0*58.0*20.8*0.8098*1025 ~= 3.20e8 kg.
        mass=3.20e8,
        # ESTIMATED — yaw moment of inertia via the common naval-architecture
        # radius-of-gyration approximation k_zz ~= 0.25*Lpp (not a measured
        # value for this specific hull): mass * (0.25*320.0)**2.
        moment_of_inertia=2.05e12,
        # CITED — KVLCC2 design speed, 15.5 kn = 7.97 m/s.
        max_speed=7.97,
        # ESTIMATED — no measured tactical diameter available; IMO
        # Res. MSC.137(76) sets a regulatory ceiling of 5*Lpp, used here only
        # as an upper-bound reference point, not KVLCC2's actual trial result.
        turning_circle=1600.0,
        # ESTIMATED — MMG derivative coefficients, retuned (not sourced) to
        # keep simulated yaw response plausible at the real mass/inertia above.
        X_u_dot=-900000.0,
        Y_v_dot=-2200000.0,
        N_r_dot=-1.1e8,
        X_u=-4000.0,
        Y_v=-20000.0,
        N_r=-450000.0,
        # ESTIMATED — typical slow-speed VLCC main-engine shaft rpm range.
        max_rpm=80.0,
        # CITED — KVLCC2 rudder area, 273.3 m^2.
        rudder_area=273.3,
        # CITED — KVLCC2 full-scale propeller diameter, 9.86 m.
        propeller_diameter=9.86,
        # DERIVED — see module docstring; makes this hull's own thrust/drag
        # balance reach its cited max_speed (7.97 m/s) at full throttle.
        thrust_coefficient=15.1214,
        yaw_coefficient=150.701001,
    ),
    VesselType.PASSENGER: dict(
        length=200.0,
        beam=30.0,
        draft=7.0,
        mass=3.5e7,
        moment_of_inertia=4.0e9,
        max_speed=14.0,
        turning_circle=350.0,
        X_u_dot=-90000.0,
        Y_v_dot=-150000.0,
        N_r_dot=-700000.0,
        X_u=-1600.0,
        Y_v=-8000.0,
        N_r=-28000.0,
        max_rpm=170.0,
        rudder_area=28.0,
        propeller_diameter=4.0,
        thrust_coefficient=152.5952,
        yaw_coefficient=24.360450,
    ),
    VesselType.USV: dict(
        length=30.0,
        beam=8.0,
        draft=2.0,
        mass=5.0e5,
        moment_of_inertia=5.0e7,
        max_speed=12.0,
        turning_circle=120.0,
        X_u_dot=-8000.0,
        Y_v_dot=-15000.0,
        N_r_dot=-60000.0,
        X_u=-400.0,
        Y_v=-1500.0,
        # IMO-criteria-verified, not formula-derived -- see CARGO's N_r above.
        N_r=-4.0e7,
        max_rpm=180.0,
        rudder_area=4.0,
        propeller_diameter=4.0,
        thrust_coefficient=25.0,
        yaw_coefficient=24000.0,
    ),
    VesselType.FERRY: dict(
        length=100.0,
        beam=18.0,
        draft=4.0,
        mass=4.0e6,
        moment_of_inertia=1.5e8,
        max_speed=15.0,
        turning_circle=250.0,
        X_u_dot=-30000.0,
        Y_v_dot=-60000.0,
        N_r_dot=-150000.0,
        X_u=-900.0,
        Y_v=-4500.0,
        N_r=-10000.0,
        max_rpm=190.0,
        rudder_area=22.0,
        propeller_diameter=4.0,
        thrust_coefficient=78.8824,
        yaw_coefficient=8.681179,
    ),
    VesselType.FISHING: dict(
        length=25.0,
        beam=7.0,
        draft=3.0,
        mass=3.0e5,
        moment_of_inertia=3.0e7,
        max_speed=9.0,
        turning_circle=100.0,
        X_u_dot=-6000.0,
        Y_v_dot=-12000.0,
        N_r_dot=-40000.0,
        X_u=-350.0,
        Y_v=-1300.0,
        N_r=-3000.0,
        max_rpm=130.0,
        rudder_area=3.5,
        propeller_diameter=4.0,
        thrust_coefficient=23.5901,
        yaw_coefficient=60.630455,
    ),
}
