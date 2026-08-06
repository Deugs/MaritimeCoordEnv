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
        N_r=-20000.0,
        max_rpm=150.0,
        rudder_area=20.0,
        propeller_diameter=4.0,
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
        N_r=-4000.0,
        max_rpm=180.0,
        rudder_area=4.0,
        propeller_diameter=4.0,
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
    ),
}
