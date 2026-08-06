"""Per-`VesselType` hydrodynamic/specification profiles for heterogeneous fleets.

Only CARGO and USV were ever actually instantiated before this — every other
`VesselType` was a label with no data behind it, and CARGO/USV themselves
shared nearly identical MMG coefficients (only mass/size differed), so
"heterogeneous fleet" produced almost-identical maneuvering behavior
regardless of vessel type. This table gives every type distinct
`length/beam/draft/mass/moment_of_inertia` plus the MMG coefficients that
actually drive `VesselDynamics.compute_derivatives` (added mass
`X_u_dot/Y_v_dot/N_r_dot`, damping `X_u/Y_v/N_r`, and `max_rpm`/
`rudder_area`), scaled roughly by size/mass relative to the original
CARGO/USV pair, with `max_speed`/`turning_circle` hand-tuned per type's real
maneuvering character (TANKER slow and wide-turning; FERRY/USV fast and
agile; FISHING small, agile, and slow-topped).
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
    ),
    VesselType.TANKER: dict(
        length=250.0,
        beam=45.0,
        draft=15.0,
        mass=1.2e8,
        moment_of_inertia=1.4e10,
        max_speed=8.0,
        turning_circle=900.0,
        X_u_dot=-250000.0,
        Y_v_dot=-600000.0,
        N_r_dot=-3000000.0,
        X_u=-1200.0,
        Y_v=-6000.0,
        N_r=-25000.0,
        max_rpm=110.0,
        rudder_area=30.0,
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
    ),
}
