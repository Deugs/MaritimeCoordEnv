# ============================================================================
# FILE: tests/test_diagram_generation.py
# ============================================================================

import os

import numpy as np

from scripts.generate_ieee_diagrams import (
    render_fig1_system_architecture,
    render_fig2_digital_twin_flowchart,
)
from scripts.generate_ieee_figures import (
    render_fig3_gat_attention_diagram,
    render_fig5_sea_trials,
    render_fig6_digital_twin_blackout,
    render_fig8_degradation_heatmap,
    render_fig9_benchmark_resilience,
    render_fig10_extended_training,
    render_fig11_real_ais_validation,
)


def test_generate_ieee_diagrams():
    render_fig1_system_architecture()
    render_fig2_digital_twin_flowchart()

    assert os.path.exists("figures/fig1_system_architecture_ieee.png")
    assert os.path.exists("figures/fig2_digital_twin_flowchart_ieee.png")


def test_generate_ieee_figures():
    render_fig3_gat_attention_diagram()
    render_fig5_sea_trials()
    render_fig6_digital_twin_blackout()
    render_fig8_degradation_heatmap()
    render_fig9_benchmark_resilience()
    render_fig10_extended_training()
    render_fig11_real_ais_validation()

    assert os.path.exists("figures/fig3_gat_attention_diagram_ieee.png")
    assert os.path.exists("figures/fig5_sea_trials_ieee.png")
    assert os.path.exists("figures/fig6_digital_twin_blackout_ieee.png")
    assert os.path.exists("figures/fig8_degradation_heatmap_ieee.png")
    assert os.path.exists("figures/fig9_benchmark_resilience_ieee.png")
    assert os.path.exists("figures/fig10_extended_training_5k_seeds_ieee.png")
    assert os.path.exists("figures/fig11_real_ais_validation_ieee.png")


def test_digital_twin_figures_are_order_independent():
    """DigitalTwinEstimator.update() draws its own measurement/dead-reckoning noise
    from the global NumPy RNG rather than any generator passed into it, so fig6/fig11's
    reported RMSE previously depended on whatever else had drawn from the global RNG
    earlier in the same process -- not solely on the trajectory and seed, as their own
    docstrings claimed. Perturb the global RNG before each call and confirm the result
    is unaffected, proving the fix (seeding the global state, not just a local
    generator, at the top of each driver function) actually closes that gap."""
    np.random.seed(12345)
    np.random.normal(size=1000)
    fig6_perturbed = render_fig6_digital_twin_blackout()
    fig11_perturbed = render_fig11_real_ais_validation()

    np.random.seed(67890)
    np.random.normal(size=37)
    fig6_clean = render_fig6_digital_twin_blackout()
    fig11_clean = render_fig11_real_ais_validation()

    assert fig6_perturbed == fig6_clean
    assert fig11_perturbed == fig11_clean
