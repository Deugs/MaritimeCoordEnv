# ============================================================================
# FILE: tests/test_diagram_generation.py
# ============================================================================

import os
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
