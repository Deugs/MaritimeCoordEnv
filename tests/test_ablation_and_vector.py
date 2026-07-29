# ============================================================================
# FILE: tests/test_ablation_and_vector.py
# ============================================================================

import os
import pytest
from scripts.run_ablation_study import main as run_ablation_main

def test_ablation_study_execution():
    run_ablation_main()

    assert os.path.exists("figures/fig12_ablation_study_ieee.png")
    assert os.path.exists("figures/vector_pdf/fig12_ablation_study_ieee.pdf")
    assert os.path.exists("figures/vector_svg/fig12_ablation_study_ieee.svg")
