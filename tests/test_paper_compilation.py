# ============================================================================
# FILE: tests/test_paper_compilation.py
# ============================================================================

import os
import re
import zipfile
import pytest
from scripts.compile_paper import create_overleaf_zip_package

def test_paper_files_exist():
    assert os.path.exists("paper/main.tex")
    assert os.path.exists("paper/references.bib")
    assert os.path.exists("paper/IEEEtran.cls")

def test_main_tex_structure_and_figure_references():
    with open("paper/main.tex", "r", encoding="utf-8") as f:
        content = f.read()

    # Check key sections
    assert "\\title{" in content
    assert "\\begin{abstract}" in content
    assert "\\section{Introduction}" in content
    assert "\\section{Related Work}" in content
    assert "\\section{3-DOF Hydrodynamic Vessel Model" in content
    assert "\\section{The MARLIN-Twin Framework}" in content
    assert "\\section{Experimental Evaluation}" in content
    assert "\\section{Conclusion" in content

    # Check vector PDF figure inclusions
    figures = [
        "fig1_system_architecture_ieee.pdf",
        "fig2_digital_twin_flowchart_ieee.pdf",
        "fig3_gat_attention_diagram_ieee.pdf",
        "fig5_sea_trials_ieee.pdf",
        "fig6_digital_twin_blackout_ieee.pdf",
        "fig8_degradation_heatmap_ieee.pdf",
        "fig9_benchmark_resilience_ieee.pdf",
        "fig10_extended_training_5k_seeds_ieee.pdf",
        "fig11_real_ais_validation_ieee.pdf",
        "fig12_ablation_study_ieee.pdf"
    ]
    for fig in figures:
        assert fig in content, f"Missing vector PDF figure reference in main.tex: {fig}"

def test_references_bib_recency_and_count():
    with open("paper/references.bib", "r", encoding="utf-8") as f:
        content = f.read()

    entries = re.findall(r'@\w+\{([^,]+),', content)
    assert len(entries) >= 25, f"Expected at least 25 citations, found {len(entries)}"

    years = [int(y) for y in re.findall(r'year\s*=\s*\{?(\d{4})\}?', content)]
    assert len(years) > 0, "No publication years found in references.bib"

    recent_years = [y for y in years if y >= 2021]
    recency_ratio = len(recent_years) / len(years)
    assert recency_ratio >= 0.70, f"Expected >= 70% recent citations (2021-2026), got {recency_ratio:.2%}"

def test_overleaf_zip_creation():
    zip_path = create_overleaf_zip_package()
    assert os.path.exists(zip_path)
    assert os.path.getsize(zip_path) > 0

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        namelist = zipf.namelist()
        assert "main.tex" in namelist
        assert "references.bib" in namelist
        assert "IEEEtran.cls" in namelist
        assert "figures/vector_pdf/fig1_system_architecture_ieee.pdf" in namelist
        assert "figures/vector_pdf/fig12_ablation_study_ieee.pdf" in namelist
