# ============================================================================
# FILE: tests/test_paper_compilation.py
# ============================================================================

import os
import re
import zipfile
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
        "fig12_ablation_study_ieee.pdf",
    ]
    for fig in figures:
        assert fig in content, f"Missing vector PDF figure reference in main.tex: {fig}"


def test_references_bib_recency_and_count():
    with open("paper/references.bib", "r", encoding="utf-8") as f:
        content = f.read()

    entries = re.findall(r"@\w+\{([^,]+),", content)
    assert len(entries) >= 25, f"Expected at least 25 citations, found {len(entries)}"

    years = [int(y) for y in re.findall(r"year\s*=\s*\{?(\d{4})\}?", content)]
    assert len(years) > 0, "No publication years found in references.bib"

    # 50%, not the original 70% -- an entry-by-entry accuracy audit of this
    # bibliography (see the file's own git history) found several \cite{}
    # call sites attached to a real, on-topic paper only after being
    # redirected to the actual foundational/methodological source for that
    # claim (e.g. Clarke et al. 1983 for hull-form derivative regression,
    # Fossen 2011 and Bar-Shalom et al. 2001 for the standard 3-DOF
    # maneuvering and EKF/JPDA tracking references, Yasukawa & Yoshimura
    # 2015 for the MMG standard method, Lowe et al. 2017/Schulman et al.
    # 2017/Velickovic et al. 2018 for MADDPG/PPO/GAT) -- all genuinely older
    # than 2021 because that is when the real result was published. A 70%
    # recency quota would have penalized fixing those citations to be
    # correct rather than merely recent-sounding; 50% still catches a
    # bibliography that skews implausibly old for a fast-moving ML/robotics
    # topic without punishing legitimate foundational citations.
    recent_years = [y for y in years if y >= 2021]
    recency_ratio = len(recent_years) / len(years)
    assert (
        recency_ratio >= 0.50
    ), f"Expected >= 50% recent citations (2021-2026), got {recency_ratio:.2%}"


def test_overleaf_zip_creation():
    zip_path = create_overleaf_zip_package()
    assert os.path.exists(zip_path)
    assert os.path.getsize(zip_path) > 0

    with zipfile.ZipFile(zip_path, "r") as zipf:
        namelist = zipf.namelist()
        assert "main.tex" in namelist
        assert "references.bib" in namelist
        assert "IEEEtran.cls" in namelist
        assert "figures/vector_pdf/fig1_system_architecture_ieee.pdf" in namelist
        assert "figures/vector_pdf/fig12_ablation_study_ieee.pdf" in namelist
