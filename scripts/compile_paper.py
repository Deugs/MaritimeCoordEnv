#!/usr/bin/env python3
"""
IEEE Paper Compilation & Overleaf Packaging Script
Compiles paper/main.tex into paper/main.pdf via pdflatex/bibtex,
or creates a complete zip bundle paper/marlin_twin_ieee_paper.zip.

Usage:
    python scripts/compile_paper.py
"""

import os
import shutil
import subprocess
import zipfile

def check_latex_installed() -> bool:
    """Checks whether pdflatex executable is installed and available on PATH."""
    return shutil.which("pdflatex") is not None

def compile_latex_paper():
    """Compiles main.tex using pdflatex and bibtex if available."""
    if not check_latex_installed():
        print("[INFO] pdflatex not found on system PATH. Skipping local PDF build.")
        return False

    paper_dir = os.path.abspath("paper")
    print(f"=== Compiling IEEE LaTeX Paper in {paper_dir} ===")

    try:
        # Run pdflatex -> bibtex -> pdflatex -> pdflatex
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir, check=True)
        subprocess.run(["bibtex", "main"], cwd=paper_dir, check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir, check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir, check=True)
        print("[SUCCESS] Compiled paper/main.pdf successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] LaTeX compilation encountered errors: {e}")
        return False

def create_overleaf_zip_package():
    """Creates a standalone zip file containing all files for Overleaf import."""
    print("=== Creating Overleaf Upload Package: paper/marlin_twin_ieee_paper.zip ===")
    zip_path = os.path.join("paper", "marlin_twin_ieee_paper.zip")
    
    files_to_zip = [
        ("paper/main.tex", "main.tex"),
        ("paper/references.bib", "references.bib"),
        ("paper/IEEEtran.cls", "IEEEtran.cls")
    ]

    # Add all vector PDF figures
    pdf_dir = os.path.join("figures", "vector_pdf")
    if os.path.exists(pdf_dir):
        for fig_file in os.listdir(pdf_dir):
            if fig_file.endswith(".pdf"):
                src = os.path.join(pdf_dir, fig_file)
                dst = f"figures/vector_pdf/{fig_file}"
                files_to_zip.append((src, dst))

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for src, dst in files_to_zip:
            if os.path.exists(src):
                zipf.write(src, arcname=dst)
                print(f"  + Added: {dst}")
            else:
                print(f"  ! Missing file: {src}")

    print(f"[SUCCESS] Created Overleaf package at: {zip_path}")
    return zip_path

def main():
    print("=== MARLIN-Twin IEEE Paper Compilation Utility ===")
    compiled = compile_latex_paper()
    zip_path = create_overleaf_zip_package()
    print("=== Paper Compilation & Packaging Completed! ===")

if __name__ == "__main__":
    main()
