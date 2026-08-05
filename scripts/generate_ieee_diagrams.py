#!/usr/bin/env python3
"""
IEEE Diagram Generator Script:
Programmatically renders camera-ready system architecture block diagrams,
EKF/JPDA flowcharts, GAT attention graph diagrams, and curriculum flowcharts at 300 DPI.
Usage:
    python scripts/generate_ieee_diagrams.py
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import networkx as nx  # noqa: E402


def setup_ieee_style():
    """Sets Matplotlib global rcParams to IEEE Transactions standards."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 11.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig_all_formats(name: str):
    """Saves active figure in high-DPI PNG, vector PDF, and vector SVG formats."""
    os.makedirs("figures", exist_ok=True)
    os.makedirs("figures/vector_pdf", exist_ok=True)
    os.makedirs("figures/vector_svg", exist_ok=True)

    plt.savefig(f"figures/{name}.png", dpi=300)
    plt.savefig(f"figures/vector_pdf/{name}.pdf")
    plt.savefig(f"figures/vector_svg/{name}.svg")
    plt.close()


def render_fig1_system_architecture():
    """Renders MARLIN-Twin System Architecture Block Diagram."""
    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    ax.axis("off")

    # Color Palette
    bg_blue = "#e6f0fa"
    bg_green = "#e6f5e6"
    bg_orange = "#fff0e6"
    bg_purple = "#f0e6f5"
    edge_dark = "#333333"

    # Box 1: Physical Environment & Dynamics
    rect1 = patches.FancyBboxPatch(
        (0.02, 0.55),
        0.28,
        0.38,
        boxstyle="round,pad=0.02",
        facecolor=bg_blue,
        edgecolor="#1f77b4",
        linewidth=2,
    )
    ax.add_patch(rect1)
    ax.text(
        0.16,
        0.86,
        "Physical Environment\n& Vessel Dynamics",
        ha="center",
        va="center",
        fontweight="bold",
        color="#003366",
    )
    ax.text(
        0.16,
        0.70,
        "• 3-DOF MMG RK4 Solver\n• Non-linear Maneuvering\n"
        "• IMO Sea Trials (Turning/Zigzag)\n• CPA / TCPA Collision Engine",
        ha="center",
        va="center",
        fontsize=7.5,
    )

    # Box 2: Digital Twin Fusion Layer
    rect2 = patches.FancyBboxPatch(
        (0.36, 0.55),
        0.28,
        0.38,
        boxstyle="round,pad=0.02",
        facecolor=bg_green,
        edgecolor="#2ca02c",
        linewidth=2,
    )
    ax.add_patch(rect2)
    ax.text(
        0.50,
        0.86,
        "Digital Twin\nState Estimator",
        ha="center",
        va="center",
        fontweight="bold",
        color="#004d00",
    )
    ax.text(
        0.50,
        0.70,
        "• ITU-R M.1371 AIS Noise\n• 5x5 Extended Kalman Filter\n"
        "• JPDA Track Association\n• Blackout Dead Reckoning",
        ha="center",
        va="center",
        fontsize=7.5,
    )

    # Box 3: Communication & Policy Layer
    rect3 = patches.FancyBboxPatch(
        (0.70, 0.55),
        0.28,
        0.38,
        boxstyle="round,pad=0.02",
        facecolor=bg_orange,
        edgecolor="#ff7f0e",
        linewidth=2,
    )
    ax.add_patch(rect3)
    ax.text(
        0.84,
        0.86,
        "Graph Neural Network\n& Policy Layer",
        ha="center",
        va="center",
        fontweight="bold",
        color="#804000",
    )
    ax.text(
        0.84,
        0.70,
        "• PyG Graph Builder\n• Multi-Head GAT Encoder\n"
        "• Bandwidth Priority Queue\n• 128-bit Binary Telemetry",
        ha="center",
        va="center",
        fontsize=7.5,
    )

    # Box 4: Multi-Agent RL & Action Execution
    rect4 = patches.FancyBboxPatch(
        (0.20, 0.05),
        0.60,
        0.38,
        boxstyle="round,pad=0.02",
        facecolor=bg_purple,
        edgecolor="#9467bd",
        linewidth=2,
    )
    ax.add_patch(rect4)
    ax.text(
        0.50,
        0.36,
        "CTDE MAPPO Training Infrastructure",
        ha="center",
        va="center",
        fontweight="bold",
        color="#4b0082",
    )
    ax.text(
        0.50,
        0.20,
        "• Centralized Critic V(s_global) + Decentralized Actors pi_i(a_i|o_i)\n"
        "• GAE (gamma=0.99, lambda=0.95) Advantage Estimation\n"
        "• 2-Stage Curriculum: Spatial COLREGs Pre-training -> Comms Loss Resilience Fine-Tuning\n"
        "• Action Output: Propeller RPM, Rudder Angle, Message Priority",
        ha="center",
        va="center",
        fontsize=7.5,
    )

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color=edge_dark)
    ax.annotate("", xy=(0.36, 0.74), xytext=(0.30, 0.74), arrowprops=arrow_props)
    ax.annotate("", xy=(0.70, 0.74), xytext=(0.64, 0.74), arrowprops=arrow_props)
    ax.annotate("", xy=(0.50, 0.43), xytext=(0.50, 0.55), arrowprops=arrow_props)

    ax.set_title(
        "Figure 1: MARLIN-Twin Overall System Architecture Block Diagram", fontweight="bold", pad=10
    )
    plt.tight_layout()
    save_fig_all_formats("fig1_system_architecture_ieee")


def render_fig2_digital_twin_flowchart():
    """Renders Digital Twin EKF/JPDA Flowchart."""
    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    ax.axis("off")

    # Flowchart Blocks
    rects = [
        ((0.02, 0.35), "Raw Sensor Inputs\n(AIS & Radar)", "#1f77b4"),
        ((0.26, 0.35), "JPDA Track\nAssociation", "#ff7f0e"),
        ((0.50, 0.55), "EKF Measurement\nUpdate (AIS Active)", "#2ca02c"),
        ((0.50, 0.15), "Dead Reckoning\nFallback (AIS Blackout)", "#d62728"),
        ((0.76, 0.35), "Vessel State Estimate\ns_hat_i & Confidence", "#9467bd"),
    ]

    for pos, text, col in rects:
        box = patches.FancyBboxPatch(
            pos,
            0.20,
            0.32,
            boxstyle="round,pad=0.02",
            facecolor="#f9f9f9",
            edgecolor=col,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(
            pos[0] + 0.10,
            pos[1] + 0.16,
            text,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=8.0,
        )

    # Decision diamond
    diamond = patches.Polygon(
        [[0.46, 0.51], [0.49, 0.59], [0.52, 0.51], [0.49, 0.43]],
        facecolor="#fff0f0",
        edgecolor="#d62728",
        linewidth=1.5,
    )
    ax.add_patch(diamond)
    ax.text(0.49, 0.51, "AIS\nSignal?", ha="center", va="center", fontsize=7.0, fontweight="bold")

    # Connectors
    ap = dict(arrowstyle="->", lw=1.8, color="#333333")
    ax.annotate("", xy=(0.26, 0.51), xytext=(0.22, 0.51), arrowprops=ap)
    ax.annotate("", xy=(0.46, 0.51), xytext=(0.46, 0.51), arrowprops=ap)
    ax.annotate("", xy=(0.50, 0.71), xytext=(0.49, 0.59), arrowprops=ap)
    ax.annotate("Yes", xy=(0.49, 0.65), fontsize=7.5, color="green")
    ax.annotate("", xy=(0.50, 0.31), xytext=(0.49, 0.43), arrowprops=ap)
    ax.annotate("No", xy=(0.49, 0.37), fontsize=7.5, color="red")
    ax.annotate("", xy=(0.76, 0.51), xytext=(0.70, 0.71), arrowprops=ap)
    ax.annotate("", xy=(0.76, 0.51), xytext=(0.70, 0.31), arrowprops=ap)

    ax.set_title(
        "Figure 2: Digital Twin EKF/JPDA State Estimation & Outage Recovery Flowchart",
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    save_fig_all_formats("fig2_digital_twin_flowchart_ieee")


def render_fig3_gat_attention_diagram():
    """Renders Multi-Head Graph Attention Network Diagram."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axis("off")

    # Graph Nodes
    G = nx.Graph()
    G.add_node(0, pos=(0.5, 0.5), label="Target Vessel i\n(Ownship)")
    G.add_node(1, pos=(0.2, 0.8), label="Vessel j\n(Head-on)")
    G.add_node(2, pos=(0.8, 0.8), label="Vessel k\n(Crossing)")
    G.add_node(3, pos=(0.2, 0.2), label="Vessel l\n(Overtaking)")
    G.add_node(4, pos=(0.8, 0.2), label="Vessel m\n(Stand-on)")

    pos = nx.get_node_attributes(G, "pos")

    # Draw nodes
    for n, (x, y) in pos.items():
        col = "#1f77b4" if n == 0 else "#aec7e8"
        circle = plt.Circle((x, y), 0.09, color=col, ec="#003366", lw=2)
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            f"V{n}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white" if n == 0 else "black",
        )

    # Draw edge attention arrows
    edges = [
        (0, 1, r"$\alpha_{i1}=0.42$"),
        (0, 2, r"$\alpha_{i2}=0.35$"),
        (0, 3, r"$\alpha_{i3}=0.13$"),
        (0, 4, r"$\alpha_{i4}=0.10$"),
    ]
    for u, v, weight_str in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], "k--", lw=1.5, alpha=0.7)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x,
            mid_y,
            weight_str,
            fontsize=8.0,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9),
        )

    ax.set_title(
        "Figure 3: Multi-Head GAT Graph Attention Mechanism (Attention Weights alpha_ij)",
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    save_fig_all_formats("fig3_gat_attention_diagram_ieee")


def main():
    print("=== Generating IEEE Publication Diagrams & Flowcharts ===")
    setup_ieee_style()

    print("1. Rendering Figure 1: System Architecture Block Diagram...")
    render_fig1_system_architecture()

    print("2. Rendering Figure 2: Digital Twin EKF Flowchart...")
    render_fig2_digital_twin_flowchart()

    print("3. Rendering Figure 3: Multi-Head GAT Attention Diagram...")
    render_fig3_gat_attention_diagram()

    print("=== Diagrams Successfully Saved to ./figures/ ===")


if __name__ == "__main__":
    main()
