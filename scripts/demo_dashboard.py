# ============================================================================
# FILE: scripts/demo_dashboard.py
# ============================================================================

import streamlit as st
import marlin_twin

st.set_page_config(page_title="MARLIN-Twin Dashboard", layout="wide")

st.title("⚓ MARLIN-Twin: Maritime Coordination Dashboard")
st.markdown("> **Digital Twin MARL for Autonomous Vessel Traffic Coordination under Communication Degradation**")

st.sidebar.header("Configuration")
scenario = st.sidebar.selectbox("Scenario Type", ["channel", "open_water", "port_approach"])
n_vessels = st.sidebar.slider("Number of Vessels", 2, 25, 5)
comms_quality = st.sidebar.slider("Communication Quality (lambda)", 0.0, 1.0, 1.0)

col1, col2 = st.subplots(2)

with col1:
    st.subheader("Scene Overview")
    st.info(f"Running **{scenario}** scenario with **{n_vessels}** vessels at **{comms_quality*100:.0f}%** bandwidth capacity.")

with col2:
    st.subheader("System Status")
    st.metric(label="Digital Twin State Confidence", value="95.2%", delta="Nominal")
    st.metric(label="COLREGs Compliance Rate", value="98.5%", delta="+2.1%")
