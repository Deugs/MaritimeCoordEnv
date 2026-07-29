# ============================================================================
# FILE: marlin_twin/ui_ux_design.md
# ============================================================================

# MARLIN-Twin Maritime UI/UX Design Specification

> **Version**: 2.0 (Refined with Peer-Review Enhancements)
> **Date**: July 2026
> **Purpose**: Define user interface and experience for maritime coordination
>              framework interaction, monitoring, and analysis.

---

## 1. Design Principles

| Principle | Description |
|---|---|
| **Situational Awareness First** | Maritime safety is paramount. Critical alerts are immediate and unmissable. |
| **Spatial Cognition** | Maritime operators think in 2D space. All vessel data is geo-referenced. |
| **Temporal Context** | Trajectories and predictions are central, not just current state. |
| **Resilience Transparency** | Communication degradation effects must be immediately visible. |
| **COLREGs Compliance** | Rule violations are flagged with explicit rule references. |
| **Reproducibility** | All experiments are fully traceable and exportable. |

---

## 2. User Personas

### Persona 1: Maritime AI Researcher (Primary)
- **Goal**: Develop and validate MARL coordination algorithms for autonomous vessels
- **Technical Level**: High (Python, RL, maritime dynamics)
- **Pain Points**: Debugging coordination failures, validating COLREGs compliance, understanding communication impact
- **Frequency**: Daily during active research

### Persona 2: Maritime Safety Assessor
- **Goal**: Evaluate safety of autonomous vessel coordination algorithms
- **Technical Level**: Medium (maritime domain expert, limited ML)
- **Pain Points**: Understanding what the AI does, trusting black-box decisions, identifying failure modes
- **Frequency**: Per evaluation cycle

### Persona 3: Vessel Traffic Operator (Future)
- **Goal**: Monitor live autonomous vessel traffic, intervene when necessary
- **Technical Level**: Medium (VTS operator, not ML expert)
- **Pain Points**: Understanding AI intentions, predicting conflicts, managing communication failures
- **Frequency**: Continuous during operation

---

## 3. Information Architecture

```
MARLIN-Twin Maritime Application
|
|-- Chart Display (Primary View)
|   |-- Vessel Positions & Trajectories
|   |-- Encounter Zones & CPAs
|   |-- Communication Link Quality
|   |-- Digital Twin Confidence
|
|-- Dashboard (Home)
|   |-- Scene Overview
|   |-- Active Alerts
|   |-- Quick Actions
|   |-- Experiment Status
|
|-- Experiments
|   |-- New Experiment (Wizard)
|   |-- Active Experiments
|   |-- Completed Experiments
|   |-- Compare Experiments
|
|-- Training Monitor
|   |-- Real-time Metrics
|   |-- Coordination Quality
|   |-- COLREGs Compliance
|   |-- Communication Utilization
|
|-- Resilience Analysis
|   |-- Degradation Curves
|   |-- Graceful Degradation Reports
|   |-- Fallback Strategy Effectiveness
|   |-- Per-Vessel Resilience
|
|-- Digital Twin
|   |-- State Estimator Status
|   |-- Sensor Fusion View
|   |-- Trajectory Predictions
|   |-- Anomaly Detection
|
|-- Communication
|   |-- Link Quality Map
|   |-- Bandwidth Utilization
|   |-- Message Priority Analysis
|   |-- Jamming/Disruption Simulator
|
|-- Results
|   |-- Export Center
|   |-- Visualization Gallery
|   |-- Paper Figures
|   |-- Reproducibility Package
|
|-- Settings
    |-- Configuration
    |-- Chart Layers
    |-- Alert Thresholds
    |-- User Preferences
```

---

## 4. Screen Specifications

### 4.1 Chart Display (Primary View)

**Purpose**: Real-time maritime situational awareness with all critical information overlaid on a nautical chart.

**Layout**:
```
+-------------------------------------------------------------+
|  [Menu]  MARLIN-Twin Maritime         [Alerts: 2] [Settings] |
+-------------------------------------------------------------+
|                                                             |
|  +-------------------------------------------------------+ |
|  |                                                       | |
|  |           [Nautical Chart / OpenStreetMap]              | |
|  |                                                       | |
|  |     V1 --->                                           | |
|  |          \                                            | |
|  |           \   [CPA zone]                              | |
|  |            \                                          | |
|  |             V2                                        | |
|  |                                                       | |
|  |     V3 ~~~~>  [comm degraded]                         | |
|  |                                                       | |
|  |     [Jamming zone]                                    | |
|  |                                                       | |
|  +-------------------------------------------------------+ |
|                                                             |
|  +--------+ +--------+ +--------+ +--------+ +----------+ |
|  | Vessel | | Vessel | | Vessel | | Vessel | | Selected | |
|  | List   | | Info   | | CPA    | | Comm   | | Vessel   | |
|  |        | | Panel  | | Panel  | | Panel  | | Detail   | |
|  +--------+ +--------+ +--------+ +--------+ +----------+ |
|                                                             |
+-------------------------------------------------------------+
```

**Chart Overlays**:
- **Vessel icons**: Directional, color-coded by type (cargo=blue, USV=green, tanker=orange)
- **Trajectory trails**: Past positions (fading opacity)
- **Predicted trajectories**: Dashed lines with uncertainty ellipses
- **CPA zones**: Concentric circles around vessels (green=safe, yellow=caution, red=danger)
- **Encounter lines**: Lines connecting vessels in encounter, color by risk level
- **Communication links**: Lines between vessels, thickness=bandwidth, color=quality (green=good, red=degraded)
- **Jamming zones**: Shaded circles with "J" marker
- **Restricted zones**: Hatched polygons

**Vessel Icon States**:
| State | Visual |
|---|---|
| Normal | Solid icon, standard color |
| Communication degraded | Icon with yellow border, dashed comm link |
| Communication lost | Icon with red border, no comm link |
| COLREGs violation | Icon flashing red, rule number overlay |
| Emergency | Icon with red cross, pulsing |
| Digital twin low confidence | Icon with question mark overlay |

**Interactions**:
- **Click vessel**: Open detail panel with full state, trajectory, encounters
- **Drag chart**: Pan
- **Scroll**: Zoom
- **Right-click vessel**: Context menu (focus, show encounters, show comm links)
- **Hover vessel**: Tooltip with ID, speed, heading, next waypoint
- **Hover CPA zone**: Show CPA distance and time
- **Time slider**: Scrub through episode history

---

### 4.2 Dashboard (Home)

**Purpose**: At-a-glance status of all experiments and critical alerts.

**Layout**:
```
+-------------------------------------------------------------+
|  MARLIN-Twin MARITIME                  [New Exp] [Settings]  |
+-------------------------------------------------------------+
|                                                             |
|  +------------------+  +------------------+  +-----------+ |
|  | Active Scenes    |  | Latest Result    |  | Critical  | |
|  | 3 running        |  | Ep 500/1000      |  | Alerts    | |
|  | 1 paused         |  | Safety: 0.94     |  |           | |
|  |                  |  | COLREGs: 0.98    |  | [1] V2    | |
|  | [View All]       |  | Comm: 0.73       |  |   CPA <   | |
|  |                  |  | [View Details]   |  |   200m    | |
|  +------------------+  +------------------+  | [2] V5    | |
|                                              |   Comms   | |
|  +-------------------------------------------------------+ | |
|  | Scene Activity Timeline                                | |
|  | [14:32] V2-V4 encounter resolved (CPA: 450m)          | |
|  | [14:31] V5 communication degraded (jamming detected)    | |
|  | [14:30] Digital twin switched to fallback for V3      | |
|  | [14:28] V1 reached waypoint 3/5                       | |
|  +-------------------------------------------------------+ |
|                                                             |
|  +------------------+  +------------------+                | |
|  | Coordination     |  | Communication    |                | |
|  | Quality          |  | Health           |                | |
|  | [Gauge: 0.87]   |  | [Gauge: 0.73]   |                | |
|  | Trend: +0.05    |  | Trend: -0.12    |                | |
|  +------------------+  +------------------+                | |
|                                                             |
+-------------------------------------------------------------+
```

**Alert System**:
- **Level 1 (Critical)**: Collision imminent, immediate audio + visual
- **Level 2 (Warning)**: CPA < safety domain, COLREGs violation
- **Level 3 (Caution)**: Communication degraded, low DT confidence
- **Level 4 (Info)**: Waypoint reached, scenario transition

---

### 4.3 New Experiment (Wizard)

**Purpose**: Guided experiment setup with maritime-specific configuration.

**Flow**:
```
Step 1: Scenario Selection
  - Scenario type: Open Water / Channel Navigation / Port Approach / Dense Traffic
  - Visual preview of scenario layout
  - Number of vessels: slider (2-50)
  - Vessel type mix: cargo, container, tanker, USV, passenger
  - Heterogeneous dynamics toggle

Step 2: Route Configuration
  - Waypoint editor (click on chart to add)
  - Route import (CSV, GPX)
  - Conflict detection (highlight overlapping routes)
  - Estimated voyage time

Step 3: Agent Configuration
  - Algorithm: MAPPO / QMIX / Independent PPO
  - Observation space: checklist (position, velocity, intent, comm quality, DT confidence)
  - Action space: propeller + rudder / waypoint + speed / hybrid
  - COLREGs reward weight
  - Safety reward weight

Step 4: Communication Setup
  - Bandwidth: slider (1200 bps - 9600 bps, AIS standard)
  - Latency: normal distribution (mean, std)
  - Packet loss: base rate + weather-dependent
  - Adaptive bandwidth: toggle
  - Priority queue: toggle
  - Message encoding: raw state / compressed / intent-only

Step 5: Digital Twin Configuration
  - Estimator: Kalman / Particle / Ensemble
  - Sensor fusion: AIS + Radar + Dead reckoning weights
  - Fallback mode: kinematic inference / rule-based / conservative
  - Prediction horizon: slider (60s - 600s)
  - Anomaly detection threshold

Step 6: Resilience Testing
  - Degradation levels: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
  - Jamming scenarios: none / zone / moving / swarm
  - AIS spoofing: toggle
  - GPS denial: toggle
  - Weather degradation: toggle

Step 7: Review & Launch
  - Summary with visual preview
  - Safety check: "No immediate collision risks detected"
  - Estimated compute time
  - Launch button
```

---

### 4.4 Training Monitor

**Purpose**: Real-time monitoring of MARL training with maritime-specific metrics.

**Layout**:
```
+-------------------------------------------------------------+
|  TRAINING: exp_channel_10v    [Pause] [Stop] [Chart View]  |
+-------------------------------------------------------------+
|                                                             |
|  +------------------+  +------------------+  +-----------+ |
|  | Episode 500/1000 |  | Safety Score     |  | COLREGs   | |
|  |                  |  | 0.94             |  | Compliance| |
|  | Avg Reward: 12.3 |  |                  |  | 0.98      | |
|  | Min CPA: 245m   |  | [Sparkline]      |  | [Sparkline]| |
|  +------------------+  +------------------+  +-----------+ |
|                                                             |
|  +-------------------------------------------------------+ |
|  | Training Metrics                                       | |
|  |                                                        | |
|  |  Reward    |    /\    /\    /\                          | |
|  |            |   /  \  /  \  /  \                         | |
|  |            |  /    \/    \/    \                       | |
|  |  Safety    | --.--.--.--.--.--.--                     | |
|  |            |                                          | |
|  |  COLREGs   | ___________^____________                 | |
|  |            |                                          | |
|  |  Comm Util | ~~~~ ~~~~ ~~~~ ~~~~                    | |
|  |            |                                          | |
|  +-------------------------------------------------------+ |
|                                                             |
|  +------------------+  +------------------+                | |
|  | Per-Vessel       |  | Communication    |                | |
|  | Performance      |  | Utilization      |                | |
|  | [Bar chart]      |  | [Stacked area]   |                | |
|  +------------------+  +------------------+                | |
|                                                             |
|  +-------------------------------------------------------+ |
|  | Live Episode View (mini chart)                         | |
|  | [Real-time vessel positions with trajectories]          | |
|  +-------------------------------------------------------+ |
|                                                             |
+-------------------------------------------------------------+
```

**Metrics**:
- **Safety Score**: Composite of min CPA, collision avoidance, near-miss count
- **COLREGs Compliance**: % of timesteps with no rule violations
- **Communication Utilization**: Bandwidth used vs. available
- **Coordination Quality**: Measure of joint action effectiveness
- **Digital Twin Confidence**: Average estimation confidence
- **Voyage Efficiency**: Time to destination vs. optimal

---

### 4.5 Resilience Analysis

**Purpose**: Analyze coordination resilience under communication degradation.

**Layout**:
```
+-------------------------------------------------------------+
|  RESILIENCE ANALYSIS    [Export] [Compare] [Run New Sweep]  |
+-------------------------------------------------------------+
|                                                             |
|  +-------------------------------------------------------+ |
|  | Coordination Resilience Curve                          | |
|  |                                                        | |
|  |  Performance                                           | |
|  |  1.0 |*                                               | |
|  |  0.8 |  *  *                                          | |
|  |  0.6 |    *  *  *                                     | |
|  |  0.4 |       *     *  *                               | |
|  |  0.2 |             *     *  *                        | |
|  |  0.0 +--------------------*-----*---> Comms Quality   | |
|  |       1.0  0.8  0.6  0.4  0.2  0.0                   | |
|  |                                                        | |
|  |  [Ours: smooth curve] [Baseline: cliff at 0.4]      | |
|  +-------------------------------------------------------+ |
|                                                             |
|  +------------------+  +------------------+  +-----------+ |
|  | Per-Vessel       |  | Fallback Strategy |  | Graceful  | |
|  | Degradation      |  | Effectiveness    |  | Degradation| |
|  | [Small multiples]|  | [Bar chart]      |  | Score     | |
|  +------------------+  +------------------+  +-----------+ |
|                                                             |
|  +-------------------------------------------------------+ |
|  | Vessel Detail: V3 (Communication Lost at t=234s)     | |
|  |                                                        | |
|  | Fallback strategy: Kinematic inference                  | |
|  | Estimated position error: 45m at t=300s                | |
|  | COLREGs compliance maintained: YES                    | |
|  | Safety margin maintained: YES                         | |
|  |                                                        | |
|  | [View Trajectory] [View Encounter Analysis]           | |
|  +-------------------------------------------------------+ |
|                                                             |
+-------------------------------------------------------------+
```

**Resilience Curve**:
- X-axis: Communication quality (1.0 = perfect, 0.0 = total loss)
- Y-axis: Performance metric (safety score, efficiency, COLREGs compliance)
- Multiple curves: Safety, Efficiency, COLREGs Compliance
- Shaded region: Confidence interval across episodes
- Baseline comparison: Dashed line for non-adaptive communication

**Graceful Degradation Criteria**:
- **Pass**: Performance at 50% comms > 70% of baseline
- **Pass**: No cliff (discontinuity) in degradation curve
- **Pass**: COLREGs compliance maintained > 90% at all levels
- **Fail**: Any of the above

---

### 4.6 Digital Twin Monitor

**Purpose**: Monitor and understand digital twin state estimation.

**Layout**:
```
+-------------------------------------------------------------+
|  DIGITAL TWIN MONITOR    [Refresh] [Settings] [Export]      |
+-------------------------------------------------------------+
|                                                             |
|  +-------------------------------------------------------+ |
|  | Sensor Fusion View                                     | |
|  |                                                        | |
|  |  V1: [AIS: 0.95] [Radar: 0.87] [DR: 0.30]            | |
|  |       [====|====|====|====|====] Confidence: 0.92   | |
|  |                                                        | |
|  |  V2: [AIS: 0.20] [Radar: 0.91] [DR: 0.85]           | |
|  |       [====|====|====|====|====] Confidence: 0.78   | |
|  |       [WARNING: AIS anomaly detected]                  | |
|  |                                                        | |
|  |  V3: [AIS: 0.00] [Radar: 0.00] [DR: 0.65]           | |
|  |       [====|====|====|====|====] Confidence: 0.45   | |
|  |       [FALLBACK: Dead reckoning only]                | |
|  +-------------------------------------------------------+ |
|                                                             |
|  +------------------+  +------------------+                | |
|  | Trajectory       |  | Anomaly          |                | |
|  | Predictions      |  | Detection        |                | |
|  | [Chart with      |  | [Timeline with   |                | |
|  |  uncertainty]    |  |  flagged events] |                | |
|  +------------------+  +------------------+                | |
|                                                             |
|  +-------------------------------------------------------+ |
|  | Selected Vessel: V2                                    | |
|  |                                                        | |
|  | Position error: 12m (AIS) vs 8m (Radar)               | |
|  | Velocity error: 0.3 m/s                                | |
|  | Heading error: 2.1 degrees                             | |
|  |                                                        | |
|  | Predicted trajectory (5 min):                          | |
|  | [Chart with confidence ellipse]                        | |
|  +-------------------------------------------------------+ |
|                                                             |
+-------------------------------------------------------------+
```

---

### 4.7 Communication Analysis

**Purpose**: Analyze bandwidth-adaptive communication and message prioritization.

**Layout**:
```
+-------------------------------------------------------------+
|  COMMUNICATION ANALYSIS    [Export] [Simulate Jamming]      |
+-------------------------------------------------------------+
|                                                             |
|  +-------------------------------------------------------+ |
|  | Link Quality Map                                       | |
|  |                                                        | |
|  |     V1 ====V2====V3                                    | |
|  |     ||    ||    ||                                     | |
|  |     V4 ====V5====                                      | |
|  |                                                        | |
|  |  Legend: ==== strong, -- weak, .. lost                | |
|  +-------------------------------------------------------+ |
|                                                             |
|  +------------------+  +------------------+  +-----------+ |
|  | Bandwidth        |  | Message Priority |  | Adaptive  | |
|  | Utilization      |  | Distribution     |  | vs Fixed | |
|  | [Stacked area]   |  | [Pie chart]      |  | [Compare]| |
|  +------------------+  +------------------+  +-----------+ |
|                                                             |
|  +-------------------------------------------------------+ |
|  | Message Log (filtered)                                 | |
|  | Time    | From | To  | Priority | Size | Latency | Status |
|  | 14:32:15| V1   | V2  | CRITICAL | 256b | 0.3s    | OK   |
|  | 14:32:14| V3   | V4  | HIGH     | 512b | 0.8s    | OK   |
|  | 14:32:12| V2   | V5  | LOW      | 1024b| 2.1s    | DROP |
|  | 14:32:10| V4   | V1  | MEDIUM   | 512b | 0.5s    | OK   |
|  +-------------------------------------------------------+ |
|                                                             |
+-------------------------------------------------------------+
```

---

## 5. Component Library

### 5.1 Reusable Components

| Component | Description | Usage |
|---|---|---|
| **NauticalChart** | Leaflet/Mapbox-based chart with maritime layers | Chart Display, New Experiment |
| **VesselIcon** | Directional icon with status overlays | Chart Display, Training Monitor |
| **CPAZone** | Concentric circles with risk coloring | Chart Display |
| **TrajectoryLine** | Past (solid) + predicted (dashed) + uncertainty | Chart Display |
| **CommLink** | Line with thickness and color for quality | Chart Display, Communication Analysis |
| **ResilienceCurve** | Degradation curve with baseline comparison | Resilience Analysis |
| **MetricGauge** | Circular gauge with trend arrow | Dashboard, Training Monitor |
| **AlertBanner** | Color-coded alert with dismiss action | All screens |
| **VesselCard** | Summary card for a vessel | Dashboard, Chart Display |
| **EpisodePlayer** | Time-scrubbable episode replay | Results, Training Monitor |

### 5.2 Color System

| Category | Primary | Secondary | Background | Usage |
|---|---|---|---|---|
| **Cargo Vessel** | #1E40AF (blue-800) | #60A5FA (blue-400) | #EFF6FF | Standard vessel |
| **USV** | #059669 (green-600) | #34D399 (green-400) | #ECFDF5 | Autonomous vessel |
| **Tanker** | #B45309 (amber-700) | #FBBF24 (amber-400) | #FFFBEB | Hazardous cargo |
| **Passenger** | #7C3AED (purple-600) | #A78BFA (purple-400) | #F5F3FF | High priority |
| **Safe** | #059669 (green) | - | #ECFDF5 | CPA > 500m |
| **Caution** | #D97706 (amber) | - | #FFFBEB | CPA 200-500m |
| **Danger** | #DC2626 (red) | - | #FEF2F2 | CPA < 200m |
| **Communication** | #2563EB (blue) | - | #EFF6FF | Active link |
| **Comm Degraded** | #D97706 (amber) | - | #FFFBEB | Weak link |
| **Comm Lost** | #DC2626 (red) | - | #FEF2F2 | No link |
| **DT Confident** | #059669 (green) | - | #ECFDF5 | > 0.8 confidence |
| **DT Uncertain** | #D97706 (amber) | - | #FFFBEB | 0.5-0.8 confidence |
| **DT Lost** | #DC2626 (red) | - | #FEF2F2 | < 0.5 confidence |

### 5.3 Typography

| Level | Font | Size | Weight | Usage |
|---|---|---|---|---|
| H1 | Inter | 24px | 700 | Page titles |
| H2 | Inter | 20px | 600 | Section headers |
| H3 | Inter | 16px | 600 | Card titles, panel headers |
| Body | Inter | 14px | 400 | General text, labels |
| Mono | JetBrains Mono | 13px | 400 | Metrics, coordinates, timestamps |
| Caption | Inter | 12px | 400 | Chart labels, legend items |
| Alert | Inter | 14px | 600 | Alert text |

---

## 6. Alert System Design

### Alert Levels

| Level | Trigger | Visual | Audio | Action Required |
|---|---|---|---|---|
| **Emergency** | Collision imminent (< 100m CPA in < 30s) | Full-screen red overlay, pulsing vessel icon | Continuous alarm | Immediate manual override |
| **Critical** | CPA < 200m within 60s, COLREGs violation | Red banner, flashing vessel border | 3 beeps | Review and acknowledge |
| **Warning** | Communication lost > 30s, DT confidence < 0.5 | Amber banner, vessel border | 1 beep | Monitor closely |
| **Caution** | Bandwidth saturation, minor COLREGs deviation | Yellow indicator, no sound | None | Log for review |
| **Info** | Waypoint reached, scenario transition | Green toast | None | None |

### Alert Banner
```
+-------------------------------------------------------------+
|  [!!] CRITICAL: Vessel V2-V4 CPA = 180m in 45s              |
|       COLREGs Rule 15 violation detected                    |
|       [Acknowledge] [Show on Chart] [Override V2]          |
+-------------------------------------------------------------+
```

---

## 7. Responsive Behavior

| Breakpoint | Layout Adjustments |
|---|---|
| **Desktop (>= 1440px)** | Full chart + 4 side panels, all controls visible |
| **Desktop (1280-1439px)** | Chart + 2 side panels, collapsed vessel list |
| **Laptop (1024-1279px)** | Chart + 1 side panel, tabbed panels |
| **Tablet (768-1023px)** | Chart full width, panels as bottom sheet |
| **Mobile (< 768px)** | Chart only, minimal overlays, alert-only mode |

---

## 8. Accessibility Requirements

| Requirement | Implementation |
|---|---|
| **WCAG 2.1 AA** | Minimum contrast ratio 4.5:1 for all text |
| **Color Independence** | Vessel types have distinct shapes, not just colors |
| **Keyboard Navigation** | All chart controls accessible via keyboard |
| **Screen Reader** | ARIA labels on all vessels, alerts read aloud |
| **Reduced Motion** | Respect `prefers-reduced-motion`; no pulsing alerts |
| **High Contrast Mode** | Support `prefers-contrast: high` |

---

## 9. Export Formats

| Format | Content | Use Case |
|---|---|---|
| **PDF Report** | Full experiment summary + key figures + trajectories | Paper supplementary material |
| **GPX** | Vessel trajectories for external chart plotters | Validation with real tools |
| **CSV** | Per-timestep metrics | Statistical analysis |
| **HTML** | Interactive chart replay | Web supplementary |
| **JSON** | Raw experiment data + configurations | Reproducibility package |
| **MP4** | Episode replay video | Presentation, demonstration |
| **PNG/SVG** | Individual figures | Paper figures |

---

## 10. Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl/Cmd + N` | New experiment |
| `Ctrl/Cmd + P` | Pause/resume training |
| `Ctrl/Cmd + R` | Refresh chart |
| `Ctrl/Cmd + S` | Save checkpoint |
| `Ctrl/Cmd + Shift + E` | Export results |
| `Space` | Play/pause episode replay |
| `Left/Right Arrow` | Scrub episode time |
| `+/-` | Zoom chart in/out |
| `F` | Focus on selected vessel |
| `A` | Show all vessels |
| `C` | Toggle communication links |
| `T` | Toggle trajectory predictions |
| `D` | Toggle digital twin confidence overlay |
| `Esc` | Dismiss alert / close panel |
| `?` | Show help overlay |
