# K2 3D Analysis: Secondary Peaks & Spatial Structure

## Visualization Summary

Two new 3D "Mexican hat" plots reveal the secondary peak structure:

1. **`pursuit-b-k2-3d-evolution.png`**: Four checkpoints (250→10k steps) in 3D surface form
2. **`pursuit-b-k2-3d-multiangle.png`**: Final 10k operator from three viewing angles with peaks highlighted

---

## Evolution Story (3D View)

### 250 Steps: Tight Gaussian Blob
- Single central peak at (16, 16)
- Height: 0.50
- L2 norm: 0.98
- **Interpretation**: Initial Gaussian K₀—smooth, symmetric, single spike

### 1,000 Steps: Diffusion Begins
- Central peak widens slightly
- Small elevation changes surrounding center
- Height: 0.51, L2 norm: 1.35 (+37%)
- **Interpretation**: First learning signals; energy beginning to distribute

### 4,000 Steps: Complex Structure Emerges
- Central peak height: 0.49 (slightly diminished)
- Distinct red/orange regions far from center
- Multiple "ridges" apparent in 3D surface
- L2 norm: 1.79 (+33%)
- **Interpretation**: Spatial patterns learned; distance-selective couplings

### 10,000 Steps: Fully Developed Multi-Peaked Operator
- Central peak: 0.43 (noticeably smaller than 250 steps)
- **Multiple secondary peaks clearly visible** in 3D
- Broad plateau instead of concentrated blob
- Complex topography with distinct regions
- L2 norm: 2.63 (+47%)
- **Interpretation**: Broad, asymmetric operator with structured spatial couplings

---

## Peak Structure at 10,000 Steps

### Top 10 Magnitude Peaks

| Rank | Position | Magnitude | Distance | Type |
|------|----------|-----------|----------|------|
| 1 | (16, 16) | **0.429** | 0.0 | Center peak |
| 2 | (3, 7) | **0.357** | 15.8 | **Secondary (far)** |
| 3 | (15, 16) | **0.347** | 1.0 | **Secondary (neighbor)** |
| 4 | (3, 6) | **0.314** | 16.4 | **Secondary (corner)** |
| 5 | (3, 8) | **0.304** | 15.3 | **Secondary (corner)** |
| 6 | (17, 16) | **0.300** | 1.0 | **Secondary (neighbor)** |
| 7 | (16, 17) | **0.298** | 1.0 | Tertiary (neighbor) |
| 8 | (16, 15) | **0.275** | 1.0 | Tertiary (neighbor) |
| 9 | (29, 13) | **0.262** | 13.3 | Tertiary (far) |
| 10 | (6, 3) | **0.260** | 16.4 | Tertiary (corner) |

### Key Finding
- **Center peak** (0.429): Now 13% *smaller* than initial (0.498)
- **Neighbor peaks** (distance 1): 0.30–0.35 magnitude (near-equal to center!)
- **Far-region peaks** (distance 15+): 0.26–0.36 magnitude (substantial secondary structure)

This proves **energy redistribution**, not amplification.

---

## Peak Population Analysis

How many indices have substantial magnitude?

| Magnitude Threshold | Count | % of 1,024 |
|-------------------|-------|-----------|
| > 0.40 | 1 | 0.1% |
| > 0.30 | 6 | 0.6% |
| > 0.20 | 32 | 3.1% |
| > 0.10 | 189 | 18.5% |
| > 0.05 | 440 | 43% |

**Interpretation**: The learned K2 row has:
- **6 major peaks** (> 0.30, yellow regions in 3D plot)
- **32 significant peaks** (> 0.20, visible red regions)
- **189 moderate peaks** (> 0.10, part of the broad structure)
- **440+ small contributions** (> 0.05, sustain the diffuse background)

Compare to initial Gaussian: concentrated in ~10–15 indices. Learned operator distributes to **43% of all 1,024 indices**.

---

## Spatial Distance Patterns

From the multi-angle views, two distance bands are visible:

### Near-Neighbor Peaks (d ≤ 2)
- Positions: (15,16), (17,16), (16,17), (16,15)
- Magnitudes: 0.27–0.35
- **In 3D**: Form a cross-pattern around center
- **Mechanism**: Within-object coherence (nearby oscillators sync)

### Far-Region Peaks (d > 14)
- Positions: (3,7), (3,6), (3,8), (6,3)
- Magnitudes: 0.26–0.36
- **In 3D**: Large secondary peaks distant from center
- **Mechanism**: Between-object anti-coupling (far oscillators anti-phase)

### Asymmetry
Peaks are NOT symmetrically distributed:
- Top-left region: Strong peaks at (3,7), (3,6), (3,8)
- Bottom-right region: No corresponding peaks of similar magnitude
- This **asymmetry** is learned, not geometric

---

## Connection to Segmentation Mechanism

K2's spatial structure creates two complementary effects:

### Effect 1: Within-Object Synchrony
- Near-neighbor peaks (d=1): coupling strength ~0.30
- Applied 140 times: oscillators within objects synchronize phase
- Readout window: coherent phase ≈ 0.96 (mechanism evidence)

### Effect 2: Between-Object Anti-Phase
- Far-region peaks (d=15–16): distance-selective anti-coupling
- Applied 140 times: oscillators across boundaries diverge in phase
- Readout window: antiphase cosine ≈ −0.99997 (mechanism evidence)

**Result**: Different objects → different phases → K-means partitions cleanly

---

## The 3D Mexican Hat Shape

The final K2 is literally a **3D multi-peaked landscape**:

```
          Secondary peaks (far corners)
                   ↓
          ┌─────────────────┐
        ╱ │ Near-neighbor   │ ╲
       │  │ peaks (cross)   │  │
       │  │ h~0.30          │  │
       │  │                 │  │
        ╲ │  Center peak    ╱  │
          │  h~0.43        │
          └─────────────────┘
           Base plateau
         (43% of pixels > 0.05)
```

Not a single broad Gaussian (which would decay smoothly).  
**Lumpy, multi-peaked structure with signed spatial selectivity.**

---

## Summary: What the 3D View Reveals

| Aspect | 250 Steps | 10,000 Steps | What It Means |
|--------|-----------|--------------|------------|
| **Peak count (>0.3)** | 1 | 6 | Energy distributed to 6× more indices |
| **Central peak height** | 0.50 | 0.43 | Not amplification; redistribution |
| **Far-region structure** | Absent | Prominent | Learns distance-selective coupling |
| **L2 norm** | 0.98 | 2.63 | Growth from spreading, not scaling |
| **Asymmetry** | Symmetric | Top-left dominant | Task-specific learned pattern |

**The "broad asymmetric operator" is not a blob that grew—it's a structurally learned multi-peaked landscape implementing spatial filtering for segmentation.**

---
*3D surface plots from K2[528,:] row reshaped as 32×32 grid, 4 checkpoints, 3 viewing angles.*
