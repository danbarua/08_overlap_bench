# K2 Evolution: How the Gaussian Blob Becomes a Broad Asymmetric Operator

## Visual Summary

### Two-Figure Story

**Figure 1: `pursuit-b-k2-evolution.png`**  
Eight heatmaps showing K2 row 528 (center pixel) reshaped as 32×32 spatial grid across training.

**Figure 2: `pursuit-b-k2-row-evolution.png`**  
Left panel: Magnitude profiles along center row (pixel 16) over time.  
Right panel: L2 norm, max magnitude, and mean magnitude growth trajectories.

---

## The Transformation: Three Phases

### Phase 1: Gaussian Initialization (250 steps)
- **Shape**: Tight Gaussian blob centered at pixel 528 (center)
- **L2 norm**: 0.98
- **Max magnitude**: 0.50
- **Structure**: Concentrated energy, minimal asymmetry
- **Interpretation**: Initial coupling matrix K₀ (Gaussian sheet from run_pursuit_b.py)

### Phase 2: Diffusion & Structure Emergence (500–2000 steps)
- **Evolution**: Blob stays centered but diffuses; asymmetric patterns begin
- **L2 norm growth**: 0.98 → 1.51 (54% growth)
- **Max magnitude**: Stable ~0.49-0.52 (no peak explosion)
- **Observation**: Energy spreads outward but peak magnitude remains bounded
- **Implication**: Gradient descent learns spatial structure, not just amplifying peaks

### Phase 3: Learned Broad Structure (4000–10000 steps)
- **Evolution**: Fully developed asymmetric operator with spatial patterns
- **L2 norm**: 1.51 → 2.63 (74% growth from phase 2 end)
- **Max magnitude**: 0.49 → 0.43 (slightly decreased, not increased)
- **Mean magnitude**: Stays near constant ~0.02
- **Structure**: Left and right wings in magnitude profile; secondary peaks far from center
- **Key insight**: Peak magnitude does not amplify, but L2 norm grows → energy distributed across more indices

---

## Critical Distinction: Broadening ≠ Amplification

The growth pattern tells the story:

| Metric | 250 steps | 10k steps | Change |
|--------|-----------|-----------|--------|
| **L2 norm** | 0.98 | 2.63 | +168% ↑ |
| **Max magnitude** | 0.498 | 0.429 | −14% ↓ |
| **Mean magnitude** | ~0.02 | ~0.02 | Stable |

**This is NOT uniform amplification.** If K2 were simply getting larger in all entries:
- Max magnitude would increase
- Mean magnitude would increase
- L2 norm would grow proportionally

Instead, we observe:
- L2 norm grows (energy spreading)
- Max magnitude shrinks slightly (no peak explosion)
- Mean magnitude stays flat (most entries stay small)

**Conclusion**: The network learns to **distribute coupling strength across spatial locations**, not concentrate it. This is **structural learning**, not scaling.

---

## Connection to Mechanism Analysis

PURSUIT-B-MECHANISM.md confirms this interpretation:

**Lines 81-94 (Quantitative evidence):**

Mechanism table shows:
- ‖K-K₀‖_F / ‖K₀‖_F = 0.4936 at 250 steps
- ‖K-K₀‖_F / ‖K₀‖_F = 2.8823 at 10,000 steps

This relative change matches our observation: K2 restructures, it doesn't scale uniformly.

**Lines 96-106 (Spatial distance selectivity):**

Learned coupling changes vary by distance:
- Close neighbors (d ∈ [0, 1.01]): +0.04408 mean change
- Medium distance (d ∈ [4.01, 8.01]): +0.00485
- Far neighbors (d > 16.01): −0.05000

This spatial selectivity **proves structural learning**, not uniform scaling.

---

## Why This Matters for Segmentation

K2's spatial structure drives phase dynamics:

1. **K2 learns to couple nearby oscillators within objects** (positive changes for d ≤ 1)
2. **K2 learns to anti-couple distant oscillators across boundaries** (negative changes for d > 16)
3. This structure, iterated 140 times, creates **within-object synchrony** and **between-object antiphase**

Mechanism evidence shows phase coherence emergence (lines 152-159):
| Steps | Within-object coherence | Between-object cosine | FG-ARI |
|-------|-------------------------|----------------------|---------|
| 250 | 0.7654 | −0.8901 | 0.5897 |
| 10,000 | 0.9601 | −0.99997 | 0.9195 |

**K2's spatial structure → phase structure → partition quality**

---

## Summary: Gaussian to Asymmetric

| Property | Initial (Gaussian) | Learned (10k) | What Changed |
|----------|-------------------|----------------|--------------|
| Peak shape | Tight concentrated | Diffuse, broad | Energy spread across space |
| Peak magnitude | 0.50 | 0.43 | Slightly smaller |
| Total energy (L2) | 0.98 | 2.63 | 2.7x → more indices populated |
| Spatial pattern | Symmetric | Asymmetric, signed | Task-dependent structure |
| Distance selectivity | Uniform (Gaussian) | Binned | Close neighbors vs far: opposite signs |

**The transformation is not "make everything bigger," but "populate more indices with task-specific structure."**

---
*Analysis grounded in PURSUIT-B-MECHANISM.md (lines 73-106, 240-241) and 8-checkpoint inspection.*
