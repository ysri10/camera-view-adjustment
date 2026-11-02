# Camera View Adjustment Prediction for Improving Image Composition — Architecture Summary

**Paper**: *Camera View Adjustment Prediction for Improving Image Composition* ([arXiv:2104.07608](https://arxiv.org/pdf/2104.07608))  
**Authors**: Yu-Chuan Su, Raviteja Vemulapalli, Ben Weiss, Chun-Te Chu, Philip Andrew Mansfield, Lior Shapira, Colvin Pitts (Google Research)

---

## 1. Problem Formulation & Pre-Capture Paradigm
Unlike post-capture cropping—which can only discard pixels inside an already-captured frame (achieving only **0.61 IoU** with ground-truth views via GAIC)—this system guides the photographer **in real time before capture**. Given the current viewfinder image $I$, the model predicts:
1. **Trigger Decision**: Does the current view need adjustment? (Binary)
2. **Directional Action**: Which of **8 discrete camera adjustments** improves composition best?
   - **Horizontal Translation**: *Left* or *Right* (% of image width)
   - **Vertical Translation**: *Up* or *Down* (% of image height)
   - **Zoom**: *Zoom-in* or *Zoom-out* (% of image size)
   - **In-Plane Rotation**: *Clockwise* or *Counter-Clockwise* around the principal axis (radians)
3. **Adjustment Magnitude**: Continuous step size ($\in [5\%, 45\%]$ for shift/zoom; $\in [\frac{\pi}{36}, \frac{\pi}{4}]$ rad for rotation).

---

## 2. Two-Stage Semi-Supervised Training Architecture
Because human annotation of counterfactual pre-capture camera movements is infeasible and existing cropping datasets lack scene diversity, the authors propose a **Two-Stage Teacher–Student Pipeline**:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Composition Scoring Model M_c (Pairwise Ranking Teacher)                        │
│  Inputs: Labeled crop pairs (FCDB, GAICD, CPC) + Unlabeled aesthetic pairs (Unsplash)    │
│  Backbone: MobileNet (299×299) ──► SPP (1×1, 2×2, 5×5) ──► 2×FC(1024, ReLU) ──► Sigmoid  │
│  Loss: L = L_sc (scored crops) + L_bc (perturbed best crops) + L_wc (perturbed Unsplash) │
└───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │ Simulates 8 actions × 9 magnitudes = 72 views
                                            ▼ on 5.5M Open Images (margin threshold Δ = 0.2)
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: View Adjustment Prediction Model M_v (Real-Time Multi-Head Student)             │
│  Inputs: Labeled inverse-perturbation crops (FCDB, GAICD) + Pseudo-labeled Open Images   │
│  Backbone: MobileNet (299×299) ──► SPP (1×1, 2×2, 5×5) ──► 2×FC(1024, ReLU)              │
│  Multi-Head Output:                                                                      │
│    ├─► [Head 1] Suggestion Predictor (Binary Softmax / CE Loss) ──► Trigger? (Yes / No)  │
│    ├─► [Head 2] Adjustment Predictor (8-Way Softmax / CE Loss)  ──► Action Class (1..8)  │
│    └─► [Head 3] Magnitude Predictor  (8 Regressors / L1 Loss)   ──► Step Size per Class  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Composition Scoring Network ($M_c$) & Border Augmentations
* **Network Backbone**: ImageNet-pretrained **MobileNet** ($299 \times 299$ RGB input), followed by a **Spatial Pyramid Pooling (SPP)** layer pooling feature maps at $1\times 1$, $2\times 2$, and $5\times 5$ grids, two **1,024-unit FC layers** (ReLU), and a single **Sigmoid** output node $M_c(I) \in [0, 1]$.
* **Pairwise Margin Ranking Objective**: Trained on positive/negative pairs $(I_p, I_n)$ of the same scene using hinge loss $\max(0, \delta + M_c(I_n) - M_c(I_p))$ across three data streams ($L = L_{sc} + L_{bc} + L_{wc}$, batch sizes $N=16, K=16, P=16$):
  1. $L_{sc}$ (*Scored Candidate Crops*): $\frac{N(N-1)}{2}$ pairs per image from GAICD and CPC ranked by ground-truth MOS.
  2. $L_{bc}$ (*Best Crop Perturbations*): Best crop in FCDB/GAICD as $I_p$, perturbed inside the source frame via shift, zoom, or rotation as $I_n$.
  3. $L_{wc}$ (*Unlabeled Web Photos*): Curated Unsplash photos as $I_p$, randomly perturbed (shift, zoom-out, crop, rotation) as $I_n$ with out-of-frame regions zero-padded.
* **Zero-Pixel Border Augmentations**: Because out-of-frame perturbations in $I_n$ introduce black borders, $M_c$ could cheat by penalizing zero-padded boundaries. To prevent shortcut learning, three synthetic border masks (**Shift borders** $[0,40\%]$, **Zoom-out borders** $[0,40\%]$, and **Rotation borders** $R_{-\theta} \circ R_\theta$ for $\theta \in [-\frac{\pi}{4}, \frac{\pi}{4}]$) are randomly applied to $I_p$ (and labeled pairs).

### Stage 1 $\rightarrow$ 2 Bridge: Simulation-Based Pseudo-Label Generation
* For each unlabeled image in **Open Images** (5.5M subset), the system simulates **8 adjustments $\times$ 9 equally spaced magnitudes = 72 candidate views** (filling unseen regions with zeros).
* $M_c$ scores the original image $I$ and all 72 simulated views. If $\max_k M_c(I_k) - M_c(I) \ge \Delta$ ($\Delta = 0.2$), the top-scoring action and magnitude become the pseudo-label; otherwise, the image is labeled *"No adjustment needed"*.

### Stage 2: Multi-Head View Adjustment Network ($M_v$)
* **Shared Feature Extractor**: Identical to $M_c$—MobileNet ($299 \times 299$) $\rightarrow$ SPP ($1\times 1, 2\times 2, 5\times 5$) $\rightarrow$ $2 \times \text{FC}(1024, \text{ReLU})$.
* **Decoupled Three-Head Architecture** (outperforms joint multi-class heads, which fail on trigger detection):
  1. **Suggestion Predictor**: Binary classifier (Cross-Entropy loss) deciding whether to interrupt the user. Operated at a conservative **0.3 FPR** threshold to avoid annoying false alarms.
  2. **Adjustment Predictor**: 8-class classifier (Categorical Cross-Entropy loss) selecting the movement direction. **Gradient Gating**: Gradients are backpropagated *only* on positive samples requiring adjustment.
  3. **Magnitude Predictor**: 8 class-specific scalar regressors ($\ell_1$ loss). **Gradient Gating**: Gradients are backpropagated *only* through the regressor head corresponding to the ground-truth adjustment class.
* **Optimization & Balancing**: Trained asynchronously on 5 P100 GPUs using Adam ($\text{lr}=2\times 10^{-5}$, weight decay $5\times 10^{-4}$, 64 labeled + 64 pseudo-labeled samples/step) for 250K steps ($M_c$) and 480K steps ($M_v$), weighted by inverse class frequency.
* **Performance**: Achieves **0.608 AUC**, **43.6% TPR @ 0.3 FPR**, **0.750 IoU** on the 3,076-sample benchmark, and improves composition **79.0%** of the time in human studies (**94.2%** success on vertical shifts; **78.9%** multi-step success on $360^\circ$ Pano2Vid).
