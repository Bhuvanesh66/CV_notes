# Module 4 — Object Detection & Segmentation: Exam Notes

**Course:** SST Computer Vision (Scaler) · **Sessions covered:** 6, 7, 8, 9 (as taught) + 10 (from the syllabus)

**Built from:**
- the class slides for Sessions 6–9
- the lecture transcripts for Sessions 6–9 (auto-generated, so some terms were garbled; see Part 12.1)
- `question-pattern.pdf` (the final exam format)
- standard textbook facts where the slides were silent

> **Preview tip (VS Code):** press `Ctrl+Shift+V` to render the tables and the `$…$` formulas.

---

## How your exam is built, and where this module scores

**Six sections, 100 marks, 120 minutes.** Bank the marks you know first, then come back to the hard ones.

| Section | Format | What Module 4 will most likely give you | Where to practise |
|---|---|---|---|
| **A** Single MCQ (10 × 1) | One correct option | Task choice (classification / detection / segmentation), what R-CNN → Fast → Faster changed, YOLO tensor size, NMS purpose, softmax vs sigmoid | Part 11-A |
| **B** Multi-select (10 × 2) | **All correct and no wrong ones** for full marks. Watch for "always" and "never" | R-CNN limitations, anchor facts, NMS facts, DETR facts, P/R facts, small-object tricks | Part 11-B, Part 12.2 traps |
| **C** Descriptive (10 × 2, ≤ 100 words) | Define → one distinction or example → stop | Semantic vs instance vs panoptic; λ in the loss; YOLO limitations; anchor-free; mAP vs accuracy; √d_k; LayerNorm vs BatchNorm; object queries | Part 11-C (16 model answers) |
| **D** Coding (20, choose 1 of 3) | Reusable function, edge cases, hidden tests | **IoU, NMS, P/R/F1, AP, Dice**, argmax with ties, box conversion | Part 9 (tested code) |
| **E** Case study (15) | Problem → task → data → model → metric → failure mode | Shelf monitoring, pedestrian detection, medical, defects, drones, video | Part 10 |
| **F** Numericals (15) | Formula → substitute → answer | **IoU / Dice, AP and mAP**, YOLO / SSD / anchor counts, HOG length, conv sizes, ViT tokens, attention shapes, video tensors | Part 8 (36 worked problems) + Part 7 formula sheet |

---

## Contents

| Part | Topic | Session |
|---|---|---|
| 1 | Big picture: classification → localisation → detection → segmentation | 6 |
| 2 | Classical and two-stage detection: sliding windows, HOG + SVM, R-CNN, Fast R-CNN, Faster R-CNN | 6 |
| 3 | One-stage detection: SSD, YOLO v1 → v12, anchors vs anchor-free, NMS, IoU, P/R, AP / mAP, Dice | 7 |
| 4 | Transformers: attention, ViT, **DETR**, two-stage vs one-stage vs DETR, modern detectors, practical analysis | 8 |
| 5 | Video understanding: 3D CNN, two-stream, I3D, video transformers, tracking | 9 |
| 6 | Image segmentation: FCN, U-Net, Mask R-CNN, IoU / Dice / mIoU, modern models, lab | 10 |
| 7 | One-page formula sheet | all |
| 8 | Numericals bank (Section F) | all |
| 9 | Coding bank (Section D) | all |
| 10 | Case-study playbook (Section E) | all |
| 11 | Question bank (Sections A, B, C) | all |
| 12 | Traps, transcript corrections, mnemonics, last-night checklist | all |

**Symbols used:** ★ = very likely to be examined. "Teacher:" = a point the lecturer stressed in class, so expect it in a question.

---


---

# PART 1 — The Big Picture: What are we predicting?

## 1.1 The task ladder (Session 6, slide 1 — near-certain MCQ)

| Task | Output | Single / multiple objects | Example application (from slides) |
|---|---|---|---|
| **Classification** | **ONE class** for the whole image | Single | Defect / no defect on a machined part |
| **Classification + Localization** | **CLASS + ONE BOX** | Single (class known; find *where*) | Find the tumour in a brain MRI |
| **Object Detection** | **CLASSES + MANY BOXES** | Multiple, multiple classes | Cars / pedestrians / cyclists |
| **Semantic segmentation** | **CLASS PER PIXEL** (no object ids) | — | Cars / road / sky / building |
| **Instance segmentation** | **A MASK PER OBJECT** | Multiple | Individual cars |
| **Panoptic segmentation** | Stuff + things: every pixel gets a class, and things also get an id | — | Every region + every instance |

**Spatial understanding increases: CLASS → BOX → MANY BOXES → PIXELS.**

**Memory hook:** *What? → Where (one)? → What + where (many)? → Every pixel.*

## 1.2 Why not use segmentation for everything? (slide 4 + class discussion — good 2-mark answer)

| | Classification | Detection | Segmentation |
|---|---|---|---|
| Output size | Small (1 label) | 4 numbers + class per box (sparse) | **Per pixel** (dense) |
| Compute | Low | More | **Highest** |
| Annotation | Cheap (one label per image) | Draw tight boxes | **Expensive** (trace every boundary) |

- **The engineering trade-off: ACCURACY ↔ COMPUTATION ↔ LATENCY ↔ ANNOTATION COST.**
- **"Choose the simplest way to solve the application."** The application decides the task: *what do I actually need the model to output?*
- **Teacher's reason, model answer:** "Segmentation predicts a label for **every pixel**, which is expensive. A box needs only centre, width and height to say where an object is. Pixel-level detail is rich but **not required for every task**, and pixel labels are costly to annotate."

**Foundation models and deployment (teacher's point):**
- Today, big foundation models auto-label data: **SAM** for segmentation, **DINOv3**-based backbones for classification. They run on cloud GPUs, and a **human verifies** the labels (human-in-the-loop).
- **Why not deploy the big model directly?** Latency and connectivity. A self-driving car can't wait for a cloud round trip (that might be about 1 prediction per second), and it may have no network. So a **small model runs on-device**.

---

# PART 2 — Session 6: Object Detection I — Classical & Two-Stage

## 2.1 The generic detection pipeline (3 components + box fix)

```
Image → [1] Region proposals → [2] Feature extraction → [3] Classifier (person / flag / ball / none)
                                                     └→ [4] Bounding-box regression (w,h → w*,h*)
```

1. **Region proposals:** find candidate regions ("mini-images") that may contain objects.
2. **Feature extraction:** turn each region into a vector x₁ … x_d.
3. **Classifier:** decide which class, **including "none" / background**.
4. **Box regression:** the proposed box is rarely perfect (e.g., it cuts the player's legs off), so **regress** the corrected values (x\*, y\*, w\*, h\*).
   - Training: compare the predicted (w, h) with the ground truth (w\*, h\*) using an L2 loss.
   - The loss sends a signal like "stretch the height" through backprop, the same way house-price regression learns.

**How the three components evolved (the slide's master table — memorise it):**

| Era | Region proposals | Feature extraction | Classifier |
|---|---|---|---|
| **Pre-2012** | Sliding windows (brute force) | Hand-crafted (**HOG**, SIFT) | **Linear SVM** |
| **R-CNN** (2014) | **Selective Search** | **CNN** (AlexNet/VGG) | **Linear SVM** (1 per class) |
| **Fast R-CNN** (2015) | Selective Search | CNN | **CNN (softmax)**, one network |
| **Faster R-CNN** (2015) | **CNN (RPN)** | CNN | CNN |

**Trend:** step by step, every hand-designed part is replaced by a CNN and more computation is **shared**.

## 2.2 Sliding windows — the brute-force idea

- Slide a fixed-size window over every location, at **many scales** (an image pyramid) and aspect ratios, and classify each window.
- **Problems:**
  - It needs a **huge** number of windows, so it is slow.
  - Fixed aspect ratios fit objects badly.
  - Many overlapping windows fire on the same object, so **NMS** is needed to remove duplicates.

## 2.3 HOG — Histogram of Oriented Gradients (Dalal & Triggs, 2005)

**SIFT vs HOG (the slide's key box):**

| SIFT / ORB | HOG |
|---|---|
| **Detects keypoints** first | **No keypoint detection** |
| Descriptor around **sparse** keypoints | Descriptor over a **dense grid** covering the whole detection window |
| Good for matching and stitching | Good for detecting a whole object (e.g., a person) |

**The hierarchy: pixels → cells → blocks → one descriptor.**

1. **Gradients at every pixel** (computed **once**):
   - $G_x = I(x{+}1,y) - I(x{-}1,y)$, $\;G_y = I(x,y{+}1) - I(x,y{-}1)$
   - **Magnitude** $= \sqrt{G_x^2 + G_y^2}$ (*how strong*)
   - **Orientation** $= \tan^{-1}(G_y/G_x)$ (*which direction*)
2. **Cells** (typically 8×8 pixels): each pixel casts a **vote into an orientation bin**, weighted by its gradient magnitude. This gives an **orientation histogram** (typically **9 bins over 0–180°**, 20° each) that answers "what edge directions are common here?"
   - Unlike SIFT, HOG does **not** pick a dominant orientation; it keeps the whole histogram.
3. **Blocks** (typically **2×2 cells**): concatenate the neighbouring cell histograms and **normalise** them. This gives robustness to **illumination and contrast**. Blocks overlap (stride = 1 cell).
4. **Collect all blocks** across the window: `[hist | hist | hist | …]` → **one long HOG feature vector**.

**Classic numerical — HOG length for the Dalal-Triggs person window:**
- Window **64×128**, cell 8×8 → **8 × 16 cells**.
- Block 2×2 cells, stride 1 cell → (8−1) × (16−1) = **7 × 15 = 105 blocks**.
- Each block = 4 cells × 9 bins = **36** values.
- Total = 105 × 36 = **3,780 features**.

**Gradient numerical:** $G_x = 30$, $G_y = 40$ → magnitude = √(900 + 1600) = **50**, orientation = tan⁻¹(40/30) ≈ **53.1°**. With 9 bins of 20° each (0–20, 20–40, 40–60, …), it falls in the **40–60° bin**. Dalal-Triggs actually splits the vote between the two nearest bin centres (50° and 70°).

## 2.4 HOG + Linear SVM detector

**Training:**
- Positive windows (a person) and negative windows (background) → HOG vector x for each.
- Train a **linear SVM** to separate them. It learns the weights w₁ … w_n.

**Testing:**
- **Slide the window** everywhere, and at multiple scales.
- For each window, score $= \mathbf{w}^\top\mathbf{x} + b$. A **high score means object**.
- **NMS** removes duplicate detections.

**Pipeline to remember: SEARCH EVERYWHERE → HOG FEATURES → SVM → NMS.**

## 2.5 The 2012 moment

AlexNet (2012) showed that CNNs are excellent **feature extractors** (AlexNet, VGG and ResNet are all feature extractors). Teacher's question: *"It's 2012. What do you replace in the HOG pipeline?"* **Answer: replace the hand-crafted feature extractor (HOG) with a CNN.** That gives **R-CNN**.

## 2.6 R-CNN (Girshick et al., 2014) — "Regions with CNN features"

```
Image → Selective Search (~2000 proposals) → crop + WARP each to fixed size (e.g., 227×227)
      → CNN (AlexNet/VGG, features from fc7)   ── run 2000 times!
      → class-specific linear SVMs (1 per class)  +  bbox regressors (on pool5 features)
```

1. **Selective Search** does hierarchical grouping:
   - Start from small over-segmented patches.
   - Repeatedly **merge the most similar neighbouring regions** (similar colour, texture, size, fill).
   - Do this at **different scales**, because objects appear at different scales.
   - Result: **≈ 2,000 proposals per image**. Merging is done between **neighbouring** regions.
2. **Warp:** a CNN needs a **fixed input size** (H and W are hyper-parameters fixed by the FC layers), but the proposals come in all sizes. So **every proposal is cropped and resized** to the same H×W.
3. **Feature extraction:** an ImageNet-pretrained CNN, **fine-tuned on cropped ground-truth objects**. The **fc7** outputs are the features.
4. **Classifier:** **one linear SVM per class** (cat vs others, dog vs others, …).
5. **Box regression:** **4 regressors** predict x\*, y\*, w\*, h\* from the **pool5** features.

**Parameters:**
- **W_conv:** from ImageNet, then fine-tuned on cropped ground-truth objects.
- **W_classifier:** learned from cropped ground-truth objects.
- **W_regression:** learned from ground-truth boxes.

**Inference cost (slide formula):**

$$\text{Time} = \text{Proposal time} + \#\text{Proposals}\times(\text{Conv time} + \text{classification} + \text{regression})$$

**R-CNN problems (list ≥ 3 in a 2-mark answer):**
1. **Slow:** ~2,000 separate CNN forward passes per image. **Test: 47 s / image; training ≈ 3 days** (VGG).
2. **No joint learning:** proposals, CNN, SVM and regressor are trained **separately**. The loss never reaches the proposal stage.
3. **Ad-hoc, multi-stage objectives:**
   - fine-tune the CNN with softmax (log loss)
   - then post-hoc SVMs (hinge loss)
   - then post-hoc box regressors (squared loss)
4. **Huge disk space:** features for every proposal are cached to disk.
5. **Warping distorts** the aspect ratio of objects.

## 2.7 Receptive field — the key that unlocks Fast R-CNN

- **Receptive field** = the region of the **input image** that influences one value in a feature map.
- Stacked stride-1 k×k convs: $RF_L = 1 + L(k-1)$. For 3×3 kernels:

| Layers | Receptive field |
|---|---|
| 1 | **3×3** |
| 2 | **5×5** |
| 3 | **7×7** |

- In general: $RF_{l} = RF_{l-1} + (k_l - 1)\times\prod_{i<l} s_i$. **Strides multiply the growth.**

> **Watch the wording:** the slide asks for the "region of influence of each pixel", answering "5×5" after one 3×3 conv and "7×7" after the second. The standard textbook answer is 3×3 after one layer and 5×5 after two. In the exam, show the formula and state your convention. Either way, the region **grows by 2 with each 3×3 layer**.

**Why it matters:** every location of a deep feature map "sees" a known patch of the input image. So instead of cropping the **image** 2,000 times, run the CNN **once** and **crop the feature map**. That is Fast R-CNN.

## 2.8 Fast R-CNN (Girshick, 2015)

```
Whole image → CNN ONCE → conv5 feature map
           → project each Selective-Search RoI onto the feature map
           → RoI Pooling → fixed 7×7×512 → FC layers
           → (a) Linear + softmax (K+1 classes)  (b) Linear bbox regressor
           Multi-task loss = log loss + smooth L1 loss   (trainable end-to-end, except proposals)
```

**RoI Pooling** (= a single-level SPP layer). It is how a variable-size region becomes a fixed size:
1. Take the RoI on the feature map (any size h × w).
2. Divide it into a fixed **H × W grid** (e.g., 7 × 7 = k = 49 sub-regions).
3. **Max-pool each sub-region.** The output is always H × W per channel.

**VGG-16 numbers (the slide's example):**
- VGG-16's last pool outputs **512 × 7 × 7**.
- Replace that last max-pool with RoI pooling with **H = W = 7** (k = 49).
- The output is **512 × 49 = 25,088**, exactly the same as before.
- So the **pretrained FC weights still fit**.

**RoI-pool numerical:** a 4×4 RoI pooled to 2×2 → split it into four 2×2 quadrants and take the max of each.

```
RoI:  1 3 | 2 1          max of each quadrant:
      5 2 | 0 4    →        5  4
      ----+----             8  9
      7 8 | 1 9
      6 0 | 3 2
```

**Why Fast R-CNN is faster than R-CNN:**
- **One** CNN pass per image instead of ~2,000; the conv features are shared.
- **Softmax replaces the SVMs** and the box regressor sits **inside the network**, so it trains end-to-end with one multi-task loss.
- **No disk caching** of features.

**Remaining bottleneck:** **Selective Search is still separate, slow (CPU) and not learned.**
- Teacher's puzzle: *"Find the odd one out"* in the summary row (Selective Search | CNN | CNN).
- **Answer: Selective Search.** Replace it too. That gives Faster R-CNN.

## 2.9 Faster R-CNN (Ren et al., 2015) — Region Proposal Network (RPN)

**Idea:** use a CNN to make the proposals as well, and **share the backbone** between the RPN and the detector.

**RPN mechanics (VGG backbone):**
1. Take the last conv feature map (h × w × 512).
2. Slide a **3×3 conv** over it, which gives a **512-d vector for each of the w × h positions**.
3. At each position, centre **k anchor boxes** of different **sizes and aspect ratios**. The paper uses **k = 9: 3 scales × 3 ratios (1:1, 1:2, 2:1)**.
4. Two sibling 1×1 conv heads ask two questions per anchor:
   - **Classification:** *does this anchor contain an object?* (objectness) → **2k scores** (or k with sigmoid)
   - **Regression:** *how do I move and resize this anchor to fit the true box?* → **4k offsets**
5. Top proposals (~300) after NMS → **RoI pooling → Fast R-CNN head** (class + box refinement).

**Anchor labelling (ground truth for the RPN):**
- An anchor is **positive (1) if IoU > 0.7** with a ground-truth box. The paper also marks the highest-IoU anchor for each GT box as positive.
- An anchor is **negative (0) if IoU < 0.3**.
- Anchors in between are ignored in training.

**RPN loss:**

$$\mathcal{L}(\{p_i\},\{t_i\}) = \frac{1}{N_{cls}}\sum_i \mathcal{L}_{cls}(p_i, p_i^*) + \lambda\,\frac{1}{N_{reg}}\sum_i p_i^*\,\mathcal{L}_{reg}(t_i, t_i^*)$$

- $p_i$ = predicted objectness; $p_i^*$ = 1 if the anchor contains an object, else 0.
- $N_{cls}$ = batch size; $N_{reg}$ = batch size × k.
- **$p_i^*$ in the regression term means only positive anchors are regressed.** It makes no sense to fit a box to background.
- $\mathcal{L}_{reg}$ = **smooth L1**: $0.5x^2$ if $|x|<1$, else $|x| - 0.5$. It is less sensitive to outliers than L2.

**Why λ? (The teacher spent a long time on this — very likely 2-mark question.)**
- **λ balances two losses that live on different scales.**
- If the classification loss is around 10s and the regression loss around 1000s, the regression gradient **overpowers** everything: the network only improves boxes, and the classification signal never reaches the weights.
- λ (e.g., 1/100) brings both into a similar range.
- It can also be used deliberately to **prioritise** one task, or be **scheduled** (e.g., weight classification more after 20 epochs).
- It is **not** regularisation.

**Box parameterisation (anchor-relative offsets — standard, likely numerical):**

$$t_x = \frac{x - x_a}{w_a},\quad t_y = \frac{y - y_a}{h_a},\quad t_w = \log\frac{w}{w_a},\quad t_h = \log\frac{h}{h_a}$$

- Example: anchor (x_a, y_a, w_a, h_a) = (100, 100, 50, 100); GT box = (110, 90, 100, 100).
  - t_x = 10/50 = **0.2**, t_y = −10/100 = **−0.1**
  - t_w = log(100/50) = ln 2 = **0.693**, t_h = log 1 = **0**
- Why log? It keeps width and height positive after decoding ($w = w_a e^{t_w}$) and makes scale changes symmetric.

**Anchor-count numerical:** feature map 40 × 60, k = 9 → 40 × 60 × 9 = **21,600 anchors**. The RPN outputs 2k = 18 class scores + 4k = 36 offsets **per position**.

**Training (slide): 4-step alternating training**, sharing one VGG:
1. Train the **RPN** from an ImageNet-initialised network.
2. Train **Fast R-CNN** from ImageNet init, using the step-1 proposals.
3. **Freeze the shared conv layers** (from step 2) and fine-tune the RPN-only layers.
4. Keep the shared conv layers frozen and fine-tune the Fast R-CNN FC layers.

Pretrained ImageNet backbones mean **transfer learning** does most of the work. **Impact:** Faster R-CNN + RPN were the basis of **1st-place entries in ILSVRC & COCO 2015**: ImageNet detection and localization, COCO detection and segmentation.

## 2.10 The R-CNN family at a glance (high-yield comparison)

| | R-CNN | Fast R-CNN | Faster R-CNN |
|---|---|---|---|
| Proposals | Selective Search (~2k, CPU) | Selective Search | **RPN (learned, anchors)** |
| CNN passes per image | **~2,000** (one per proposal) | **1** | **1** (shared with the RPN) |
| Fixed-size trick | **Warp image crops** | **RoI pooling on the feature map** | RoI pooling |
| Classifier | Linear **SVM** per class | **Softmax** (in-network) | Softmax |
| Box regression | Separate linear regressors | In-network (smooth L1) | In-network (RPN + head) |
| Training | Multi-stage, ~3 days | Single stage (except proposals) | ~End-to-end (4-step / joint) |
| Test speed | **~47 s/img** | ~2 s/img (mostly SS time) | **~0.2 s/img (5–7 fps)** |
| PASCAL VOC mAP (slide chart) | ~53% | ~68% | ~76% |

**PASCAL VOC history (slide chart):** hand-crafted methods **plateaued around 40% mAP** (2010–2012). Deep CNN detectors rose to the **mid-70s within 3 years**.

## 2.11 Two-stage detector philosophy (syllabus item — 2-mark question)

- **Stage 1: "Where might objects be?"** Propose a **sparse** set of **class-agnostic** regions (Selective Search or RPN).
- **Stage 2: "What is it, exactly where?"** For **each region**, classify it and **refine** its box (RoI pooling → heads).
- **Pros:**
  - **High accuracy**, especially localisation and **small objects**.
  - Stage 1 removes most of the easy background, so stage 2 sees a **balanced** set of candidates.
- **Cons:**
  - **Slower** (two sequential stages, per-RoI computation).
  - More complex pipeline; historically not real-time.
- **Contrast with one-stage (Session 7):** one-stage detectors predict class + box **densely in a single pass**, with no proposal stage. They are faster but must cope with extreme **foreground/background imbalance**.

**Model 2-mark answer (≤100 words):** "A two-stage detector first generates a small set of candidate object regions (Selective Search or an RPN), then classifies each region and refines its bounding box using pooled features (RoI pooling). Separating *where* from *what* gives high accuracy and good localisation, because the second stage only sees likely objects. The cost is speed: proposals and per-region heads make it slower than one-stage detectors such as YOLO and SSD, which predict boxes and classes in a single forward pass."

## 2.12 Session 6 quick self-test

1. Why must R-CNN warp proposals? → The CNN's FC layers need a fixed input size.
2. What does RoI pooling replace in VGG-16? → The last max-pool (it outputs 512×7×7).
3. What does Faster R-CNN replace, and with what? → Selective Search, with the RPN.
4. What makes an anchor positive? → IoU > 0.7 with a ground-truth box (negative if < 0.3).
5. Why is $p_i^*$ multiplied into the regression loss? → So only positive anchors contribute to box regression.
6. HOG vs SIFT in one line → HOG is a dense grid of gradient histograms with no keypoints; SIFT builds descriptors around sparse keypoints.


---

# PART 3 — Session 7: One-Stage Detectors (SSD, YOLO v1→v12) & Evaluation

## 3.1 Why one-stage?

- **Detection problem:** one image contains N objects, and N is **unknown** in advance. For each object we must jointly predict:
  - **box** (x, y, w, h)
  - **class**
  - **confidence**
- **Two-stage** (R-CNN family): propose ~2000 regions → run a CNN per region. Most of the time goes into proposing.
- **One-stage** (SSD, YOLO): image → **one CNN, one forward pass**, dense predictions over the whole image → boxes + classes. This gives **real-time** speed.
- **Historical trade-off:** one-stage detectors gave up some accuracy, especially on **small and crowded** objects, for a big speed gain. *"Closing that gap is the whole story of YOLOv2 → v12."*

## 3.2 SSD — Single Shot MultiBox Detector (Liu et al., ECCV 2016)

**Key idea:**
- **Skip the proposal stage**: discretise the space of possible boxes up front as **default boxes** (≈ anchors).
- One CNN predicts **class scores + box offsets** for every default box in **one pass**.
- **Extra ingredient:** predict from **multiple feature-map resolutions**, not just the last layer.

**Architecture:**
- VGG-16 base (pretrained, truncated; fc6/fc7 converted to conv) plus **extra conv layers** that keep shrinking the map.
- A small 3×3 conv predictor on each of **6 feature maps**.

| Map | Layer | Boxes per cell | Boxes |
|---|---|---|---|
| 38×38 | conv4_3 | 4 | 5,776 |
| 19×19 | conv7 (fc7) | 6 | 2,166 |
| 10×10 | conv8_2 | 6 | 600 |
| 5×5 | conv9_2 | 6 | 150 |
| 3×3 | conv10_2 | 4 | 36 |
| 1×1 | conv11_2 | 4 | 4 |
| | | **Total** | **8,732** |

**Why multi-scale?**
- **Early, high-resolution maps** (38×38) have small receptive fields, so they detect **small objects**.
- **Late, low-resolution maps** (1×1) have seen the whole image, so they detect **large objects**.
- Teacher: initial feature maps carry **local** information; final ones carry **global** information.

**Default boxes:**
- **Scale** for map k of m:

$$s_k = s_{min} + \frac{s_{max}-s_{min}}{m-1}(k-1),\quad s_{min}=0.2,\ s_{max}=0.9$$

  With m = 6: **0.20, 0.34, 0.48, 0.62, 0.76, 0.90** (as a fraction of the image size).
- **Aspect ratios** {1, 2, 3, 1/2, 1/3}, plus an extra square box of scale $\sqrt{s_k s_{k+1}}$.
  - Width $= s_k\sqrt{a_r}$, height $= s_k/\sqrt{a_r}$.
  - Why several ratios? A car seen side-on is a wide box; a pedestrian is a tall box.
- **Output per default box:** **4 offsets** (Δcx, Δcy, Δw, Δh) + **(C+1) class scores** (C classes + **background**).
  - The prediction conv for a map has **k·(C+4)** output channels (the slide writes 3×3×(4×(Classes+4))). Here k is the number of boxes per cell, and the C in "C+4" counts background as one of the classes.

**Matching (assigning ground truth):**
- Each GT box is matched to its **best-overlap** default box **and** to **every** default box with **IoU > 0.5**.
- So there are **many positives per object**, not just one.

**Imbalance and hard negative mining:**
- Out of 8,732 boxes only a handful are positive; the rest are background. Without care, the model learns to say "background".
- **Hard negative mining:**
  - Rank the negatives by **confidence loss**, i.e., the boxes the model *most wrongly* thinks contain an object.
  - Keep only enough negatives for a **neg : pos ratio of 3 : 1**.
  - This stabilises training and speeds up convergence.
- Teacher: *"hard negatives are where there is no object but the model thinks there is."*

**Loss:**

$$L = \frac{1}{N}\big(L_{conf} + \alpha\,L_{loc}\big),\quad N = \#\text{matched positives},\ \alpha = 1$$

- $L_{loc}$ = **Smooth L1** on the offsets, **positives only**.
- $L_{conf}$ = **softmax** over C+1 classes, on **positives + mined hard negatives**.
- **α plays the same role as λ** in Faster R-CNN: it **balances the two losses**.

**Results (VOC07):**

| Detector | mAP | Speed |
|---|---|---|
| **SSD300** | **72.1%** | **58 FPS** |
| SSD512 | 75.1% | — |
| Faster R-CNN | 73.2% | **7 FPS** |

**SSD ≈ Faster R-CNN accuracy at ~8× the speed.** It also beats YOLOv1 on both speed and accuracy.

## 3.3 YOLOv1 — You Only Look Once (Redmon et al., CVPR 2016) ★ deepest topic

**Reframing:**
- Detection becomes **one regression problem**: look at the **whole image once** and regress a **fixed-size tensor** encoding every box, confidence and class probability.
- **No proposals, no sliding windows.**
- **45 FPS at 63.4% VOC07 mAP**; **Fast YOLO runs at 155 FPS**.
- Because it sees the **entire image**, it encodes **global context**, so it makes **far fewer background-as-object false positives**.

### The S×S grid and the responsibility rule

- Divide the image into an **S×S grid (S = 7)**.
- **A cell is "responsible" for an object iff the object's GT centre falls inside that cell.**
- Each cell predicts **B = 2 boxes** plus **ONE set of class probabilities** (C = 20 for PASCAL VOC).
- Cells with no object centre are trained to output **low confidence** ("nothing here").
- S is a hyper-parameter. If the objects are large (zoomed-in images), a smaller S such as 3 or 4 is fine.

### What each cell predicts (★ numerical)

```
Box 1: x, y, w, h, conf   (5)
Box 2: x, y, w, h, conf   (5)
Class probs P1..P20       (20)
────────────────────────────────
per cell = 5B + C = 30   →   output = S × S × (5B + C) = 7 × 7 × 30 = 1470 values
```

- x and y are relative to the cell; w and h are relative to the image.
- **7 × 7 × 2 = 98 candidate boxes per image.**

**Why 5 values per box and not 4?** The 5th value is the **confidence** that the box contains an object.

### Confidence — precisely defined (★ numerical)

- **Box confidence** $= \Pr(\text{Object}) \times \text{IoU}^{truth}_{pred}$.
  - It is **0 if there is no object**; otherwise it measures **how well the box fits**.
- **Class-specific confidence (test time):**

$$\Pr(\text{Class}_i\mid\text{Object})\times\Pr(\text{Object})\times\text{IoU} = \Pr(\text{Class}_i)\times\text{IoU}$$

- Class probabilities are predicted **once per CELL**, so **both boxes of a cell share them**. This is the root of the "one class per cell" limitation.

**Slide example:**
- Box confidence = 0.8; P(car | obj) = 0.9, P(dog | obj) = 0.05, P(cat | obj) = 0.03.
- **Car = 0.9 × 0.8 = 0.72**, **Dog = 0.05 × 0.8 = 0.04**, **Cat = 0.03 × 0.8 = 0.024**.
- Keep the high-scoring boxes, then run **NMS**.

### Architecture

- **Input 448×448×3.**
- **24 conv layers** (GoogLeNet-inspired 1×1 reduce + 3×3 conv, no inception modules) + **2 FC layers** (FC 4096 + dropout 0.5) → reshape to **7×7×30**.
- The first ~20 conv layers are **pretrained on ImageNet at 224×224**. Then the resolution is raised to 448 for detection (fine detail matters for localisation), 4 conv + 2 FC layers are added, and everything is **fine-tuned end-to-end**.
- **Leaky ReLU (α = 0.1)** everywhere; the **final layer is linear**.
- No region proposals, so **the loss signal reaches every layer** (unlike R-CNN, where proposals got no gradient).

### The loss (sum-squared error, 4 terms) ★

$$\begin{aligned}
L =\;& \lambda_{coord}\sum_{i}\sum_{j}\mathbb{1}^{obj}_{ij}\Big[(x_i-\hat x_i)^2 + (y_i-\hat y_i)^2 + (\sqrt{w_i}-\sqrt{\hat w_i})^2 + (\sqrt{h_i}-\sqrt{\hat h_i})^2\Big] && \text{(1) localisation}\\
+& \sum_i\sum_j \mathbb{1}^{obj}_{ij}(C_i-\hat C_i)^2 && \text{(2) object confidence}\\
+& \lambda_{noobj}\sum_i\sum_j \mathbb{1}^{noobj}_{ij}(C_i-\hat C_i)^2 && \text{(3) no-object confidence}\\
+& \sum_i \mathbb{1}^{obj}_i\sum_{c}(p_i(c)-\hat p_i(c))^2 && \text{(4) class probabilities}
\end{aligned}$$

- **λ_coord = 5:** localisation errors get **higher weight**.
- **λ_noobj = 0.5:** most cells have no object, so this term is **down-weighted** to stop it overwhelming the others. This is the same imbalance idea as SSD's hard-negative mining.
- **√w, √h:** an error of 10 px matters much more on a small box than on a big one. Square roots **reduce the penalty for large boxes** relative to small ones.
- **Only one predictor is "responsible"** per object: the box with the **highest IoU** with the GT. The confidence target for that box is the IoU.
- The class loss only counts for **cells that contain an object**.

### NMS in YOLO

- There are 98 candidates, but each object should give **one** detection.
- Steps:
  1. Sort the boxes by confidence.
  2. Keep the top box.
  3. Suppress the others with IoU > threshold.
  4. Repeat.
- ★ **"NMS is not learned — it's a fixed post-processing step"** (the teacher stressed this).

### Strengths & limitations (★ 2-mark question)

| Strengths | Limitations |
|---|---|
| **Extremely fast** (45 fps; Fast YOLO 155 fps) | **Only 1 class and 2 boxes per cell**: fails when several objects share a cell (a flock of birds; cars queued front-on) |
| **Global reasoning**, so far fewer background false positives | **Coarse 7×7 grid** + heavy down-sampling: **imprecise localisation**, poor on **small objects** |
| **Generalises well** to new domains (artwork, paintings) | Struggles with **unusual aspect ratios** |
| Simple, unified, **end-to-end** trainable | **Loss treats small and large box errors similarly** despite the √ fix; **localisation is the biggest error source** |

**Teacher's industry anecdote — "stixels":**
- Instead of a grid, split the image into **vertical columns** of fixed width and regress only the **height** of the object in each column.
- Combine each column with a **depth/disparity map** to place it in 3D, which gives the **free space** a car can drive in.
- Point: YOLO is a flexible concept that people modify for their needs.

## 3.4 The YOLO lineage, v2 → v12 (what changed and WHY)

| Version | Year | Backbone | Anchors | Headline idea (remember this) |
|---|---|---|---|---|
| **v1** | 2016 | Custom 24-conv | None | Detection = one regression problem; S×S grid |
| **v2** (YOLO9000) | 2017 | **Darknet-19** | **Yes (k-means)** | **k-means anchors**, **BatchNorm** (+2% mAP), hi-res classifier, passthrough layer, **multi-scale training** |
| **v3** | 2018 | **Darknet-53** (residual) | Yes, 3 scales × 3 = 9 | **3-scale prediction (FPN-like)**, **sigmoid per class (multi-label)** |
| **v4** | 2020 | **CSPDarknet-53** | Yes | **Bag of Freebies** (Mosaic, CutMix…) + **Bag of Specials** (SPP, PANet, Mish) + **CIoU loss** |
| **v5** | 2020 | CSPDarknet (PyTorch) | Yes | Ultralytics engineering & deployment (not a paper) |
| v6 | 2022 | EfficientRep | No (default) | Re-parameterisation for industrial deployment |
| **v7** | 2022 | **E-ELAN** | Yes | **Trainable bag-of-freebies: auxiliary head** (train only) |
| **v8** | 2023 | C2f | **No** | **Fully anchor-free**, decoupled head, DFL, multi-task framework |
| v9 | 2024 | GELAN | No | Programmable Gradient Information (PGI) |
| **v10** | 2024 | Enhanced CSPNet | No | **NMS-free** (consistent dual assignment) |
| v11 | 2024 | C3k2 + C2PSA | No | Refinement + position-sensitive attention |
| **v12** | 2025 | **Area Attention + R-ELAN** | No | **Attention-centric** yet real-time |

**Pattern across the Anchors column:** none (v1) → anchors (v2–v7) → anchor-free (v8–v12).
- Anchors made learning **easier** (good starting shapes).
- But anchors **cost** hand-designed sizes and ratios, complex matching and many redundant predictions. That is why later versions dropped them.

**Three eras:**
1. **Darknet era**, v1–v4 (original authors).
2. **Ultralytics/PyTorch engineering era**, v5, v8, v11.
3. **Parallel research race on efficiency**, v6, v7, v9, v10, v12.

### YOLOv2 (2017) — "Better, Faster, Stronger"

- **Anchor boxes from k-means clustering** of the training-set GT boxes ("dimension clusters", **k = 5**).
  - Answers the class question: *"you are placing anchor aspect ratios randomly"*. Instead, learn the typical (w, h) shapes **from the data**.
- **Batch normalisation** on every conv layer: **+2% mAP**, more stable training.
- **High-resolution (448) classifier** fine-tuning before detection training.
- **Passthrough (skip) layer:** concatenates an earlier, higher-resolution feature map. A first step toward multi-scale features: **early feature maps carry local detail**.
- **Grid 13×13 at 416×416 input** (169 cells instead of 49), so better localisation of small objects.
- **Multi-scale training:** the input size changes randomly **every 10 batches (320–608)**, so one network works across resolutions.
- **YOLO9000:** joint training on classification + detection using a **WordTree** hierarchy. It detects **>9,000 categories** with only ~200 having box labels.
- **Results:** 76.8% VOC07 at **67 FPS**; 21.6% COCO.

### YOLOv3 (2018)

- **Darknet-53**: deeper, with **residual** connections (ResNet-style).
- **Predictions at 3 scales** via upsampling + concatenation (**FPN-like**):
  - 13×13 → **large** objects
  - 26×26 → **medium** objects
  - 52×52 → **small** objects
  - With **3 anchors per scale = 9 anchors** (k-means).
- **Independent sigmoid (logistic) classifiers + binary cross-entropy instead of softmax** enable **multi-label** predictions (e.g., "person" AND "woman").
  - **Softmax:** classes compete and the probabilities sum to 1. Use it for **multi-class** (exactly one label).
  - **Sigmoid per class:** each class is an independent yes/no. Use it for **multi-label** (several labels can be true at once).
  - Teacher: *"If someone asks how you train for multi-label, answer: a sigmoid for each class."*
- **Head size (COCO):** each scale outputs N×N×[3 × (4 + 1 + 80)] = **N×N×255**.
- **Total boxes at 416:** (13² + 26² + 52²) × 3 = **10,647**.
- **Result:** 57.9% AP₅₀ on COCO, on par with SSD/RetinaNet at **3–4× the speed**.

**Feature Pyramid Network (FPN):**
- Take the shrinking backbone maps, **upsample** the deep (semantic) ones and **merge** them with the earlier high-resolution ones.
- Every scale then has both **local detail and global semantics**.

### YOLOv4 (2020)

- **CSPDarknet-53 (Cross-Stage-Partial):**
  - **Split the channels into two halves.** One half goes through the conv blocks; the other half **skips** them.
  - **Concatenate** the two halves.
  - This gives fewer parameters and less compute, better gradient flow, and mixes old and new information.
- **Bag of Freebies** (raise accuracy, **no extra inference cost**; training-time only):
  - **Mosaic** augmentation: stitch **4 training images** into one. This gives more context and more small objects.
  - **CutMix:** paste a patch of one image onto another; the labels are mixed by area.
  - Label smoothing, DropBlock.
- **Bag of Specials** (small inference cost):
  - **SPP** (Spatial Pyramid Pooling): parallel max-pools 5×5, 9×9, 13×13 (stride 1, "same" padding) plus identity, concatenated. Example: 13×13×1024 → **13×13×4096**.
    - Larger receptive field, multi-scale context, **without shrinking the map**.
  - **PANet neck:**
    - **FPN's top-down path** carries semantics ("**what**") to the high-resolution maps.
    - **PAN adds a bottom-up path** carrying precise localisation ("**where**") back to the deep maps.
    - **Bidirectional fusion.** BiFPN (EfficientDet) is an improved bidirectional version.
  - **Mish** activation.
- **CIoU loss** (box regression) penalises **overlap + centre distance + aspect-ratio mismatch** together; **DIoU-NMS**.

> **Transcript correction:** in class, "CutMix" was described as overlaying two images at 30% / 70% intensity with soft labels. That is actually **MixUp**. **CutMix** cuts a **rectangular patch** from one image and pastes it into another, with labels weighted by patch area. Know both.

### YOLOv7 (2022)

- **E-ELAN** (Extended Efficient Layer Aggregation Network). The progression is CSP → ELAN → E-ELAN:
  - **Expand** the channels (1×1 conv, C → kC).
  - Run **many parallel group-conv branches**.
  - **Concatenate**, then **channel shuffle** so information mixes across groups.
  - It **controls the gradient path length** so very deep networks still train efficiently.
- **Trainable bag-of-freebies — auxiliary head:**
  - An extra head (deep supervision at a middle layer) gives **extra loss signal during training only**.
  - Only the **lead head** is used at inference, so there is **zero deployment cost**.
  - **How it differs from an ensemble:** an ensemble also uses its extra models **at test time**; the auxiliary head is **discarded after training**.
- Planned re-parameterisation.

### YOLOv8 (2023) and anchor-free detection ★

**Anchor-based vs anchor-free:**

| Anchor-based (v2–v7, SSD, Faster R-CNN) | Anchor-free (v8–v12, FCOS) |
|---|---|
| Predefine several anchors (sizes and ratios) per location; predict **offsets** from them | **No predefined boxes**. Each feature-map location is a **point**; predict the **distances to the 4 sides** (l, t, r, b) |
| Cons: anchors must be designed, complex matching, many redundant predictions, inflexible for odd shapes | Pros: simpler, fewer hyper-parameters, flexible shapes |

**Decoding** (★ numerical): for a point (x, y) with predicted distances (l, t, r, b):

$$x_1 = x - l,\quad y_1 = y - t,\quad x_2 = x + r,\quad y_2 = y + b$$

Example: point (100, 80), (l, t, r, b) = (20, 10, 30, 40) → box **(80, 70, 130, 120)**: width 50, height 50.

**Points at 640×640** (P3/P4/P5 at strides 8/16/32):

| Level | Grid | Points |
|---|---|---|
| P3 | 80×80 | 6,400 |
| P4 | 40×40 | 1,600 |
| P5 | 20×20 | 400 |
| | **Total** | **8,400** |

**Distribution Focal Loss (DFL):**
- Instead of predicting **one number** per side, predict a **probability distribution over reg_max = 16 bins**.
- The distance is the **expectation** $l = \sum_i p_i\cdot i$.
- Why? It expresses uncertainty about where the edge is (blurry or occluded edges) and gives more precise localisation.
- **Output per point:** 4 × 16 = **64** regression values + 80 class scores (sigmoid) = **144 values** (COCO).

**Decoupled head:** a separate **box branch** (DFL) and **class branch** (sigmoid), predicted at 3 scales.

**Task-Aligned Assignment:** a point becomes positive based on **both** its classification score **and** its IoU, so only the best points inside or near the object are positives.

### YOLOv10 (2024) — NMS-free

- **Consistent dual assignment:**
  - A **one-to-many head** (rich supervision, train-only) is trained **in parallel** with a **one-to-one head**.
  - **At inference, only the one-to-one head runs, so there is no NMS at all.**
  - It is the **first mainline YOLO to remove NMS**, which every detector since SSD needed.
- **Efficiency redesign:** lightweight classification head, decoupled down-sampling, **PSA** (partial self-attention).
- **YOLOv10-B:** ~46% lower latency and 25% fewer parameters than YOLOv9-C at comparable accuracy.

### YOLOv12 (2025) — attention-centric

- The **first mainline YOLO built around attention**. Attention used to be considered too slow for real time.
- **Area Attention (A²):**
  - **Split the feature map into a few large areas** and attend within each.
  - This keeps a large receptive field but **cuts the O(N²) cost** of global attention.
  - Images are mostly **local**, so full global attention isn't needed.
- **R-ELAN:** ELAN + **residual connections with scaling** to stabilise deep attention networks.
- **Anchor-free with DFL**; FlashAttention; MLP ratio 1.2; 7×7 depthwise convs.
- Input 640×640; predictions at 80×80, 40×40, 20×20.
- **v12 still uses NMS** (unlike v10). Watch for this trap in multi-select questions.

## 3.5 Evaluating detectors ★★ (numerical + descriptive goldmine)

### Why "accuracy" doesn't work for detection

Each prediction raises two questions at once: **is the box right?** (continuous) and **is the class right?** A detector can fail in **4 ways**:
1. Right class, **wrong box**
2. Right box, **wrong class**
3. **No box** at all (missed)
4. A **confident box where nothing exists**: this one has no classification analogue

### IoU — Intersection over Union

$$\text{IoU}(A,B) = \frac{\text{area}(A\cap B)}{\text{area}(A\cup B)} = \frac{\lvert A\cap B\rvert}{\lvert A\rvert + \lvert B\rvert - \lvert A\cap B\rvert}\in[0,1]$$

- The intersection is subtracted in the union because it would otherwise be **counted twice**.
- IoU is the same as SSD's "Jaccard overlap".
- Slide examples: 0.82 near-perfect, 0.41 partial, 0.13 barely touching, 0.00 none.
- **Used for:** matching (anchors and GT), **NMS**, deciding TP/FP, mAP.

### TP / FP / FN (at an IoU threshold, usually 0.5)

| | Definition |
|---|---|
| **TP** | Prediction with IoU ≥ threshold with a GT object **and** the correct class |
| **FP** | Prediction with IoU < threshold with every object, **or a duplicate** of an already-matched GT |
| **FN** | A GT object with **no** matching prediction (missed) |
| **TN** | **Doesn't exist** in detection: you can't count all the boxes correctly *not* drawn. That is why we use P/R/AP, not accuracy |

### Precision, Recall, F1

$$\text{Precision} = \frac{TP}{TP+FP}\;\;(\text{of what I predicted, how much was right?})\qquad \text{Recall} = \frac{TP}{TP+FN}\;\;(\text{of what was really there, how much did I find?})$$

$$F_1 = \frac{2PR}{P+R}\quad(\text{harmonic mean, at ONE confidence threshold})$$

- **Trivial extremes:**
  - Boxes everywhere → perfect recall, terrible precision.
  - Predict only when ultra-sure → perfect precision, terrible recall.
- **Raising the confidence threshold:** FP ↓ (TP may drop a little), so **precision ↑ and recall ↓**.
  - Teacher's numbers: P = 5/8 = 0.625 becomes 4/5 = 0.80 after raising the threshold.
- **The IoU threshold** also changes TP, FP and FN. A **stricter IoU** gives fewer TPs, so AP drops.
- **F1** is used when you must ship **one operating point**. AP answers "how good *could* this model be"; F1 answers "how good is it *at this setting*".

**Which to optimise? (Teacher's interview question — ★ likely case-study or descriptive question):**

| Scenario | Optimise | Why |
|---|---|---|
| **Cancer screening / tumour detection** | **Recall** | Missing a cancer (FN) can kill; a false alarm only means a doctor double-checks |
| **Pedestrian detection** (braking) | **Recall** | A missed person (FN) is catastrophic |
| **Autonomous lane change** ("is it safe?") | **Precision** | A wrong "safe" call hits someone. A missed lane-change opportunity costs nothing: "there will be many opportunities to overtake, but no second chance if you kill someone" |
| **Spam filter / auto-reject a good product** | **Precision** | False positives are costly |
| Both matter | **F1** | Balance |

The teacher's scenario: precision is 92% but recall is 60% in a life-critical car system. You can't retrain, only move the threshold. **Decide based on which error is fatal.**

### Precision–Recall curve → AP → mAP

**Building the PR curve (one class at a time):**
1. Sort **all** predictions for the class (across the test set) by **confidence, highest first**.
2. Walk down the list one at a time, marking each **TP or FP** (IoU ≥ 0.5, each GT matched at most once).
3. After each prediction, compute the **cumulative** precision and recall.
4. Plot P (y-axis) against R (x-axis).
   - Precision starts near 1 (confident predictions tend to be right).
   - As lower-confidence predictions are accepted, recall rises and precision usually falls.

**Interpolated precision:** $p_{interp}(r) = \max_{\tilde r\ge r} p(\tilde r)$, the **maximum precision at any recall ≥ r**. It removes the zig-zags.

**AP = area under the interpolated PR curve, for one class.**
- **11-point (PASCAL VOC 2007):** average $p_{interp}$ at r = 0, 0.1, …, 1.0: $AP_{11} = \frac{1}{11}\sum p_{interp}(r)$.
- **All-point (VOC 2010+, the basis for COCO):** the exact area under the monotone envelope, $\sum (r_{k}-r_{k-1})\,p_{interp}(r_k)$.

**★ Slide worked example:** class "car", 5 GT cars, 5 detections.

| Rank | Conf | TP/FP | cum TP | cum FP | Precision | Recall (÷5) |
|---|---|---|---|---|---|---|
| 1 | 0.95 | TP | 1 | 0 | 1.00 | 0.20 |
| 2 | 0.90 | TP | 2 | 0 | 1.00 | 0.40 |
| 3 | 0.85 | FP | 2 | 1 | 0.67 | 0.40 |
| 4 | 0.80 | TP | 3 | 1 | 0.75 | 0.60 |
| 5 | 0.70 | FP | 3 | 2 | 0.60 | 0.60 |

- **Max recall = 3/5 = 0.60.** Two cars are never found.
- **Interpolated precision at each recall level:**
  - r = 0.0, 0.1, 0.2, 0.3, 0.4 → **1.00** (five values)
  - r = 0.5, 0.6 → **0.75** (two values)
  - r = 0.7, 0.8, 0.9, 1.0 → **0** (four values; no predictions reach those recalls)
- **$AP_{11} = (5\times1.00 + 2\times0.75 + 4\times0)/11 = 6.5/11 = \mathbf{0.591}$**
- **All-point AP** = 0.4 × 1.00 + 0.2 × 0.75 = **0.55**

> **Slide typo:** the slide's table shows 0.75 at r = 0.4. It should be **1.00**, because the point (R = 0.40, P = 1.00) exists. The slide's final answer, 0.591, uses 1.00 and is correct.

**"Why does precision go UP from rank 3 to rank 4 (0.67 → 0.75)?"** Rank 4 is a TP, so the numerator and denominator both increase by 1: 2/3 → 3/4. Adding a TP always raises precision when precision is below 1.

**mAP = mean of per-class AP:**

$$\text{mAP} = \frac{1}{N_{classes}}\sum_c AP_c$$

- Slide example: car 0.87, pedestrian 0.71, bicycle 0.58, sign 0.62 → (0.87 + 0.71 + 0.58 + 0.62)/4 = 2.78/4 ≈ **0.70**.
- Averaging **per class** stops common classes from drowning out rare ones.

**COCO metric suite:**

| Metric | Meaning |
|---|---|
| **AP@[.5:.95]** ("AP") | mAP averaged over **10 IoU thresholds** 0.50, 0.55, …, 0.95. The standard benchmark; rewards precise boxes |
| **AP₅₀** | IoU ≥ 0.5 (lenient, ≈ the VOC metric) |
| **AP₇₅** | IoU ≥ 0.75 (strict) |
| **AP_S / AP_M / AP_L** | Small (< 32² px) / medium / large (> 96² px) objects; diagnoses *where* a model struggles |

### Dice vs IoU

$$\text{Dice} = \frac{2\lvert A\cap B\rvert}{\lvert A\rvert+\lvert B\rvert} = \frac{2\,\text{IoU}}{1+\text{IoU}}\quad\ge\ \text{IoU}$$

- Slide values: IoU 0.3 → Dice 0.46; 0.5 → **0.67**; 0.7 → 0.82; 0.9 → 0.95. They agree only at 0 and 1.
- **IoU** dominates **detection** (box matching, mAP, NMS).
- **Dice** dominates pixel-wise **segmentation, especially medical imaging**:
  - It counts the intersection twice, so it is **more forgiving for small foreground regions**.
  - Teacher: *"pixels are already very small, so give more weight to overlap."*

### IoU-based losses (YOLOv4+)

**GIoU / DIoU / CIoU** are box-regression **losses**, not just metrics. **Plain IoU gives zero gradient when the boxes don't overlap.**
- **GIoU** adds a penalty using the smallest enclosing box, so it still gives a signal when boxes are disjoint.
- **DIoU** adds a **centre-distance** term.
- **CIoU** adds centre distance **+ aspect ratio**: $L_{CIoU} = 1 - IoU + \frac{\rho^2(b,b^{gt})}{c^2} + \alpha v$.

### Evaluation at a glance (slide summary)

| Metric | What it answers |
|---|---|
| IoU | How well do two boxes overlap? (matching, NMS, TP/FP) |
| Precision / Recall | Correct of predicted? Found of real? |
| PR curve | How does the trade-off move with confidence? |
| AP | Area under one class's PR curve |
| mAP | AP averaged over classes; the headline number |
| COCO AP@[.5:.95] | mAP averaged over 10 IoU thresholds |
| AP₅₀ / AP₇₅ / AP_S,M,L | Diagnose at a threshold or by object size |
| Dice | Overlap weighted toward small regions (segmentation) |
| GIoU / DIoU / CIoU | Overlap-aware regression **loss** |
| FPS / latency | How fast, on which hardware |
| F1 | P/R balance at one deployed threshold |

## 3.6 Session 7 quick self-test

1. YOLOv1 with S = 13, B = 3, C = 80 → 13 × 13 × (15 + 80) = **13 × 13 × 95**.
2. SSD default boxes on a 10×10 map with 6 boxes per cell → **600**.
3. Box confidence 0.6 and P(dog | obj) = 0.7 → class-specific confidence = **0.42**.
4. Why λ_noobj = 0.5? Most cells are empty, so their confidence errors would swamp the gradient. Down-weighting them balances the loss.
5. Softmax vs sigmoid head? Multi-class (one label) vs **multi-label** (independent per-class yes/no).
6. Which YOLO first removed NMS? **v10** (one-to-one head). Does v12 use NMS? **Yes.**
7. IoU = 0.6 → Dice = 1.2 / 1.6 = **0.75**.


---

# PART 4 — Session 8: Object Detection III — Transformers & DETR

**Story of the session:**
1. Transformer (from NLP)
2. Self-attention maths
3. Why vision is hard for transformers
4. ImageGPT (pixels as tokens)
5. ViT (patches as tokens)
6. **DETR** (detection as set prediction)
7. Metrics and results
8. Tracking

## 4.1 Transformer building blocks (from "Attention Is All You Need", Vaswani et al. 2017)

- **Encoder:** a stack of alternating **self-attention** and **MLP (feed-forward)** blocks, with **residual connections** and **LayerNorm**.
- **Decoder:** used when the **output space differs from the input space**, e.g., translation. It has **masked self-attention**, then **cross-attention** to the encoder, then the MLP.
- **Tokens:**
  1. Each word or patch becomes a fixed-size vector (e.g., 768-d).
  2. Self-attention has no idea of order, so a **positional encoding** is **added** to each token.

### Self-attention — the maths (numerical / descriptive favourite)

Input $X$ is $n$ tokens of dimension $m$ (or $d$), so $X\in\mathbb{R}^{n\times d}$.

**Step 1: three learned projections of every token.**

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V \qquad (W_\ast \in \mathbb{R}^{d\times d_k})$$

**Step 2: the attention output.**

$$\text{Attention}(Q,K,V) = \underbrace{\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)}_{n\times n\ \text{attention map (rows sum to 1)}}\;V$$

**Shapes:**

| Matrix | Shape |
|---|---|
| Q, K, V | n × d_k |
| QKᵀ | **n × n** |
| softmax(·)·V | **n × d_k** |

**What each part means:**
- **Query** = "what am I looking for?"
- **Key** = "what do I contain?"
- **Value** = "what I give if you attend to me".
- Row *i* of the attention map tells you **how much token *i* attends to every token** (including itself).
- **Self-attention is a global, permutation-invariant (set) operation.** It aggregates information from **all** tokens, so it needs position embeddings to remember order.

**Why divide by √d_k? (The teacher refused to continue until someone answered — high-probability 2-mark question.)**
- Suppose the q and k components have mean 0 and variance 1 (as LayerNorm and good initialisation give you).
- Their dot product $q\cdot k = \sum_{i=1}^{d_k} q_i k_i$ is a sum of $d_k$ terms, so its **variance grows to $d_k$**.
- Large dot products push the softmax into a **very peaked (saturated)** region where the **gradients vanish**.
- Dividing by the **standard deviation √d_k** brings the variance back to 1.

**Multi-head attention:**
- Use **h different sets** of $W_Q, W_K, W_V$, each projecting to a **smaller** $d_k = d/h$ (e.g., 512 / 8 = 64).
- Compute attention separately in each head, **concatenate**, then project with $W^O$:

$$\text{MultiHead} = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O$$

- **Why?** Each head can focus on a **different kind of relationship or feature**. The teacher compared it to an **ensemble**: the same input passed through several different "views", then combined.

**Numerical:** n = 10 tokens, d = 768, h = 12 heads.
- $d_k$ = 768 / 12 = **64**.
- Each head's Q, K and V are **10 × 64**.
- Each head's attention map is **10 × 10**.
- Concatenating the heads gives **10 × 768**, and $W^O$ (768 × 768) keeps it **10 × 768**.

**Positional encoding (fixed, sinusoidal):**

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

- Positional encodings have the **same dimension** as the tokens and are **added** to them.
- They can be **learned** (ViT, DETR queries) or **fixed** (sinusoidal).

**Transformer layer (the original post-norm version):**
1. X ← X + MultiHeadSelfAttn(X)
2. X ← LayerNorm(X)
3. x ← x + MLP(x) (per token)
4. X ← LayerNorm(X)

**Pre-norm (ImageGPT, ViT, modern LLMs):** **LayerNorm moves before** the self-attention and FFN blocks, inside the residual path. The original position trained less stably. (Teacher: *"now in every paper the position of layer norm is different from Attention Is All You Need."*)

### LayerNorm vs BatchNorm (the teacher's "very good interview question")

| | **BatchNorm** | **LayerNorm** |
|---|---|---|
| Normalises over | The **batch**, for each feature/channel (one dimension across many samples) | The **features of a single token/sample** (within itself) |
| Depends on batch size? | **Yes** (needs batch statistics; behaves differently in train and test) | **No** (works with batch size 1 and variable-length sequences) |
| Typical use | **CNNs / images** | **Transformers / text** |
| Formula | per channel over N (and H, W) | $y = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x]+\epsilon}}\gamma + \beta$ **per token** |

**Why LayerNorm for transformers:**
- Different tokens can have very different magnitudes.
- Without LayerNorm, one high-magnitude embedding would **dominate the dot products** in attention.
- LayerNorm puts each token at mean 0 and variance 1, so attention compares **directions, not magnitudes**.

### Cross-attention vs self-attention

| | Q from | K, V from |
|---|---|---|
| Self-attention | the same sequence | the same sequence |
| **Cross-attention** | **the decoder** | **the encoder output** |

- The attention matrix is $N_{dec} \times N_{enc}$; the output has the **decoder's length**.
- **Masked self-attention** in the decoder means a token cannot see future tokens (autoregressive, like GPT: "My name is …").

**BERT vs GPT:**
- **BERT** is bidirectional, trained with **Masked Language Modelling** (15% of tokens masked) plus next-sentence prediction.
- **GPT** is autoregressive (predicts the next token).

**How many blocks?**
- The original Transformer repeats the block **6 times in the encoder and 6 in the decoder**.
- LLMs repeat it ~100+ times.

## 4.2 Why transformers struggle with images

- Self-attention cost is **quadratic** in sequence length: O(n²).
- With 1 token per pixel, a 224×224 image gives **50,176 tokens (≈ 50k)**, so the attention map would have ~2.5 billion entries. It gets even worse at high resolution or for video.
- There is also **much more data per sample** than in text.
- Transformers lack CNN **inductive biases** (locality, translation equivariance), so they **need much more data**.

## 4.3 ImageGPT (OpenAI, 2020) — pixels as tokens

- To shrink the context, it **reduced the resolution to 32×32, 48×48 or 64×64**.
- It **reduced the colour palette** from 24-bit (3 × 8) to a **9-bit palette (512 colours)** by **k-means clustering** of RGB values (k = 512).
- It **linearised** the image into a 1-D pixel sequence and trained it two ways:
  - **Autoregressive** (predict the next pixel)
  - **BERT-style** (predict masked pixels)
- It used **pre-norm** layers (LayerNorm before attention and FFN).
- It could **complete the bottom half of an image** given the top half.
- **Pro:** it showed that transformers learn strong representations with **unsupervised** training.
- **Con:** it only worked on **tiny, down-scaled images (32–64 px)**, so a lot of information is lost.

**Why learn to regenerate an image at all? (Teacher's autoencoder point.)**
- The aim is not the copy; it is the **latent representation z₀** learned along the way.
- z₀ is a compressed "world knowledge" representation that can be reused for many tasks.
- Teacher's example: Google Photos-style compressed storage.

## 4.4 Vision Transformer — ViT (Dosovitskiy et al.; "An Image is Worth 16×16 Words")

**It fixes ImageGPT's resolution limit by using PATCHES as tokens.**

**ViT recipe — memorise these 7 steps:**
1. Divide the image into **P×P patches**.
2. Make a sequence of length **N = HW / P²**.
3. **Flatten** each patch (P·P·3 values) and map it to **D dimensions** with a **trainable linear projection**.
4. Add a **learned 1-D position embedding**.
5. Prepend a **learnable [class] token** (token "0\*").
6. Pass the sequence through a **multi-layer Transformer encoder** (pre-norm, **encoder only**).
7. Put an **MLP head** on the **[class] token's output** to predict the class.

**Numerical (very likely):** 224×224×3 image, P = 16.
- Patches per side = 224 / 16 = 14, so **N = 14² = 196 patches**.
- Each patch = 16 × 16 × 3 = **768 values**. That equals ViT-Base's D = 768.
- Sequence length with [CLS] = **197**.
- Attention map per head = **197 × 197**.
- With P = 32: 7 × 7 = **49 patches** (+1 = 50).

**Datasets:**

| Dataset | Classes | Images |
|---|---|---|
| ImageNet-1k (ILSVRC-2012) | 1k | 1.3M |
| ImageNet-21k | 21k | 14M |
| JFT | 18k | 303M |

**Model variants:**

| Model | Layers | Hidden D | MLP size | Heads | Params |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | **86M** |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

- The **MLP size is always 4 × D**.
- **Naming:** **ViT-L/16** = Large variant with **16×16 patches**.

**Key results and claims (slide):**
- Reaches **88.55% ImageNet top-1**, but **performs poorly when trained only on ImageNet**. It lacks CNN inductive bias.
- It is **competitive only when pre-trained on very large datasets (14M–300M images)**.
- The slide's provocative line: **"Large-scale training trumps inductive bias???"**
- The slide's caveat: be wary of claims made with different datasets and compute.
- **Teacher's domain-shift example:** a ViT trained only on pinhole-camera images **fails on fisheye images** (a 180° field of view, distorted at the corners) because it never saw that distribution.

> **Transcript correction:** the lecture said "ViT-Base has 6 encoder + 6 decoder layers". That is the original **Transformer** (6 + 6). **ViT is encoder-only; ViT-Base has 12 encoder layers.**

## 4.5 DETR — DEtection TRansformer (Carion et al., Facebook, 2020) ★ core of the session

**Big idea:** treat detection as **direct set prediction**.
- **Always output a fixed set of N predictions**. N is much larger than the number of objects; the paper uses **N = 100**.
- Use **bipartite (Hungarian) matching** to pair predictions with ground-truth objects.
- So DETR needs **no anchors, no region proposals and no NMS**. It is end-to-end.

### Architecture (four blocks)

```
Image 3×H0×W0
 → CNN backbone (ResNet-50)           → f ∈ R^{C×H×W},  C = 2048,  H = H0/32, W = W0/32
 → 1×1 conv: reduce C → d (e.g., 256) → z0 ∈ R^{d×H×W}
 → flatten → d × HW sequence  (+ positional encodings at every attention layer)
 → Transformer ENCODER (self-attention + FFN): global reasoning over all image features
 → Transformer DECODER: input = N learned "object queries" (d×N);
        K, V = encoder output; transforms N queries → N output embeddings IN PARALLEL (non-autoregressive)
 → Shared FFN heads per query:
        • box: 3-layer MLP (ReLU, hidden d) → normalised (cx, cy, w, h)
        • class: linear + softmax over classes + "∅ no object"
```

**Shape numerical:** input 800 × 1216.
- Backbone output: 2048 × 25 × 38 (800/32 = 25; 1216/32 = 38).
- After the 1×1 conv: 256 × 25 × 38.
- Flattened: **950 tokens × 256**.
- Encoder attention map: **950 × 950**.
- Decoder: **100 queries × 256**, giving **100 predictions**.

**Why the 1×1 conv?** It **reduces the channels** (2048 → d) cheaply by mixing across channels at each pixel. Fewer channels make the transformer cheaper.

**Object queries:**
- **N learned embeddings** (learned positional encodings), added at every decoder attention layer.
- Each query learns to "ask for" an object in some region or size. After training, each one specialises.

**Auxiliary decoding loss:**
- The prediction heads (shared weights) and the **Hungarian loss are applied after EVERY decoder layer**, not just the last.
- This gives **deep supervision**: every layer gets a direct gradient signal, so training is faster and more stable.
- Teacher: *"you're enforcing from the initial layers itself that I have to predict the boxes."*

### Set prediction loss — Hungarian matching (likely 2-mark / multi-select)

**Step 1 — Matching.**
- Pad the ground truth with ∅ ("no object") up to N.
- Find the **one-to-one assignment** σ that **minimises the total matching cost** using the **Hungarian algorithm**:

$$\hat\sigma = \arg\min_{\sigma\in\mathfrak{S}_N}\sum_i^N \mathcal{L}_{match}(y_i, \hat y_{\sigma(i)}),\qquad \mathcal{L}_{match} = -\mathbb{1}_{c_i\ne\varnothing}\,\hat p_{\sigma(i)}(c_i) + \mathbb{1}_{c_i\ne\varnothing}\,\mathcal{L}_{box}(b_i,\hat b_{\sigma(i)})$$

- The matching cost uses **class probability AND box similarity**.

**Step 2 — Hungarian loss on the matched pairs:**

$$\mathcal{L}_{Hungarian} = \sum_{i=1}^N\Big[-\log \hat p_{\hat\sigma(i)}(c_i) + \mathbb{1}_{c_i\ne\varnothing}\,\mathcal{L}_{box}(b_i, \hat b_{\hat\sigma(i)})\Big]$$

$$\mathcal{L}_{box} = \lambda_{iou}\,\mathcal{L}_{GIoU} + \lambda_{L1}\,\lVert b_i - \hat b_{\sigma(i)}\rVert_1$$

**Facts to remember:**
- Matching uses the **probability** $-\hat p$; the loss uses the **log-probability** $-\log\hat p$ (NLL).
- The **box loss counts only for real objects** (not ∅).
- **Why L1 + GIoU?**
  - L1 alone **depends on box scale**: big boxes give big errors.
  - **GIoU is scale-invariant**, so combining them balances small and large boxes.
- **Why no NMS?** One-to-one matching **punishes duplicates**. A second prediction on the same object is matched to ∅ and trained to say "no object". So the network learns to output **exactly one box per object**.

**At inference:**
- DETR always gives **100 boxes**.
- Most are classified as "no object"; only a few carry real classes.
- Keep the predictions whose class is not ∅ and whose score is above a threshold.

### DETR results (slide table — COCO)

| Model | GFLOPs / FPS | Params | AP | AP₅₀ | AP_S | AP_L |
|---|---|---|---|---|---|---|
| Faster R-CNN-FPN+ (strong baseline) | 180 / 26 | 42M | 42.0 | 62.1 | **26.6** | 53.4 |
| **DETR** (R50) | **86 / 28** | 41M | **42.0** | 62.4 | 20.5 | **61.1** |
| DETR-DC5-R101 (best) | 253 / 10 | 60M | **44.9** | 64.7 | 23.7 | **62.3** |

The "+" baselines got longer training, the same random-crop augmentation and GIoU added to the box loss, to make the comparison fair.

**Takeaways (exam gold):**
1. DETR **matches** a well-tuned Faster R-CNN (AP 42.0), with **far simpler pipeline** (no anchors, no NMS, no hand-designed components).
2. It is **much better on LARGE objects** (AP_L 61.1 vs 53.4), thanks to the encoder's **global self-attention** over the whole image.
3. It is **worse on SMALL objects** (AP_S 20.5 vs 26.6): it uses a single low-resolution (1/32) feature map with no FPN.
4. **It trains slowly** (the paper uses ~500 epochs vs ~36 for Faster R-CNN). Hungarian matching is unstable early in training.
5. **Compute:** transformers are compute-hungry, and FPS is far below YOLO (~140+ fps). The teacher's point: at 60 fps input, a slow detector must **skip frames**.

**Fixes that came later** (modern approaches):
- **Deformable DETR:** attends to a few sampling points and uses multi-scale features, giving faster convergence and better small-object AP.
- **DINO / DN-DETR:** denoising queries.
- **RT-DETR:** a real-time DETR.

## 4.6 Detection metrics as presented in Session 8 (recap; full treatment in Part 3)

- Precision = TP / (TP + FP): correct predictions out of all predictions.
- Recall = TP / (TP + FN): correct predictions out of all ground-truth objects.
- AP = **area under the PR curve**, using **interpolated precision = max precision to the right**.
- COCO notation:
  - **AP** = mean over IoU thresholds 0.50:0.05:0.95
  - **AP₅₀** = IoU ≥ 0.5 counts as a TP
  - **AP₇₅** = stricter, IoU ≥ 0.75
  - **AP_S / AP_M / AP_L** = small (< 32² px) / medium / large (> 96² px) objects

## 4.7 Two-stage vs One-stage vs DETR (syllabus item — make this table yours)

| Aspect | **Two-stage** (Faster R-CNN) | **One-stage** (YOLO, SSD) | **DETR** (transformer) |
|---|---|---|---|
| Pipeline | Proposals (RPN) → RoI pool → classify + refine | **Single pass**: dense grid/anchors → class + box | CNN → encoder → decoder with **object queries** → set |
| Anchors? | Yes | Yes (v2–v7, SSD) / **no** (YOLOv8+, FCOS) | **No** |
| NMS? | Yes | Yes (YOLOv10 removed it) | **No** (one-to-one matching) |
| Label assignment | IoU thresholds (0.7 / 0.3) | Grid cell / best IoU / task-aligned | **Hungarian bipartite matching** |
| Speed | Slow (5–7 fps originally) | **Fastest** (45–150+ fps) | Moderate (~28 fps R50); heavy compute |
| Accuracy | High; good small objects | Was lower, now state-of-the-art | High; **best large objects**, weak small |
| Training | Multi-stage / complex | Simple, end-to-end | Simple but **slow convergence** |
| Context | Local (per RoI) | Mostly local (YOLO sees the whole image) | **Global** (self-attention) |
| Use when | Accuracy is critical, offline | **Real-time, edge devices** | Want a simple pipeline, large objects, research |

**Mnemonic: "Propose-then-refine / Look once / Ask N questions."**

## 4.8 Modern detection approaches (overview — name + one line)

| Family | Examples | One-liner |
|---|---|---|
| Modern YOLO | YOLOv8 → v12 | Anchor-free, decoupled heads, DFL; v10 is **NMS-free**; v12 adds **area attention** |
| DETR descendants | **Deformable DETR, DINO, RT-DETR** | Fix DETR's slow convergence and small-object weakness; RT-DETR runs in real time |
| Open-vocabulary | **Grounding DINO, OWL-ViT, YOLO-World** | Detect *any* object named in a **text prompt** (vision + language) |
| Foundation models | **SAM / SAM 2, DINOv2/v3** | Promptable segmentation; strong general features; used for **auto-labelling** |
| Tracking | **ByteTrack, SORT/DeepSORT** | Detection + **Kalman filter** + association, giving consistent IDs over frames |

## 4.9 Practical detector analysis (how to choose — case-study gold)

Ask these questions in order:
1. **Latency budget?** Real-time on an edge device (car, drone, phone) → YOLO-family / RT-DETR. Offline or cloud batch → Faster R-CNN / DINO.
2. **Object size?**
   - **Small objects** (drone imagery, far pedestrians): use FPN, higher input resolution, tiling; avoid plain DETR.
   - **Large objects**: DETR does well.
3. **Crowding/occlusion?** Dense scenes break YOLOv1's "2 boxes per cell" limit and cause NMS to suppress true overlapping objects. Lower the NMS threshold carefully or use Soft-NMS / NMS-free models.
4. **Data size?** Small datasets → a **pretrained** backbone + fine-tuning (transfer learning). Transformers need lots of data.
5. **Classes known in advance?** If not → open-vocabulary detectors.
6. **Which error is costly?** Missing a pedestrian (FN) → tune for **recall**. False alarms are costly → tune for **precision**.
7. **Report:** AP / AP₅₀ / AP_S, **FPS, GFLOPs, #params**. (The DETR table reports exactly these.)

**Reading a results table:**
- GFLOPs = compute per image.
- FPS = speed.
- #params = model size (memory).
- Compare AP_S and AP_L to see *where* each model wins.

## 4.10 Session 8 quick self-test

1. Shape of QKᵀ for 50 tokens → **50 × 50**.
2. Why √d_k? → the dot product's variance grows with d_k; scaling keeps the softmax out of saturation and keeps gradients alive.
3. Where do Q, K and V come from in DETR's decoder cross-attention? → **Q from the object queries** (decoder); **K, V from the encoder output**.
4. Why doesn't DETR need NMS? → **one-to-one Hungarian matching** trains duplicates to predict ∅.
5. Why does DETR do worse on small objects? → a single coarse 1/32 feature map with no multi-scale FPN.
6. ViT with a 384×384 image and P = 16 → 24² = **576 patches** (577 with CLS).


---

# PART 5 — Session 9: Video Understanding (what was actually taught)

> The syllabus calls Session 9 "Discussion on Quiz + Project and Practicals", but the slides and lecture were **"Lecture 9: Video Understanding"**. Assume this content is examinable. It reuses ideas from the whole module: CNNs, receptive fields, attention, and ViT.

## 5.1 Video = 2D + Time

- A video is a **sequence of images** → a **4D tensor `T × 3 × H × W`**. 3D CNNs write it channels-first: **`3 × T × H × W`**.
- **Images → recognise objects** (dog, cat, truck). **Videos → recognise actions** (running, jumping, swimming).
- Videos are about **30 fps**.

**Videos are BIG** (uncompressed, 3 bytes per pixel):

| Resolution | Size per minute |
|---|---|
| SD 640×480 | ≈ **1.5 GB / min** (640·480·3 × 30 × 60 = 1.66 GB) |
| HD 1920×1080 | ≈ **10 GB / min** (1920·1080·3 × 30 × 60 = 11.2 GB) |

**Class numerical:** a 30-second greyscale video at 30 fps with 256×256 frames.
- Frames: 30 s × 30 fps = **900 frames**.
- Tensor: **900 × 1 × 256 × 256** (keep the channel dim = 1).
- Values: 900 × 65,536 = **58,982,400** (≈ 59 MB at 1 byte per value).

**The fix: train on short clips with low fps and low resolution.** Example from the slide: T = 16, H = W = 112 → 3.2 s at 5 fps, and 16 × 3 × 112 × 112 = 602,112 bytes ≈ **588 KB**.

**Train vs test:**
- **Train** on short, low-fps clips.
- **Test** by running the model on many overlapping clips of the long video and **averaging the predictions**.

**Industry practice the teacher described:**
- **Drop near-duplicate frames.** A car stuck at a signal for an hour gives the same scene over and over: that wastes storage and adds no variety.
- **Auto-tag clips** with metadata: `start_time, end_time, tag` (e.g., "traffic sign present"). You can then query them with SQL instead of watching hours of footage.

## 5.2 Video classification architectures (from simple to smart)

| # | Approach | How | Key phrase |
|---|---|---|---|
| 1 | **Single-frame CNN** | Run a 2D CNN on each frame independently, then **average the predicted probabilities** (or vote) | "Often a **very strong baseline**" |
| 2 | **Late fusion (FC)** | 2D CNN per frame gives `T×D×H'×W'`. **Flatten + concatenate** to `TDH'W'`, then MLP, then C scores | Combine high-level appearance of each frame |
| 3 | **Late fusion (pooling)** | 2D CNN per frame, then **average-pool over space AND time** to `D`, then Linear, then C | Far fewer parameters than #2 |
| 4 | **Early fusion / 3D CNN** | **3D conv + 3D pooling** throughout; activations are `D×T×H×W` | **Slowly** fuses temporal information across the network |
| 5 | **Two-stream** | Spatial stream on an RGB frame **+** temporal stream on stacked optical flow; fuse the class scores | Separates **appearance** and **motion** |
| 6 | **I3D** | **Inflate** a pretrained 2D CNN into 3D | Reuse ImageNet weights |
| 7 | **Video transformers** | Divided attention, local windows (Video Swin), tubelets (ViViT) | 2021–2026 era |

**Weakness of late fusion:** motion is only combined at the very end, so the network **cannot easily capture low-level motion** (e.g., an edge moving left to right between frames). 3D CNNs fix this by mixing time from the first layers.

**Timeline on the slide:** **2014–2021 was the era of 3D CNNs + RNNs. 2021–2026 is the era of Transformers.**

## 5.3 3D convolution (numerical-friendly)

**2D recap:** a `3×32×32` image and **6 filters of `3×5×5`** (plus a 6-dim bias) give **6 × 28 × 28** (32 − 5 + 1 = 28).

**3D:**
- Input: T = 6 frames, `T×3×32×32`.
- **One filter = `2×3×5×5`.** It now has a **time dimension** (here 2 frames).
- The filter slides in **x, y and t**. Stride works in time too.
- **5 filters (`5×2×3×5×5`)** plus a 5-dim bias give **5 activation maps, each `T'×28×28`**, stacked as **`T'×5×28×28`**.
- T' = (6 − 2)/1 + 1 = **5** (stride 1, no padding).

**General formula (apply it per dimension):**

$$T' = \left\lfloor\frac{T + 2p_t - k_t}{s_t}\right\rfloor + 1,\quad H' = \left\lfloor\frac{H + 2p - k}{s}\right\rfloor + 1$$

**Parameter count** for one 3D conv layer: `C_out × (C_in × k_t × k_h × k_w) + C_out`. For the example: 5 × (3·2·5·5) + 5 = 5 × 150 + 5 = **755**.

## 5.4 Optical flow and two-stream networks

**Optical flow** is a displacement field between frames $I_t$ and $I_{t+1}$:

$$F(x,y) = (dx, dy), \qquad I_{t+1}(x+dx,\, y+dy) = I_t(x,y)$$

- It tells you **where each pixel moves** in the next frame and highlights **local motion**.
- You get **two maps: horizontal flow `dx` and vertical flow `dy`**.
- It is computed with a classical algorithm (gradient-based, like SIFT/HOG gradients) or a **faster neural-network approximation**. The teacher described a nearest-neighbour patch match: find where a pixel's patch went in the next frame, which gives dx and dy.

**Two-Stream Network** (Simonyan & Zisserman, 2014):

```
Single RGB frame  (3 × H × W)            ──► Spatial-stream ConvNet  ──┐
                                                                       ├─► class-score fusion
Stacked optical flow ([2·(T−1)] × H × W) ──► Temporal-stream ConvNet ──┘
```

- **Spatial stream** answers *what* objects/scene are present (appearance).
- **Temporal stream** answers *what motion* is happening (e.g., moving left to right).
- **Why 2·(T−1) channels?** (The teacher asked this in class.)
  - **T − 1:** flow is computed between consecutive frames, so T frames give T − 1 flow fields. The first frame has no previous frame.
  - **× 2:** each flow field has two channels, **dx (horizontal) and dy (vertical)**.
  - Example: T = 11 frames → 10 flows → **20 input channels**.
- Early fusion happens inside the temporal stream: its first 2D conv sees all the flow images at once.

## 5.5 I3D — Inflating 2D networks to 3D (Carreira & Zisserman, 2017)

**Question it answers:** can we **reuse image architectures and their pretrained weights** for video?

1. Take a 2D CNN architecture (e.g., Inception or ResNet).
2. Replace each 2D `K_h×K_w` conv/pool with a 3D `K_t×K_h×K_w` version.
3. **Initialise from the 2D weights: copy the 2D kernel K_t times along time and divide by K_t.**
4. For a "constant" video (the same frame repeated), the 3D net then gives **exactly the same output** as the 2D net.

**Why it works (exam reasoning):** a video is a stack of images. The features a pretrained 2D kernel extracts (edges, textures, parts) are still the right features for each frame. Inflating keeps that knowledge, and the network only has to learn the temporal part during fine-tuning.

**Why divide by K_t?** Summing K_t identical copies would multiply the response by K_t. Dividing keeps the activation scale the same as the 2D net.

## 5.6 Transformers for video

**Recall ViT:**
- A 224×224×3 image is cut into 16×16×3 patches.
- Each patch is flattened (16·16·3 = **768** values) and linearly projected to D.
- Positional encoding is added.
- That gives **14×14 = 196 tokens per image**.

**Token explosion — the slide's numbers are a likely numerical:**

| Clip | Frames T | Tokens = 196 × T |
|---|---|---|
| Small clip | 16 | **3,136** (≈ a book chapter) |
| 5 min at 1 fps | 300 | **58,800** (≈ a short novel) |
| 5 min at 24 fps | 7,200 | **1,411,200** (≈ the context limit of modern LLMs) |

Self-attention cost grows with the **square** of the token count, so naive patch-level attention **scales very poorly**. There are two broad fixes.

### Strategy 1 — Modify the attention operator

| Attention type | Each token attends to… | Cost per token |
|---|---|---|
| **Joint space-time** | **All** N·T tokens in all frames | O(NT). "**Extremely inefficient**" |
| **Divided space-time** (TimeSformer, Bertasius 2021) | **Time step:** the **same spatial location** in the other frames. **Space step:** other tokens **in the same frame** | **O(N + T)** |
| **Video Swin** (Liu 2022) | Only tokens inside a **small local 3D window (cube)**; windows **shift** between layers | Fixed small window |

- **Divided attention block order:** Input → **Time Att.** → ⊕ → **Space Att.** → ⊕ → MLP → ⊕ → Output, with a residual around each step.
- **Why divided attention still works:** after many blocks, information still spreads across all of space and time, just as stacked convolutions grow the receptive field.
- **Video Swin** is "very similar to 3D CNNs, but with self-attention inside each cube". Shifting the windows lets information cross cube boundaries.
  - Slide example: 8×8×8 tokens with 4×4×4 windows give 2×2×2 = **8 windows** at layer l, and **3×3×3 = 27** shifted windows at layer l+1.
- **Why local attention is fine for images and video but not text (the teacher's point):**
  - Visual information is mostly **local**: neighbouring pixels and patches are related.
  - In text, the last word of a sentence can depend strongly on the first word.
  - Also, with too many tokens every attention weight becomes close to 0, so the signal gets diluted.

### Strategy 2 — Reduce the number of tokens: tubelets (ViViT, Arnab 2021)

- **Tubelet = a 3D space × time block** (e.g., 16×16 pixels × 4 frames) that becomes **one** token, instead of one token per patch per frame.
- **Patches carry no motion information; tubelets do**, because time is inside the token.
- **Tubelets over 4 frames give 4× fewer tokens and 16× less compute**, because attention is quadratic (4² = 16).
- The tubelet is still linearly projected to the transformer dimension D.
- Used by **ViViT, VideoMAE, Video Swin, MViT, V-JEPA**.

**Numerical:** 16 frames at 224×224, tubelet 4×16×16 → (16/4) × 14 × 14 = 4 × 196 = **784 tokens**, versus 3,136 with per-frame patches. That is 4× fewer tokens and ≈16× less attention compute.

## 5.7 Tracking (mentioned at the end of Session 8 — the "real challenge")

- **Detection** finds objects in one frame. **Tracking** keeps the **same ID** for the same object across frames.
- Problems in real dashcam video:
  - Detections are jittery.
  - Detections are missed in some frames (occlusion, fast motion).
  - **ID switches** happen: car #1 suddenly becomes car #2.
- **ByteTrack** and SORT-style trackers use a **Kalman filter**:
  1. **Predict** each track's next position from its motion state.
  2. **Correct** that prediction with the new detection.
  3. If a detection is missing, the prediction fills the gap (like interpolation).
- The same Kalman-filter idea (the "extended Kalman filter") is used for rocket trajectory estimation.

## 5.8 Kaggle / assignment note from Session 9

- Assignment 2 is **video accident detection** ("was there an accident in this clip?"), evaluated with **ROC-AUC**.
- **ROC-AUC = 0.5 means random guessing** (the model learned nothing), so it maps to 0 marks. 1.0 is perfect.

## 5.9 Quick self-test (Session 9)

1. Video tensor for 10 s, 25 fps, RGB, 128×128 → **250 × 3 × 128 × 128**.
2. Input channels to the temporal stream for T = 6 frames → 2 × (6 − 1) = **10**.
3. Why is a single-frame CNN a strong baseline? Many actions can be recognised from appearance alone (a pool means swimming).
4. Why can a 2D net be inflated? Video frames are images, so the same spatial filters apply. Copy the kernel K_t times and divide by K_t, and the output is unchanged on a static video.
5. Divided attention cost per token → O(N + T) instead of O(NT).
6. Tubelet over 2 frames: how many times less attention compute? Tokens drop 2×, so compute drops **4×**.


---

# PART 6 — Session 10: Image Segmentation

> **Source note:** No slides/transcript for Session 10 were uploaded, so this part is built from the **syllabus line** (Semantic vs Instance vs Panoptic; FCN, U-Net, Mask R-CNN; IoU / Dice; modern segmentation; lab with a pretrained model) plus what Session 6 & 7 already said about segmentation. These are the standard textbook facts the exam will expect.

## 6.1 The three kinds of segmentation (very high exam probability)

| | **Semantic** | **Instance** | **Panoptic** |
|---|---|---|---|
| Question answered | "What **class** is each pixel?" | "Which **object** does each pixel belong to?" | "Class **and** object id for **every** pixel" |
| Two cars side by side | Same label "car" → one blob | car#1, car#2 separate masks | car#1, car#2 separate **+** road/sky/building labelled |
| Background (sky, road) | Labelled | **Ignored** (only countable objects) | Labelled |
| Covers | Stuff + things, no ids | **Things only**, with ids | **Stuff + things**, things with ids |
| Typical model | FCN, U-Net, DeepLab | Mask R-CNN | Panoptic FPN, Mask2Former |
| Metric | mIoU, Dice | Mask AP | PQ (Panoptic Quality) |

- **Things** = countable objects you care to separate (cars, people, birds).
- **Stuff** = amorphous regions, no need to count (sky, road, grass, buildings in the driving example).
- **Panoptic = Semantic (stuff) + Instance (things).** The teacher's encoding idea: label `2 = car`, and `2.1, 2.2, 2.3` = individual cars.

**Why instance matters (teacher's autonomous-driving example):** with a semantic mask, two overlapping cars become one blob → when you project the mask to 3D using a depth map you assign *one* depth to two different cars and lose the gap between them. Instance masks keep them separate. Buildings don't need ids → stuff.

**Memory hook:** *Semantic = "what", Instance = "which one", Panoptic = "what + which one, everywhere".*

## 6.2 Why segmentation is harder than detection

- Output is **dense**: one prediction per pixel (H × W × C) vs 4 numbers + class per box.
- Needs **both** global context (what is it?) **and** fine spatial detail (exact boundary). CNN down-sampling gives context but destroys detail → architectures must **recover resolution** (upsampling + skip connections).
- Annotation is expensive (someone must draw polygons around each object). Today, foundation models like **SAM** pre-label, humans only verify.

## 6.3 FCN — Fully Convolutional Network (Long et al., 2015)

**Idea in one line:** take a classification CNN, **replace the fully connected layers with convolutions** so the network outputs a *map of class scores* instead of one vector, then **upsample** that map back to image size.

```
Image (H×W×3)
  → CNN backbone (VGG) — downsamples ×32
  → 1×1 conv  (FC layers turned into convolutions) → coarse score map (H/32 × W/32 × C)
  → upsample (transposed convolution / "deconvolution") → H × W × C
  → argmax over C per pixel → segmentation map
```

- **Why 1×1 conv instead of FC?** FC needs a fixed input size; convs accept **any input size** and keep the spatial layout. (Same reason Fast R-CNN needed RoI pooling — FC layers want fixed size.)
- **Skip connections:** upsampling ×32 in one jump gives blobby masks. FCN adds scores from earlier, higher-resolution layers:
  - **FCN-32s** — upsample ×32 from the last layer only (coarsest).
  - **FCN-16s** — fuse with pool4, then ×16.
  - **FCN-8s** — also fuse pool3, then ×8 (sharpest).
- **Loss:** per-pixel cross-entropy (softmax over C classes at every pixel).

**Transposed convolution output size** (likely numerical):

$$\text{out} = (n-1)\cdot s - 2p + k \;(+\text{output\_padding})$$

Example: n = 7, k = 4, s = 2, p = 1 → (7−1)·2 − 2 + 4 = **14** (doubles the size).

## 6.4 U-Net (Ronneberger et al., 2015 — biomedical images)

```
 Encoder (contracting)                     Decoder (expanding)
 conv-conv ───────────── copy & concat ──────────► up-conv, conv-conv  → 1×1 conv → mask
   ↓ maxpool                                         ↑ up-conv
 conv-conv ───────────── copy & concat ──────► conv-conv
   ↓                                                 ↑
 conv-conv ───── copy & concat ─────► conv-conv
   ↓                                    ↑
           bottleneck (conv-conv)  ─────┘
                     "U" shape
```

- **Encoder** captures *what* (context), **decoder** recovers *where* (localisation).
- **Skip connections concatenate** encoder feature maps with decoder feature maps at the same resolution → restores fine boundary detail lost in pooling.
- Works with **very few training images** (heavy augmentation, elastic deformations) → hugely popular in medical imaging.
- Original paper: 572×572 input → 388×388 output (unpadded "valid" convs); modern versions use padding so output = input size.

**FCN vs U-Net (classic 2-mark question):**

| | FCN | U-Net |
|---|---|---|
| Skip fusion | **Adds** (sums) score maps | **Concatenates** feature maps |
| Decoder | Very light (1–3 upsampling steps) | **Symmetric**, as deep as encoder |
| Strength | First end-to-end dense predictor | Sharp boundaries, small datasets (medical) |

## 6.5 Mask R-CNN (He et al., 2017) — instance segmentation

**One line:** *Faster R-CNN + a small FCN "mask head" that predicts a binary mask inside every detected box.*

```
Image → Backbone (ResNet + FPN) → RPN proposals
      → RoIAlign (fixed 7×7 / 14×14 features per RoI)
      → ┌ class head   (softmax over K+1)
        ├ box head     (bbox regression)
        └ mask head    (small FCN → K masks of 28×28, sigmoid per pixel)
```

Key points examiners love:

1. **Three parallel heads**; loss $L = L_{cls} + L_{box} + L_{mask}$.
2. **Mask loss = per-pixel binary cross-entropy (sigmoid)**, computed **only on the mask of the ground-truth class** → masks of different classes don't compete. *Decouples* "what class" (class head) from "what shape" (mask head). Softmax across classes for the mask was worse.
3. **RoIAlign replaces RoIPool.** RoIPool **rounds** (quantises) the RoI coordinates and bin boundaries to integers → a misalignment of up to a few pixels — harmless for boxes, harmful for pixel-accurate masks. **RoIAlign uses no rounding; it samples with bilinear interpolation.** Gave a big jump in mask accuracy.
4. Mask head is small (28×28 output per RoI) → only ~20% overhead over Faster R-CNN (~5 fps in the paper).
5. Same framework also does **human keypoint detection** (one-hot mask per keypoint).

**Mnemonic: "Mask R-CNN = Faster R-CNN + Mask head + RoIAlign."**

## 6.6 Segmentation metrics (Session 7 introduced IoU & Dice — numerical favourites)

For a predicted mask **P** and ground-truth mask **G** (sets of pixels):

| Metric | Formula | Note |
|---|---|---|
| Pixel accuracy | correct pixels / all pixels | **Misleading** with class imbalance (90% background → 90% accuracy by predicting all background) |
| IoU (Jaccard) | $\dfrac{\lvert P\cap G\rvert}{\lvert P\cup G\rvert} = \dfrac{TP}{TP+FP+FN}$ | per class |
| mIoU | mean of per-class IoU | standard for semantic segmentation (PASCAL VOC, Cityscapes) |
| Dice (F1) | $\dfrac{2\lvert P\cap G\rvert}{\lvert P\rvert+\lvert G\rvert} = \dfrac{2TP}{2TP+FP+FN}$ | **Dice ≥ IoU** always; standard in medical imaging |
| Conversion | $D = \dfrac{2J}{1+J}$, $\;J = \dfrac{D}{2-D}$ | J = IoU |
| PQ (panoptic) | $\dfrac{\sum_{TP} \text{IoU}}{\lvert TP\rvert + \tfrac12\lvert FP\rvert + \tfrac12\lvert FN\rvert}$ = SQ × RQ | a segment is a TP if IoU > 0.5 |

**Why Dice for small objects / medical (teacher's point):** Dice counts the overlap **twice** in the numerator, so it gives more weight to overlap; for tiny structures (few pixels) IoU punishes small misalignments very harshly, Dice is gentler and gives a smoother training signal. **Dice loss = 1 − Dice** is used to fight foreground/background imbalance.

**Worked numerical:** GT mask = 100 px, predicted mask = 80 px, overlap = 60 px.
- Union = 100 + 80 − 60 = 120 → **IoU = 60/120 = 0.50**
- **Dice = 2·60 / (100 + 80) = 120/180 = 0.667**
- Check: 2(0.5)/(1+0.5) = 0.667 ✓

**mIoU numerical:** IoU(road)=0.9, IoU(car)=0.6, IoU(person)=0.3 → mIoU = (0.9+0.6+0.3)/3 = **0.60**. Notice how a rare class (person) drags the mean — that's why mIoU is fairer than pixel accuracy.

## 6.7 Modern segmentation (overview level — know names + one line each)

| Model | One-line idea |
|---|---|
| **DeepLab (v1–v3+)** | **Atrous (dilated) convolution** enlarges receptive field without down-sampling; **ASPP** (Atrous Spatial Pyramid Pooling) = parallel dilated convs at several rates for multi-scale context; v3+ adds a decoder. |
| **PSPNet** | Pyramid pooling module — pool the feature map at several grid sizes for global context. |
| **Mask2Former / MaskFormer** | Transformer with **mask queries** (like DETR's object queries) — one architecture for semantic, instance **and** panoptic. |
| **SAM (Segment Anything, Meta 2023)** | **Promptable** foundation model: give a point / box / text-like prompt → mask. Trained on 1B+ masks; used to **auto-label** data (teacher: "for segmentation we have Segment Anything"). SAM 2 extends to video. |
| **Panoptic FPN** | Mask R-CNN + a semantic-segmentation branch on the FPN → panoptic output. |

**Dilated convolution numerical:** effective kernel size $k_{eff} = k + (k-1)(r-1)$. A 3×3 kernel with dilation r = 2 → 3 + 2·1 = **5×5** receptive field with only 9 weights.

## 6.8 Lab: run a pretrained segmentation model (expect a "predict the output / fill the blank" coding MCQ)

```python
import torch, torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()      # eval(): freeze dropout / BN stats
preprocess = weights.transforms()                       # resize + normalise like training

img = Image.open("street.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)                        # [1, 3, H, W]  (add batch dim)

with torch.no_grad():                                   # inference: no gradients
    out = model(x)["out"]                               # [1, 21, H, W]  21 = 20 VOC classes + background
mask = out.argmax(dim=1)[0]                             # [H, W]  class id per pixel
```

- Output shape **[N, C, H, W]** → `argmax(dim=1)` over the class dimension gives the label map.
- Instance segmentation version: `torchvision.models.detection.maskrcnn_resnet50_fpn` → returns a list of dicts with `boxes`, `labels`, `scores`, `masks` ([N, 1, H, W] soft masks → threshold at 0.5).
- Common bugs examiners test: forgetting `.eval()`, forgetting `unsqueeze(0)`, argmax on the wrong dim, not normalising like training.

## 6.9 Quick self-test (Session 10)

1. Self-driving car must know free road area **and** separate each pedestrian → **panoptic**.
2. Counting cells in a microscope image where cells touch → **instance** (semantic would merge touching cells).
3. Why does Mask R-CNN use sigmoid + BCE for masks instead of softmax? → decouples mask from class; avoids inter-class competition; class is decided by the class head.
4. RoIPool vs RoIAlign → quantisation vs bilinear interpolation; RoIAlign needed for pixel-accurate masks.
5. Dice = 0.8 → IoU = 0.8 / (2 − 0.8) = **0.667**.


---

# PART 7 — One-Page Formula Sheet (revise this the night before)

| # | Topic | Formula / number |
|---|---|---|
| 1 | Conv / pool output | $\lfloor (N + 2P - K)/S \rfloor + 1$ (per dimension; same for 3D in time) |
| 2 | Transposed conv output | $(N-1)S - 2P + K$ |
| 3 | Dilated kernel | $k_{eff} = k + (k-1)(r-1)$ |
| 4 | Receptive field (stride 1, k×k) | $RF = 1 + L(k-1)$ → 3, 5, 7 for 3×3 layers; general $RF_l = RF_{l-1} + (k_l-1)\prod_{i<l}s_i$ |
| 5 | HOG gradient | $\lvert G\rvert=\sqrt{G_x^2+G_y^2}$, $\theta=\tan^{-1}(G_y/G_x)$, 9 bins over 0–180° |
| 6 | HOG length (64×128, 8×8 cells, 2×2 blocks, 9 bins) | 7 × 15 blocks × 36 = **3,780** |
| 7 | SVM score | $\mathbf{w}^\top\mathbf{x} + b$ |
| 8 | R-CNN inference time | Proposal time + #Proposals × (conv + classify + regress) |
| 9 | RoI pool (VGG-16) | 512 × 7 × 7 = **25,088** |
| 10 | Anchors (Faster R-CNN) | H' × W' × k (k = 9 = 3 scales × 3 ratios); RPN outputs 2k scores + 4k offsets per position |
| 11 | RPN labels | positive IoU > **0.7**; negative < **0.3** |
| 12 | RPN loss | $\frac{1}{N_{cls}}\sum L_{cls}(p_i,p_i^*) + \lambda\frac{1}{N_{reg}}\sum p_i^* L_{reg}(t_i,t_i^*)$ |
| 13 | Box offsets | $t_x=\frac{x-x_a}{w_a},\ t_y=\frac{y-y_a}{h_a},\ t_w=\log\frac{w}{w_a},\ t_h=\log\frac{h}{h_a}$; decode $w = w_a e^{t_w}$ |
| 14 | Smooth L1 | $0.5x^2$ if $\lvert x\rvert<1$, else $\lvert x\rvert-0.5$ |
| 15 | SSD default boxes | 38²·4 + 19²·6 + 10²·6 + 5²·6 + 3²·4 + 1²·4 = **8,732** |
| 16 | SSD scale | $s_k = 0.2 + \frac{0.7}{m-1}(k-1)$ → 0.2, 0.34, 0.48, 0.62, 0.76, 0.9 |
| 17 | SSD box shape | $w = s_k\sqrt{a_r},\ h = s_k/\sqrt{a_r}$; output per box = 4 + (C+1) |
| 18 | SSD loss | $\frac{1}{N}(L_{conf} + \alpha L_{loc})$, α = 1, neg:pos = 3:1 |
| 19 | YOLOv1 tensor | $S\times S\times(5B + C)$ = 7×7×30 = **1,470**; 98 boxes |
| 20 | YOLO confidence | Pr(Obj) × IoU; class-specific = Pr(Class_i \| Obj) × Pr(Obj) × IoU = Pr(Class_i) × IoU |
| 21 | YOLOv1 loss weights | λ_coord = 5, λ_noobj = 0.5, √w, √h |
| 22 | YOLOv2/v3 anchor head | per scale S × S × B × (5 + C); v3 COCO: 3 × 85 = **255** channels; boxes at 416 = **10,647** |
| 23 | Anchor-free decode | $x_1=x-l,\ y_1=y-t,\ x_2=x+r,\ y_2=y+b$; 640² → 6,400 + 1,600 + 400 = **8,400** points |
| 24 | DFL | $d = \sum_i p_i\, i$ over 16 bins; per point 4×16 + 80 = **144** |
| 25 | IoU | $\frac{\lvert A\cap B\rvert}{\lvert A\rvert+\lvert B\rvert-\lvert A\cap B\rvert}$ |
| 26 | Dice | $\frac{2\lvert A\cap B\rvert}{\lvert A\rvert+\lvert B\rvert} = \frac{2J}{1+J}$; $J = \frac{D}{2-D}$ |
| 27 | Precision / Recall | $\frac{TP}{TP+FP}$ / $\frac{TP}{TP+FN}$ |
| 28 | F1 | $\frac{2PR}{P+R}$ |
| 29 | Interpolated precision | $p_{interp}(r)=\max_{\tilde r\ge r}p(\tilde r)$ |
| 30 | AP (11-point) | $\frac{1}{11}\sum_{r\in\{0,0.1,\ldots,1\}}p_{interp}(r)$ |
| 31 | AP (all-point) | $\sum_k (r_k - r_{k-1})\,p_{interp}(r_k)$ |
| 32 | mAP | $\frac{1}{N_{cls}}\sum_c AP_c$; COCO AP = mean over IoU 0.50:0.05:0.95 |
| 33 | Pixel accuracy / mIoU | correct px / all px; mean of per-class IoU |
| 34 | Panoptic Quality | $\frac{\sum_{TP}\text{IoU}}{\lvert TP\rvert+\frac12\lvert FP\rvert+\frac12\lvert FN\rvert}$ |
| 35 | Attention | $\text{softmax}(QK^\top/\sqrt{d_k})\,V$; QKᵀ is n×n; output n×d_k |
| 36 | Multi-head | $d_k = d/h$; Concat(heads) $W^O$ |
| 37 | Sinusoidal PE | $\sin(pos/10000^{2i/d})$, $\cos(pos/10000^{2i/d})$ |
| 38 | LayerNorm | $\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma+\beta$ per token |
| 39 | ViT tokens | $N = HW/P^2$ (+1 CLS); 224/16 → 196 (+1 = 197); patch = P²·3 = 768 |
| 40 | ViT-Base | 12 layers, D = 768, MLP 3072, 12 heads, 86M params |
| 41 | DETR feature map | C = 2048, H/32 × W/32 → 1×1 conv to d → HW tokens; N queries (100) |
| 42 | DETR matching / loss | $-\hat p(c) + L_{box}$ / $-\log\hat p(c) + L_{box}$; $L_{box} = \lambda_{iou}L_{GIoU} + \lambda_{L1}\lVert b-\hat b\rVert_1$ |
| 43 | Video tensor | T × 3 × H × W; frames = seconds × fps |
| 44 | Two-stream flow input | 2(T−1) × H × W |
| 45 | I3D init | copy the 2D kernel K_t times, divide by K_t |
| 46 | Video tokens | 196 × T; tubelet over t frames → tokens ÷ t, attention compute ÷ t² |
| 47 | Divided attention | per-token O(NT) → O(N + T) |
| 48 | Uncompressed video size | W × H × 3 B × fps × seconds (HD ≈ 10 GB/min, SD ≈ 1.5 GB/min) |


---

# PART 8 — Section F: Numericals Bank (15 marks — "formula → substitute → calculate → interpret")

> **The 4-step method from the exam slide:** (1) write the formula, (2) substitute, (3) calculate carefully, (4) state the answer with its meaning or units. Clear working earns partial marks even if the final number is wrong.

## Detection geometry

**F1. IoU and Dice of two boxes.** GT = [10, 20, 110, 120], Pred = [40, 50, 140, 140] (x1, y1, x2, y2).
- GT area = 100 × 100 = 10,000. Pred area = 100 × 90 = 9,000.
- Intersection: x from max(10, 40) = 40 to min(110, 140) = 110 → 70; y from max(20, 50) = 50 to min(120, 140) = 120 → 70. Area = **4,900**.
- Union = 10,000 + 9,000 − 4,900 = 14,100.
- **IoU = 4,900 / 14,100 = 0.348.** Below 0.5, so this is an **FP** at IoU threshold 0.5.
- **Dice** = 2 · 4,900 / 19,000 = **0.516**. Check: 2(0.348)/1.348 = 0.516 ✓.

**F2. IoU with centre-format boxes.** A = (cx, cy, w, h) = (50, 50, 40, 40), B = (60, 60, 40, 40).
- Convert: A = [30, 30, 70, 70], B = [40, 40, 80, 80].
- Intersection = (70 − 40) × (70 − 40) = 900. Union = 1,600 + 1,600 − 900 = 2,300.
- **IoU = 0.391.**
- **Interpretation:** a 10-px shift in both x and y on a 40-px box already drops IoU below 0.5. That is why small objects are hard to localise.

**F3. Non-overlapping boxes.** A = [0, 0, 2, 2], B = [3, 3, 5, 5].
- x-overlap = min(2, 5) − max(0, 3) = −1 → clamp to 0.
- **IoU = 0.** (Without the clamp, (−1) × (−1) = +1 would be a wrong "intersection".)

**F4. NMS by hand.** Scores: A = 0.90, B = 0.80, C = 0.75, D = 0.60. Pairwise IoUs: (A,B) = 0.70, (A,C) = 0.10, (A,D) = 0.30, (B,C) = 0.20, (B,D) = 0.40, (C,D) = 0.65. Threshold = 0.5.
1. Sort by score: A, B, C, D. **Keep A.** Drop any box with IoU(A, ·) > 0.5: **B removed** (0.70); C (0.10) and D (0.30) stay.
2. The next highest remaining is C. **Keep C.** Drop boxes with IoU(C, ·) > 0.5: **D removed** (0.65).
3. **Final = {A, C}.**
- **Sensitivity:** with threshold 0.7, B is not removed (0.70 is not > 0.70), giving {A, B, C}. A higher NMS threshold keeps more boxes, so duplicates rise and so do FPs.

**F5. Box-regression targets (Faster R-CNN parameterisation).** Anchor (x_a, y_a, w_a, h_a) = (100, 100, 50, 100); GT = (110, 90, 100, 100).
- t_x = (110 − 100)/50 = **0.2**
- t_y = (90 − 100)/100 = **−0.1**
- t_w = ln(100/50) = **0.693**
- t_h = ln(100/100) = **0**
- **Decode check:** w = 50 · e^0.693 = 100 ✓.

**F6. Anchor-free decode.** Point (x, y) = (320, 240); predicted (l, t, r, b) = (40, 60, 80, 20).
- Box = (320 − 40, 240 − 60, 320 + 80, 240 + 20) = **(280, 180, 400, 260)**.
- Width 120, height 80, area 9,600.

**F7. DFL expectation.** The bins 0–3 have probabilities (0.1, 0.2, 0.6, 0.1).
- **d = 0(0.1) + 1(0.2) + 2(0.6) + 3(0.1) = 1.7** bins.
- Multiply by the stride to get pixels: with stride 8, that is **13.6 px**.

## Counting boxes / tensors

**F8. YOLOv1 output.** S = 7, B = 2, C = 20 → 7 × 7 × (2·5 + 20) = **7 × 7 × 30 = 1,470** values and **98 boxes**.
- With S = 13, B = 3, C = 80: 13 × 13 × (15 + 80) = **13 × 13 × 95 = 16,055**.

**F9. Anchor-based head (YOLOv2/v3 style: classes predicted per anchor).**
- Per scale: S × S × B × (5 + C). YOLOv3 on COCO: B = 3, C = 80 → **255 channels**.
- **Total boxes at 416 input:** (13² + 26² + 52²) × 3 = (169 + 676 + 2,704) × 3 = **10,647**.
- **Strides:** 416/13 = 32, 416/26 = 16, 416/52 = 8.

**F10. SSD default boxes.**
- 38² · 4 = 5,776; 19² · 6 = 2,166; 10² · 6 = 600; 5² · 6 = 150; 3² · 4 = 36; 1² · 4 = 4.
- **Total = 8,732.**
- Scales (m = 6): s_k = 0.2 + 0.14(k − 1) → 0.20, 0.34, 0.48, 0.62, 0.76, 0.90.
- For a 300-px image, layer 3 (s = 0.48) with aspect ratio 2: **w = 0.48 · √2 · 300 ≈ 204 px, h = 0.48/√2 · 300 ≈ 102 px.**

**F11. Faster R-CNN anchors.** VGG stride 16, 640 × 960 image → feature map 40 × 60.
- Anchors = 40 × 60 × 9 = **21,600**.
- Per position, the RPN outputs 2k = **18** objectness scores and 4k = **36** offsets.
- Most anchors are negative or ignored, so the RPN samples a mini-batch (e.g., 256 anchors, up to 1:1 positive:negative).

**F12. Anchor-free points.** 640 × 640 input at strides 8/16/32 → 80² + 40² + 20² = 6,400 + 1,600 + 400 = **8,400** points.
- Per point on COCO: 4 × 16 + 80 = **144** values.
- At input 1280: 160² + 80² + 40² = 25,600 + 6,400 + 1,600 = **33,600** points (4× the 640 count).

**F13. YOLO class-specific confidence.** Box confidence = Pr(obj) × IoU = 0.9 × 0.8 = **0.72**. P(person | obj) = 0.7.
- **Score = 0.7 × 0.72 = 0.504.** If the detection threshold is 0.5, the box is (just) kept.

## Classical features / CNN geometry

**F14. HOG descriptor length.**
- **64 × 128 window:** 8 × 16 cells, 7 × 15 blocks, 36 values per block → **3,780**.
- **64 × 64 window:** 8 × 8 cells, 7 × 7 blocks → 49 × 36 = **1,764**.

**F15. Gradient.** For pixel values left = 50, right = 80, top = 40, bottom = 80:
- G_x = 80 − 50 = 30; G_y = 80 − 40 = 40.
- **Magnitude = 50**; **orientation = 53.13°**.
- It votes into the **40–60° bin** (9 bins of 20°).

**F16. Receptive field.**
- Three 3×3 stride-1 convs: RF = 1 + 3 × 2 = **7**.
- Mixed strides: conv3 (s1) → conv3 (s2) → conv3 (s1).
  - RF after each layer: 3 → 5 → 5 + 2 × 2 = **9**.
  - The jump used for each layer (the product of the strides before it) is 1, 1, 2.

**F17. Conv / pool output size.**
- 28 × 28 map, 2 × 2 max-pool, stride 2 → ⌊(28 − 2)/2⌋ + 1 = **14** (the exam's own example).
- 224 input, 7 × 7 conv, stride 2, padding 3 → ⌊(224 + 6 − 7)/2⌋ + 1 = **112**.
- VGG-16: 224 → 112 → 56 → 28 → 14 → **7** (five 2×2 pools).

**F18. RoI pooling.**
- A 14 × 10 RoI pooled to 7 × 7 gives bins of about 2 × 1.43 cells. RoIPool **rounds** these sizes; RoIAlign **doesn't**.
- The output is always 512 × 7 × 7 = **25,088** features, which matches VGG's FC input.

## Evaluation

**F19. P, R, F1.** 10 GT objects; the model outputs 12 boxes, 8 of them correct.
- TP = 8, FP = 12 − 8 = 4, FN = 10 − 8 = 2.
- **P = 8/12 = 0.667; R = 8/10 = 0.80; F1 = 2(0.667)(0.8)/(1.467) = 0.727.**

**F20. Effect of raising the threshold (teacher's numbers).**
- Before: TP = 5, FP = 3 → P = 5/8 = **0.625**.
- After raising the confidence threshold: TP = 4, FP = 1 → P = 4/5 = **0.80**.
- **Precision rises.** With, say, 8 GT objects, recall falls from 5/8 to 4/8.

**F21. AP — the slide example** (5 GT; ranked TP, TP, FP, TP, FP).
- P = 1, 1, 0.67, 0.75, 0.60; R = 0.2, 0.4, 0.4, 0.6, 0.6.
- **AP₁₁ = (5 × 1 + 2 × 0.75 + 4 × 0)/11 = 0.591.**
- **All-point AP = 0.4 × 1 + 0.2 × 0.75 = 0.55.**

**F22. AP — a longer example** (5 GT; 7 detections ranked by confidence: TP, TP, FP, TP, FP, TP, FP).

| Rank | TP/FP | cum TP | cum FP | P | R |
|---|---|---|---|---|---|
| 1 | TP | 1 | 0 | 1.000 | 0.2 |
| 2 | TP | 2 | 0 | 1.000 | 0.4 |
| 3 | FP | 2 | 1 | 0.667 | 0.4 |
| 4 | TP | 3 | 1 | 0.750 | 0.6 |
| 5 | FP | 3 | 2 | 0.600 | 0.6 |
| 6 | TP | 4 | 2 | 0.667 | 0.8 |
| 7 | FP | 4 | 3 | 0.571 | 0.8 |

**Interpolated precision at each recall level:**

| Recall levels | p_interp | Why |
|---|---|---|
| 0.0 – 0.4 | 1.0 | precision 1.0 is reached at R = 0.4 |
| 0.5 – 0.6 | 0.75 | max P at R ≥ 0.6 is 0.75 |
| 0.7 – 0.8 | 0.667 | max P at R ≥ 0.8 is 0.667 |
| 0.9 – 1.0 | 0 | recall never reaches 0.9 |

- **11-point:** the 11 levels are r = 0, 0.1, …, 1.0.
  - (5 × 1 + 2 × 0.75 + 2 × 0.667 + 2 × 0)/11 = 7.833/11 = **0.712**.
- **All-point:** 0.2(1) + 0.2(1) + 0.2(0.75) + 0.2(0.667) = **0.683**.
- Both numbers were verified with the code in Part 9.

**F23. mAP.** AP(car) = 0.87, AP(pedestrian) = 0.71, AP(bicycle) = 0.58, AP(sign) = 0.62 → **mAP = 2.78/4 = 0.695 ≈ 0.70.**

**F24. COCO AP.** APs at IoU 0.50, 0.55, …, 0.95 are 0.70, 0.68, 0.65, 0.62, 0.58, 0.52, 0.45, 0.36, 0.25, 0.10.
- **AP@[.5:.95] = 4.91 / 10 = 0.491.**
- **AP₅₀ = 0.70** (the value at IoU 0.50) and **AP₇₅ = 0.52** (the 6th value, at IoU 0.75). The steep drop at strict IoU shows **imprecise boxes**.

**F25. Dice ↔ IoU.**
- IoU 0.5 → Dice = 1/1.5 = **0.667**.
- Dice 0.8 → IoU = 0.8/1.2 = **0.667**.
- IoU 0.6 → Dice = **0.75**.

**F26. Segmentation masks.** GT = 100 px, Pred = 80 px, overlap = 60 px.
- Union = 120, so **IoU = 0.5**.
- **Dice = 120/180 = 0.667.**

## Transformers

**F27. ViT tokens.**
- 224 × 224, P = 16 → (224/16)² = **196** patches + CLS = **197** tokens.
- Each patch = 16 × 16 × 3 = **768** values.
- Attention map per head = 197 × 197 = **38,809** entries.
- With P = 32: 7² = **49** (+1 = 50).
- 384 × 384, P = 16: 24² = **576**.
- **Linear projection parameters** (768 → D = 768): 768 × 768 + 768 = **590,592**.

**F28. Attention shapes.** n = 50 tokens, d = 512, h = 8.
- d_k = 512/8 = **64**.
- Q, K, V per head: **50 × 64**.
- QKᵀ: **50 × 50**.
- Head output: 50 × 64. Concatenated: **50 × 512**. W^O: 512 × 512.

**F29. Why √d_k (number version).** If the q and k components each have variance 1, then q·k has variance d_k. With d_k = 64 the standard deviation is 8, so the logits are about ±8 and the softmax saturates. **Divide by √64 = 8** to bring the standard deviation back to 1.

**F30. DETR shapes.** Input 800 × 1216.
- Backbone output: 2048 × 25 × 38.
- After the 1×1 conv: 256 × 25 × 38.
- Tokens: **950**. Encoder attention map: **950 × 950 ≈ 902k** entries.
- Decoder: 100 queries, 100 predictions.
- The 1×1 conv has 2048 × 256 + 256 = **524,544** parameters.

**F31. ImageGPT palette.** 24-bit colour = 2²⁴ ≈ 16.7M colours → 9-bit palette = 2⁹ = **512** colours (k-means, k = 512). A 32 × 32 image gives a sequence of **1,024** tokens.

## Video

**F32. Video tensor.** 30 s × 30 fps = **900** frames.
- Greyscale 256 × 256 → **900 × 1 × 256 × 256** = 58,982,400 values ≈ **59 MB** at 1 byte each.
- RGB: ≈ **177 MB**.

**F33. Uncompressed HD.** 1920 × 1080 × 3 B × 30 fps × 60 s = **11.2 GB/min** (the slide says ~10 GB/min).
- SD 640 × 480: **1.66 GB/min** (~1.5 GB/min).
- The class clip (T = 16, 112 × 112 × 3) = 602,112 B ≈ **588 KB**.

**F34. 3D conv.** Input T = 6 frames, 3 × 32 × 32. Five filters of size 2 × 3 × 5 × 5, stride 1, no padding.
- T' = 6 − 2 + 1 = **5**; H' = W' = 32 − 5 + 1 = **28**.
- **Output: 5 × 5 × 28 × 28** (filters × T' × H' × W').
- Parameters = 5 × (2·3·5·5) + 5 = **755**.

**F35. Two-stream.** T = 11 frames → 10 flow fields × 2 (dx, dy) = **20** input channels to the temporal stream.

**F36. Video tokens.**
- 16 frames × 196 = **3,136**.
- 5 min at 1 fps = 300 frames × 196 = **58,800**.
- 5 min at 24 fps = 7,200 × 196 = **1,411,200**.
- **Tubelets over 4 frames:** 3,136 / 4 = **784** tokens, giving 16× less attention compute (quadratic).
- **Divided attention** for N = 196, T = 16: per token 196 + 16 = **212** comparisons, versus 3,136 for joint attention.


---

# PART 9 — Section D: Coding Bank (20 marks, choose 1 of 3, hidden test cases)

**The exam's advice:** "Think like a DSA candidate."
- Write a **reusable function**.
- Handle **empty input and boundary cases**.
- Check **indexing and zero denominators**.

From this module, the most likely coding tasks are **IoU, NMS, Precision/Recall/F1, AP, Dice/IoU for masks, box-format conversion, conv output size and argmax with ties**. Functions D1–D9 were **run and tested**; the expected outputs are shown. The D10 shapes follow from the conv formula.

> **Universal edge-case checklist** (hidden test cases target these):
> - Empty list → return `[]`, `0.0` or `-1`, as the spec says.
> - Division by zero (union = 0, TP + FP = 0) → return 0.0 and never crash.
> - Boxes that **don't overlap** → clamp the intersection width/height at 0 with `max(0, ·)`.
> - Ties → keep the **first** index (use a strict `>`).
> - Box format: is it `[x1, y1, x2, y2]` or `[cx, cy, w, h]`? Convert first.
> - Don't modify the input list in place unless asked.

## D1. IoU of two boxes

```python
def iou(box_a, box_b):
    """Boxes as [x1, y1, x2, y2]. Returns IoU in [0, 1]."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)          # clamp: no overlap -> 0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0              # avoid divide-by-zero
```

| Test | Output |
|---|---|
| `iou([0,0,4,4],[2,2,6,6])` | inter = 2·2 = 4, union = 16 + 16 − 4 = 28 → **0.1429** |
| `iou([1,1,5,5],[2,2,6,6])` | inter = 9, union = 23 → **0.3913** |
| `iou([0,0,2,2],[3,3,5,5])` | no overlap → **0.0** |
| `iou([0,0,0,0],[0,0,0,0])` | zero-area boxes → **0.0** (no crash) |

**Classic bug:** forgetting `max(0, …)`. Two non-overlapping boxes then give a **negative × negative = positive** "intersection".

## D2. Non-Maximum Suppression (greedy)

```python
def nms(boxes, scores, iou_thresh=0.5):
    """Returns indices of kept boxes, highest score first."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        best = order.pop(0)               # 1. pick highest-scoring box
        keep.append(best)
        order = [i for i in order          # 2. drop boxes overlapping it too much
                 if iou(boxes[best], boxes[i]) <= iou_thresh]
    return keep
```

**Test:**
- `boxes = [[10,10,50,50],[12,12,52,52],[100,100,140,140],[11,9,49,51]]`, `scores = [0.9,0.8,0.7,0.6]`
- IoUs with box 0: box 1 = 0.822, box 2 = 0.0, box 3 = 0.907.
- `nms(boxes, scores, 0.5)` → **[0, 2]** (boxes 1 and 3 are suppressed as duplicates of 0).
- `nms([], [], 0.5)` → **[]**.

**Remember:**
- Run NMS **per class** (a dog box shouldn't suppress an overlapping person box).
- Usually apply a **score threshold first** (e.g., drop boxes with score < 0.25).
- Complexity is O(n²) in the worst case.

## D3. Precision, Recall, F1 (zero-safe)

```python
def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1
```

- `precision_recall_f1(8, 2, 4)` → P = 0.8, R = 0.667, F1 = **0.727**
- `precision_recall_f1(0, 0, 0)` → (0.0, 0.0, 0.0)

## D4. Average Precision (all-point interpolation — VOC 2010+ / COCO style)

```python
def average_precision(confidences, is_tp, num_gt):
    """All-point interpolated AP (area under the monotone PR curve)."""
    if num_gt == 0 or not confidences:
        return 0.0
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    for i in order:
        if is_tp[i]:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / num_gt)
    # make precision monotonically decreasing (right-to-left max)
    for k in range(len(precisions) - 2, -1, -1):
        precisions[k] = max(precisions[k], precisions[k + 1])
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precisions, recalls):
        ap += (r - prev_r) * p
        prev_r = r
    return ap
```

**11-point version (VOC 2007, the one taught in class):**

```python
def ap_11_point(confidences, is_tp, num_gt):
    if num_gt == 0 or not confidences:
        return 0.0
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    tp_cum = fp_cum = 0
    pr = []
    for i in order:
        tp_cum += is_tp[i]; fp_cum += (not is_tp[i])
        pr.append((tp_cum / (tp_cum + fp_cum), tp_cum / num_gt))
    total = 0.0
    for t in [x / 10 for x in range(11)]:           # 0.0, 0.1, ..., 1.0
        ps = [p for p, r in pr if r >= t]
        total += max(ps) if ps else 0.0              # max precision at recall >= t
    return total / 11
```

**Test** (worked by hand in Part 8, F22): `conf = [0.95,0.90,0.85,0.80,0.70,0.60,0.50]`, `tp = [1,1,0,1,0,1,0]`, `num_gt = 5`
- all-point AP = **0.6833**
- 11-point AP = **0.7121**

**mAP** = the mean of AP over classes: `sum(ap_per_class) / len(ap_per_class)`, guarding against an empty list.

## D5. Dice and IoU for binary masks

```python
def dice_iou_masks(pred, gt):
    """pred, gt: 2-D lists of 0/1."""
    inter = sum(p & g for rp, rg in zip(pred, gt) for p, g in zip(rp, rg))
    sp = sum(map(sum, pred)); sg = sum(map(sum, gt))
    union = sp + sg - inter
    dice = 2 * inter / (sp + sg) if (sp + sg) > 0 else 1.0   # both empty = perfect match
    iou_ = inter / union if union > 0 else 1.0
    return dice, iou_
```

**Test:** `pred = [[1,1,0],[1,1,0],[0,0,0]]`, `gt = [[0,1,1],[0,1,1],[0,0,0]]`
- inter = 2, |P| = 4, |G| = 4.
- Dice = 4/8 = **0.5**; IoU = 2/6 = **0.333**.

**Convention:** if both masks are empty, many libraries return 1.0 (a perfect agreement that "nothing is there"). Say which convention you use in a comment.

NumPy one-liners (if NumPy is allowed):
- `inter = np.logical_and(p, g).sum()`
- `union = np.logical_or(p, g).sum()`

## D6. Index of the highest score (the exam's own sample problem)

```python
def argmax_first(scores):
    if not scores:
        return -1
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best]:     # strict '>' keeps the FIRST index on ties
            best = i
    return best
```

| Test | Output |
|---|---|
| `[0.2,0.8,0.8,0.1]` | **1** |
| `[]` | **−1** |
| `[0.4]` | **0** |
| `[0.5,0.5]` | **0** |

## D7. Conv / pool output size

```python
def conv_out(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1
```

- `conv_out(28, 2, 2)` → **14** (the exam's max-pool sample)
- `conv_out(224, 7, 2, 3)` → **112** (ResNet stem)
- `conv_out(32, 5)` → **28**

## D8. Box format conversions

```python
def xywh_to_xyxy(b):                     # centre format -> corner format
    cx, cy, w, h = b
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

def ltrb_to_xyxy(px, py, l, t, r, b):    # anchor-free (FCOS / YOLOv8) decode
    return [px - l, py - t, px + r, py + b]
```

- `xywh_to_xyxy([50,40,20,10])` → **[40, 35, 60, 45]**
- `ltrb_to_xyxy(100, 80, 20, 10, 30, 40)` → **[80, 70, 130, 120]**

## D9. Count TP / FP / FN from predictions (matching step before AP)

```python
def match_detections(preds, gts, iou_thresh=0.5):
    """preds: list of (box, score); gts: list of boxes (one class).
    Greedy: highest score first; each GT can be matched only once."""
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    used = [False] * len(gts)
    flags = []                               # 1 = TP, 0 = FP (in score order)
    for box, _ in preds:
        best_j, best_iou = -1, iou_thresh
        for j, g in enumerate(gts):
            o = iou(box, g)
            if not used[j] and o >= best_iou:
                best_j, best_iou = j, o
        if best_j >= 0:
            used[best_j] = True
            flags.append(1)
        else:
            flags.append(0)                  # no GT or GT already taken -> duplicate = FP
    tp = sum(flags); fp = len(flags) - tp; fn = used.count(False)
    return flags, tp, fp, fn
```

**Key rule:** a **second** detection of an already-matched object is an **FP**. That is why NMS matters for precision.

## D10. Reading code (MCQ style: "what does this output?")

```python
scores = [0.9, 0.75, 0.6, 0.3]
keep = [i for i, s in enumerate(scores) if s >= 0.6]
print(len(keep))          # → 3
```

```python
import torch
x = torch.randn(1, 3, 224, 224)
conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
print(conv(x).shape)      # → torch.Size([1, 64, 112, 112])   ((224+2-3)//2+1 = 112)
```

```python
out = torch.randn(1, 21, 520, 520)   # segmentation logits
print(out.argmax(1).shape)           # → torch.Size([1, 520, 520])
```

```python
# YOLOv1 output tensor
S, B, C = 7, 2, 20
print(S * S * (B * 5 + C))           # → 1470  (7×7×30)
```

**How to answer in the exam:** state the inputs and output → write a clean function → add comments for each edge case → hand-trace one example. Choose the option whose logic you can **explain**.


---

# PART 10 — Section E: Case-Study Playbook (15 marks)

**The exam's logical chain: Problem → Task → Data → Model → Metric → Failure mode.**

Answer every case study with this **7-line template**. Each line should give a decision **plus a reason**.

| Step | Question to answer | Module-4 vocabulary to use |
|---|---|---|
| 1. Task | Classification, localisation, detection, or semantic / instance / panoptic segmentation? **Why** that one and not a richer or poorer one? | "choose the simplest output that answers the question" |
| 2. Data & labels | What to label (image tags / boxes / masks)? What variation to include? | lighting, occlusion, scale, viewpoint, class imbalance, augmentation (Mosaic, MixUp, flips, colour jitter) |
| 3. Model | Which detector family and why? | one-stage (YOLO) = real-time / edge; two-stage = accuracy, small objects; DETR = simple pipeline, large objects; U-Net / Mask R-CNN for masks; pretrained backbone + fine-tune |
| 4. Metric | What to monitor? | mAP@0.5, AP_S, recall/precision at the operating threshold, IoU/Dice, FPS/latency |
| 5. Costly error | Is an FN or an FP worse? Adjust the threshold to match | safety → recall; nuisance alarms → precision |
| 6. Failure modes | When will it break? | occlusion, small objects, crowded scenes (NMS), domain shift (night, new camera/fisheye), motion blur |
| 7. Deployment | Constraints? | edge vs cloud latency, no network, model size, FPS, privacy |

---

## Case 1 — Supermarket empty-shelf detection (the exam's own sample)

*Cameras watch the shelves. Lighting varies, and customers sometimes block the shelves.*

1. **Task: object detection** of "empty-slot" regions, with boxes.
   - **Why not classification?** "Shelf has a gap: yes/no" doesn't tell staff *where* to restock.
   - **Why not segmentation?** Pixel masks cost more to annotate and run, and a box is enough to send staff to a location.
   - (Alternative: **semantic segmentation** of "empty shelf area", if you must measure the *fraction* of empty space.)
2. **Data:**
   - Images across **all lighting** (morning/evening, flicker), stores, shelf types and camera angles.
   - Include **partial occlusion by customers** and **negative examples**: dark products or shadows that look like gaps.
   - Augment with brightness/contrast jitter and cut-out (to simulate occlusion).
3. **Model:**
   - A **YOLO-family one-stage detector** with a pretrained backbone, fine-tuned.
   - Speed matters only modestly (shelves change slowly), so a lightweight model can run on an edge box.
   - Process **one frame every few minutes**, not 30 fps.
4. **Metric:**
   - **mAP@0.5** for the empty-slot class.
   - In operation, track **recall** (missed gaps mean lost sales) at a precision staff will tolerate.
5. **Costly error:**
   - A **false negative** (a missed empty shelf) loses sales.
   - A false positive wastes a staff trip.
   - So prefer **higher recall** with moderate precision.
6. **Failure modes and fixes:**
   - **Occlusion by customers** → aggregate predictions over several frames (temporal voting, as in Session 9) and alert only if the gap persists.
   - **Lighting changes** → augmentation; fix the camera exposure.
   - **Black product packaging** mistaken for a gap → hard negatives.

## Case 2 — Pedestrian detection for an autonomous car

1. **Task:** **detection** of pedestrians and cyclists, plus **tracking** (consistent IDs across frames, Kalman filter / ByteTrack) to predict their paths.
   - Add **instance/panoptic segmentation** if a depth/3D projection is needed.
   - Teacher's point: semantic masks merge adjacent people, giving the wrong depth.
2. **Data:**
   - Day/night, rain/fog, all **scales** (far pedestrians are tiny) and **occlusion** (behind cars, crowds).
   - Diverse cities.
   - Rare cases: children, wheelchairs, people carrying objects.
3. **Model:**
   - A **real-time one-stage detector** (YOLO with FPN/PAN for multi-scale) running **on the car** (no cloud: latency and no network).
   - Must exceed the camera frame rate, or frames get skipped.
4. **Metric:**
   - **Recall** at the operating point, plus **AP_S** (small/far pedestrians).
   - **Latency/FPS**.
   - Tracking metrics such as **ID switches**.
5. **Costly error:** an **FN is life-critical**, so maximise **recall**. Accept more FPs: the planner can double-check them with other sensors.
   - Contrast the teacher's **lane-change** example: for the decision "is it safe to change lane now", a wrong "yes" kills, so you optimise **precision** of the "safe" call. Missing a lane-change opportunity costs nothing.
6. **Failure modes:**
   - Small/distant people.
   - **Crowds**: NMS suppresses true overlapping people, and old YOLOv1 is limited to 2 objects per cell.
   - Night-time domain shift.
   - Motion blur.
   - New camera types (e.g., **fisheye**) the model never saw.
7. **Deployment:** on-board GPU, model quantisation, a fixed latency budget, redundancy with LiDAR/radar.

## Case 3 — Brain-tumour localisation in MRI (Session 6's "localisation" example)

1. **Task:**
   - **Localisation** (one tumour class + one box) if you only need *where*.
   - **Semantic segmentation (U-Net)** if the surgeon needs the tumour's **exact boundary and volume**.
2. **Data:**
   - Small, expensive, expert-labelled datasets, so use **transfer learning** and heavy augmentation (rotations, elastic deformations).
   - Check scanner/hospital variation.
   - Class imbalance: the tumour is tiny relative to the image.
3. **Model:** **U-Net**. Its skip connections give sharp boundaries, and it works with few images.
4. **Metric:** **Dice** coefficient (plus IoU). Dice is preferred for small structures. Use **Dice loss** for the imbalance.
5. **Costly error:** a **missed tumour (FN)** is worse, so favour **recall/sensitivity**; a doctor confirms positives (the teacher's cancer example).
6. **Failure modes:** a different MRI machine or protocol (**domain shift**), tiny lesions, motion artefacts.

## Case 4 — Factory defect inspection

1. **Task:**
   - **Classification** (defect / no defect) if parts are just rejected. This is the slide's example, and the **cheapest**.
   - **Detection** if the defect type and location are needed for repair.
   - **Segmentation** if the defect area must be measured.
2. **Data:**
   - Defects are **rare**, giving a heavy **class imbalance**. Collect or augment defects; consider anomaly detection.
   - Fixed lighting/camera rig (control the environment).
3. **Model:** a small CNN classifier or YOLO on an edge device at line speed.
4. **Metric:** recall on the defect class at a tolerable false-reject rate; PR curve / AP rather than accuracy (99% "no defect" makes accuracy meaningless).
5. **Failure modes:** new defect types, lens dirt, product changes.

## Case 5 — Drone / aerial counting of vehicles (small objects)

- **Task:** detection, possibly with oriented boxes. **Instance segmentation** if vehicles touch.
- **Key issue:** **tiny objects**.
  - Use **high-resolution input or image tiling**, **FPN/PAN** multi-scale features, and **small anchors** or anchor-free heads.
  - **Avoid vanilla DETR**: it is weak on small objects (AP_S 20.5 in the slide table).
- **Metric:** **AP_S**, plus counting error.
- **Failure modes:** shadows, altitude changes (scale), motion blur, dense parking lots (NMS suppression).

## Case 6 — Video: detecting accidents in traffic-camera clips (Kaggle assignment 2)

- **Task:** **video classification** (accident yes/no per clip). Add temporal localisation (start/end time) if needed.
- **Data:**
  - Short clips at **low fps** for training.
  - Accidents are rare, so **class imbalance**.
  - Remove near-duplicate frames.
- **Model:**
  1. Baseline: **single-frame CNN + averaging**.
  2. Better: **3D CNN / I3D** (inflated pretrained weights) or a **two-stream** network (optical flow captures the sudden motion change).
  3. Or a **video transformer** with **tubelets / divided attention** for efficiency.
- **Metric:** **ROC-AUC** (0.5 = random), plus recall on accidents.
- **Failure modes:** night/rain, camera shake mistaken for motion, occlusion.

---

**Scoring tip:** examiners give marks for **justifications** ("because…"), not for naming many models. Always mention **one failure mode and one mitigation**: that is usually the last few marks.


---

# PART 11 — Practice Question Bank (Sections A, B, C)

> Cover the answers and try each one first. Answers sit right below each block, with the **reason the closest distractor is wrong**. That is exactly what the exam's "MCQ sprint" asks you to practise.

## Section A — Single-correct MCQs (1 mark each)

**A1.** A factory only needs to know whether a part is defective, not where the defect is. The most appropriate task is:
A. Instance segmentation  B. Object detection  C. Image classification  D. Panoptic segmentation

**A2.** In R-CNN, each region proposal is warped to a fixed size because:
A. Selective Search requires square regions  B. The CNN's fully connected layers need a fixed input size  C. SVMs only accept 227×227 images  D. It removes background

**A3.** The main computational improvement of Fast R-CNN over R-CNN is:
A. Replacing Selective Search with an RPN  B. Running the CNN once per image and pooling RoIs from the shared feature map  C. Using anchor boxes  D. Removing bounding-box regression

**A4.** Faster R-CNN removes which bottleneck of Fast R-CNN?
A. RoI pooling  B. Softmax classifier  C. Selective Search region proposals  D. Smooth L1 loss

**A5.** In the RPN, an anchor is labelled positive when its IoU with a ground-truth box is:
A. > 0.3  B. > 0.5  C. > 0.7  D. = 1.0

**A6.** YOLOv1 with S = 7, B = 2, C = 20 outputs a tensor of shape:
A. 7×7×25  B. 7×7×30  C. 7×7×40  D. 49×20

**A7.** Non-Maximum Suppression is:
A. A learned layer trained with backprop  B. A post-processing step that removes duplicate overlapping boxes  C. A data-augmentation method  D. A loss function for box regression

**A8.** Box A = [0,0,4,4], Box B = [2,2,6,6]. IoU = ?
A. 0.25  B. 0.14  C. 0.50  D. 0.33

**A9.** Using sigmoid per class instead of softmax (YOLOv3) allows:
A. Faster NMS  B. Multi-label predictions (e.g., "person" and "woman" on one box)  C. Removal of anchors  D. Higher input resolution

**A10.** Increasing the confidence threshold of a detector typically:
A. Increases recall  B. Increases precision and decreases recall  C. Increases both  D. Has no effect on precision

**A11.** DETR does not need NMS because:
A. It uses very few anchors  B. Hungarian one-to-one matching trains duplicates to predict "no object"  C. It uses a sigmoid classifier  D. It predicts only one object per image

**A12.** In scaled dot-product attention, dividing by √d_k:
A. Reduces parameters  B. Prevents large dot products from saturating the softmax  C. Adds positional information  D. Normalises across the batch

**A13.** ViT with a 224×224 image and 16×16 patches feeds the encoder a sequence of length (including [CLS]):
A. 196  B. 197  C. 256  D. 785

**A14.** Which detector is typically best for real-time detection on an edge device?
A. R-CNN  B. Fast R-CNN  C. YOLO  D. DETR (R101)

**A15.** For tiny tumours in medical images, the preferred overlap metric/loss is:
A. Pixel accuracy  B. Dice  C. Top-1 accuracy  D. BLEU

**A16.** In two-stream networks, the temporal stream input for T frames has how many channels?
A. 3T  B. 2T  C. 2(T−1)  D. T−1

**A17.** RoIAlign improves on RoIPool by:
A. Using average pooling  B. Avoiding coordinate quantisation via bilinear interpolation  C. Using larger output grids  D. Removing the backbone

**A18.** SSD300's total number of default boxes is:
A. 98  B. 8732  C. 10,647  D. 2000

**A19.** Hard negative mining in SSD addresses:
A. Overfitting to large objects  B. Extreme foreground–background class imbalance  C. Slow NMS  D. Vanishing gradients

**A20.** Which normalisation is computed within each token independently of the batch?
A. BatchNorm  B. LayerNorm  C. Dropout  D. Min-max scaling

### Answers — Section A

| Q | Ans | Why (and why the closest distractor is wrong) |
|---|---|---|
| A1 | **C** | Only *what* is needed. Detection and segmentation add unnecessary compute and annotation cost ("choose the simplest"). |
| A2 | **B** | FC layers fix the input size. Selective Search produces arbitrary rectangles, which is *why* warping is needed (A is backwards). |
| A3 | **B** | Shared feature map + RoI pooling. The RPN (A) belongs to **Faster** R-CNN. |
| A4 | **C** | The RPN replaces Selective Search. RoI pooling is kept. |
| A5 | **C** | IoU > 0.7 is positive, < 0.3 is negative. 0.5 is the usual *evaluation* threshold, not RPN labelling. |
| A6 | **B** | B·5 + C = 2·5 + 20 = 30. 25 forgets the second box. |
| A7 | **B** | NMS is **not learned** (the teacher stressed this). Pure post-processing. |
| A8 | **B** | Intersection 2×2 = 4; union = 16 + 16 − 4 = 28; 4/28 = 0.143. 0.25 wrongly divides by one box area. |
| A9 | **B** | Independent sigmoids let several classes be "on" at once. Softmax forces them to compete. |
| A10 | **B** | Fewer, more confident detections mean fewer FPs (precision ↑), but more misses (recall ↓). |
| A11 | **B** | DETR has **no** anchors (A is false). One-to-one matching penalises duplicates. |
| A12 | **B** | The variance of q·k grows with d_k; scaling keeps the softmax gradients healthy. |
| A13 | **B** | (224/16)² = 196 patches + 1 CLS = 197. 196 forgets CLS. |
| A14 | **C** | One-stage and very fast. DETR-R101 runs at ~20 fps with heavy GFLOPs. |
| A15 | **B** | Dice weights overlap more and handles imbalance. Pixel accuracy is fooled by the large background. |
| A16 | **C** | T−1 flow fields × 2 (dx, dy). |
| A17 | **B** | No rounding; bilinear sampling, so masks align with pixels. |
| A18 | **B** | 38²·4 + 19²·6 + 10²·6 + 5²·6 + 3²·4 + 1²·4 = 8732. 98 is YOLOv1; 10,647 is YOLOv3 at 416. |
| A19 | **B** | Most default boxes are background. Keep only the hardest negatives (neg:pos ≈ 3:1). |
| A20 | **B** | BatchNorm uses batch statistics. |

---

## Section B — Multiple-correct MCQs (2 marks; all correct and no wrong option for full marks)

**B1.** Which are limitations of R-CNN?
A. ~2000 CNN forward passes per image  B. No joint training of proposals and classifier  C. Uses anchor boxes  D. Large disk space for cached features  E. Separate SVM and regressor training

**B2.** Which statements about anchor boxes are TRUE?
A. They are predefined reference boxes of multiple scales and aspect ratios  B. The network predicts offsets relative to them  C. YOLOv2 chose anchor shapes using k-means on training boxes  D. DETR uses 9 anchors per location  E. Anchor-free detectors predict distances from a point to the box sides

**B3.** Which are true about NMS?
A. It is a learned component  B. It sorts boxes by confidence  C. It removes boxes whose IoU with a kept box exceeds a threshold  D. It is usually applied per class  E. It can wrongly suppress a true object in crowded scenes

**B4.** Which changes affect the size of YOLOv1's output tensor?
A. Grid size S  B. Boxes per cell B  C. Number of classes C  D. The learning rate  E. Image file name

**B5.** Which are TRUE about precision and recall in detection?
A. Precision = TP/(TP+FP)  B. Recall = TP/(TP+FN)  C. True negatives are used in both  D. Both depend on the IoU threshold used to define a TP  E. Raising the IoU threshold can only increase AP

**B6.** Which are features of SSD?
A. Predictions from multiple feature maps of different resolutions  B. Default boxes with several aspect ratios per cell  C. A separate region-proposal stage  D. Hard negative mining  E. C+1 class scores per box (including background)

**B7.** Which are TRUE of DETR?
A. Uses a CNN backbone  B. Uses a fixed number of object queries  C. Uses Hungarian matching during training  D. Requires NMS  E. Performs better than Faster R-CNN on large objects

**B8.** Which statements about self-attention are TRUE?
A. Q, K, V are linear projections of the same input (self-attention)  B. The attention map for n tokens is n×n  C. Each row of the attention map sums to 1 after softmax  D. It is inherently aware of token order  E. Cost grows quadratically with sequence length

**B9.** Which help detect small objects?
A. Feature Pyramid Network / multi-scale feature maps  B. Higher input resolution or tiling  C. Using only the last 1/32 feature map  D. Smaller anchors  E. Dice-style or IoU-aware losses that don't drown small boxes

**B10.** Which are TRUE about Dice and IoU?
A. Dice = 2|A∩B|/(|A|+|B|)  B. Dice ≥ IoU for the same masks  C. Dice = 2·IoU/(1+IoU)  D. IoU can exceed 1  E. Both equal 1 for a perfect match

**B11.** Which are valid ways to make transformers practical for video?
A. Divided space-time attention  B. Tubelet embeddings  C. Local windowed attention (Video Swin)  D. Treating every pixel of every frame as a token  E. Reducing fps / resolution of training clips

**B12.** Which are TRUE of YOLO's evolution?
A. v2 introduced k-means anchors and batch norm  B. v3 used Darknet-53 and multi-scale predictions  C. v4 popularised Mosaic augmentation, SPP and PAN  D. v8 is anchor-free  E. v10 removed NMS at inference with a one-to-one head

### Answers — Section B

| Q | Correct | Notes |
|---|---|---|
| B1 | **A, B, D, E** | Anchors (C) arrive with Faster R-CNN. |
| B2 | **A, B, C, E** | DETR is anchor-free (D false). 9 anchors is Faster R-CNN. |
| B3 | **B, C, D, E** | NOT learned (A false). |
| B4 | **A, B, C** | S×S×(B·5+C). Learning rate and file name don't change shapes. |
| B5 | **A, B, D** | Detection has no meaningful TN (C false). A stricter IoU threshold *lowers* AP (E false). |
| B6 | **A, B, D, E** | SSD is single-shot: no proposal stage. |
| B7 | **A, B, C, E** | No NMS (D false). AP_L 61.1 vs 53.4. |
| B8 | **A, B, C, E** | It is permutation-invariant and needs positional encoding (D false). |
| B9 | **A, B, D, E** | Only the coarse map (C) hurts small objects (DETR's weakness). |
| B10 | **A, B, C, E** | Both lie in [0, 1]. |
| B11 | **A, B, C, E** | D is what causes the token explosion. |
| B12 | **A, B, C, D, E** | All true. |

---

## Section C — Descriptive (2 marks, ≤ 100 words): model answers

**Structure that scores:** *Define accurately → add one distinction, reason or example → stop.*

**C1. Differentiate semantic, instance and panoptic segmentation.**
> Semantic segmentation assigns a class to every pixel but does not separate objects: two adjacent cars become one "car" region. Instance segmentation gives each countable object (thing) its own mask and id, but ignores background stuff such as road or sky. Panoptic segmentation combines both: every pixel gets a class, and things also get instance ids. Example: autonomous driving needs separate car instances (for correct depth per car) while buildings can share one label.

**C2. Why was Fast R-CNN faster than R-CNN?**
> R-CNN warped ~2000 proposals and ran the CNN on each one separately (47 s/image). Fast R-CNN runs the CNN once on the whole image, projects each proposal onto the shared feature map (receptive-field correspondence), and uses RoI pooling to get a fixed 7×7 feature per region. Classification (softmax) and box regression are heads inside one network trained with a multi-task loss, so the SVMs and disk caching are gone. Selective Search remained the bottleneck.

**C3. What is the role of λ in the Faster R-CNN loss?**
> The loss sums a classification term and a box-regression term that can live on very different numeric scales. Without weighting, the larger loss dominates the gradients, so the network improves only that task. λ rescales the regression term so both contribute comparably, e.g., λ = 1/100 if regression is ~100× larger. It can also deliberately prioritise one task, or be scheduled across epochs. It is not a regulariser.

**C4. Explain NMS. Is it learned?**
> NMS is a greedy post-processing step. Sort boxes by confidence, keep the highest, remove every remaining box whose IoU with it exceeds a threshold (e.g., 0.5), and repeat with the remaining boxes. It is applied per class. It is not learned: it has no trainable parameters. A drawback is that in crowded scenes it can suppress a genuine object that overlaps heavily with another.

**C5. State two limitations of YOLOv1.**
> (1) Each grid cell predicts only B = 2 boxes and one class, so many small or crowded objects whose centres fall in the same cell cannot all be detected (e.g., a flock of birds or cars queued in front view). (2) The coarse 7×7 grid and predefined box shapes generalise poorly to small objects and unusual aspect ratios. The loss also treats errors in small and large boxes similarly, even with the √w, √h fix.

**C6. Anchor-based vs anchor-free detection.**
> Anchor-based detectors (Faster R-CNN, SSD, YOLOv2–v7) place predefined boxes of several scales and aspect ratios at every location and regress offsets from them. This needs hand-tuned anchor shapes (or k-means) and many redundant boxes. Anchor-free detectors (FCOS, YOLOv8+) predict, for each feature-map point, its distances to the four box sides (left, top, right, bottom) plus a class score. This removes the anchor hyper-parameters and simplifies label assignment.

**C7. Why is mAP preferred over accuracy for detection?**
> A detection can fail in three ways: right class with a wrong box, right box with a wrong class, or no box at all. Plain accuracy captures neither localisation nor the confidence ranking. AP summarises the whole precision–recall curve as predictions are ranked by confidence, with a true positive defined by an IoU threshold. mAP averages AP over classes (and over IoU thresholds 0.5–0.95 in COCO), so it rewards both correct classes and well-fitting boxes.

**C8. Precision vs recall — which would you optimise for cancer screening, and why?**
> Recall. A missed cancer (false negative) can be fatal, whereas a false positive only triggers a follow-up test by a doctor. So we lower the confidence threshold to catch nearly all positive cases, accepting lower precision. In contrast, for an autonomous lane-change decision, a wrong "safe" call can kill, while a missed opportunity costs nothing, so precision is preferred.

**C9. What are object queries in DETR?**
> Object queries are N learned embeddings (e.g., 100) fed into DETR's transformer decoder. Each query attends to the encoder's image features through cross-attention (queries give Q; the encoder gives K and V) and is decoded in parallel into one prediction: a class (or "no object") plus a box. Hungarian matching assigns each ground-truth object to exactly one query during training, so each query learns to specialise, and no anchors or NMS are needed.

**C10. Why do transformers need positional encoding, and how does ViT turn an image into tokens?**
> Self-attention is permutation-invariant: shuffling the tokens gives the same set of outputs, so spatial order is lost. Positional encodings (learned in ViT, or sinusoidal) are added to the tokens to inject position. ViT splits the image into P×P patches (224/16 gives 196), flattens each patch (16·16·3 = 768 values), linearly projects it to D, adds position embeddings and prepends a learnable [CLS] token whose output is used for classification.

**C11. LayerNorm vs BatchNorm.**
> BatchNorm normalises each feature across the samples in a batch, so it depends on batch size and behaves differently at train and test time. It suits CNNs. LayerNorm normalises across the features of a single token or sample, independent of the batch. Transformers use LayerNorm because sequence lengths vary, batches can be small, and it stops a token with large magnitude from dominating the attention dot products.

**C12. Why does DETR do worse on small objects but better on large ones?**
> DETR's encoder applies global self-attention over the whole image, which helps reason about large objects and context (AP_L 61.1 vs 53.4 for Faster R-CNN-FPN). But it works on a single low-resolution feature map (1/32 stride) without a feature pyramid, so small objects occupy only a few tokens and their detail is lost (AP_S 20.5 vs 26.6). Deformable DETR adds multi-scale features to fix this.

**C13. Explain the two-stream network for action recognition.**
> It separates appearance from motion. The spatial stream is a CNN on a single RGB frame (3×H×W) that recognises objects and scene. The temporal stream is a CNN on stacked optical flow, 2(T−1)×H×W (dx and dy for each consecutive frame pair), that captures motion. The two class scores are fused at the end. Optical flow highlights local motion that a single frame cannot show.

**C14. What is hard negative mining and why does SSD need it?**
> SSD evaluates 8732 default boxes, and almost all of them are background, so the loss would be dominated by easy negatives and the model would learn to say "background". Hard negative mining keeps only the negatives with the highest confidence loss (the boxes the model wrongly thinks contain an object), at about a 3:1 ratio to positives. This balances the classes and focuses learning on confusing examples.

**C15. Why does U-Net use skip connections?**
> Down-sampling in the encoder gives context (what is in the image) but loses precise spatial detail (where the boundary is). Skip connections concatenate high-resolution encoder feature maps with the up-sampled decoder maps at the same scale, so the decoder can recover sharp boundaries. This is essential for pixel-accurate masks, e.g., in medical imaging.

**C16. Why is a single-frame CNN a "strong baseline" for video classification, and what does it miss?**
> Many actions can be recognised from appearance alone: a pool implies swimming, a track implies running. Averaging the per-frame predictions of a 2D CNN is cheap and can use ImageNet pretraining. It ignores temporal order and motion, so it cannot separate actions that differ only in movement (opening vs closing a door). 3D CNNs, two-stream networks or video transformers capture that.


---

# PART 12 — Traps, Transcript Corrections & Last-Night Checklist

## 12.1 Transcript/slide corrections (what the auto-transcript or slides got wrong)

The transcripts are speech-to-text and contain errors. Use the **left column** in the exam.

| Correct term / fact | What the transcript or slide said |
|---|---|
| **AlexNet**, **DINOv3**, **IoU**, **stixels**, **Non-Maximum Suppression** | "LXNet", "Dyno V3", "IUU/IAU", "stick cells", "non-microsoft operation" |
| **R-CNN** takes 47 s per image at test time (VGG), and training takes ≈ 3 days | Session 7 said "Faster R-CNN was taking around 47 seconds" |
| **CutMix** = paste a rectangular patch of one image into another, with labels mixed by area. **MixUp** = blend two whole images (e.g., 30% / 70%) with blended labels | The class explanation of "CutMix" actually described **MixUp** |
| **ViT is encoder-only**; ViT-Base = **12 encoder layers** | "6 layers encoder + 6 layers decoder" (that is the original Transformer) |
| **ImageGPT** reduced **24-bit** colour to a **9-bit (512-colour)** palette | "reduce it to 24 bit" |
| Receptive field of stacked 3×3 convs = **3 → 5 → 7** for 1, 2, 3 layers | The slide said a 3×3 conv gives a "5×5 region", then "7×7". State your convention and show the rule: **+2 per 3×3 layer** |
| AP example: interpolated precision at **r = 0.4 is 1.00**, and AP₁₁ = **0.591** | The slide table printed 0.75 at r = 0.4 (the final 0.591 is right) |
| YOLOv2 **already used k-means anchors** | A YOLOv3 slide panel claimed YOLOv2 had "handcrafted anchors" |
| SSD300 = **72.1% mAP at 58 FPS** vs Faster R-CNN **73.2% at 7 FPS** | — (the later SSD300 version reached 74.3%; quote the slide) |
| **YOLOv10 is NMS-free; YOLOv12 still uses NMS** | Easy to mix up in multi-select questions |

## 12.2 Classic traps (multi-select killers)

1. **NMS is NOT learned.** It is post-processing (the teacher stressed this twice).
2. **There is no TN in detection.** That is why there is no "accuracy".
3. **Three different IoU thresholds** — don't mix them:
   - **RPN labelling:** > 0.7 positive, < 0.3 negative
   - **SSD matching:** > 0.5
   - **Evaluation TP:** ≥ 0.5 (VOC / AP₅₀) or 0.5:0.95 (COCO)
   - **NMS:** ~0.45–0.7
4. **Loss-balancing weights:**
   - Faster R-CNN **λ** (scale balancing; not regularisation)
   - SSD **α = 1**
   - YOLOv1 **λ_coord = 5, λ_noobj = 0.5**
5. **YOLOv1 class probabilities are per CELL**, shared by its 2 boxes. Confidence is per BOX.
6. **Softmax = multi-class** (exactly one label). **Sigmoid per class = multi-label** (YOLOv3+).
7. **Fast R-CNN still uses Selective Search.** Only **Faster** R-CNN uses the RPN.
8. **R-CNN warps image crops; Fast R-CNN pools feature-map RoIs.**
9. **SSD has no proposal stage.** It still uses anchors ("default boxes") and NMS.
10. **DETR has no anchors and no NMS.** It **does** use a CNN backbone. It is **better on large** objects and **worse on small** ones.
11. **Cross-attention:** Q comes from the **decoder**; K and V come from the **encoder**.
12. **LayerNorm** works per token. **BatchNorm** works across the batch.
13. **ViT lacks inductive bias**, so it needs huge pretraining data (14M–300M images).
14. **Dice ≥ IoU always**; they are equal only at 0 and 1.
15. **Raising the confidence threshold:** precision ↑, recall ↓. **Raising the IoU threshold:** AP ↓.
16. **Semantic segmentation doesn't separate instances. Instance segmentation ignores stuff. Panoptic does both.**
17. **Mask R-CNN = Faster R-CNN + mask head + RoIAlign**, with a per-class sigmoid mask (not softmax).
18. **Two-stream optical-flow channels = 2(T−1).**
19. **Divided space-time attention:** per-token cost O(NT) → O(N + T). **Tubelets** reduce the *number* of tokens.
20. **An auxiliary head (YOLOv7) is used only in training.** An ensemble also uses its models at test time.

## 12.3 Memory hooks

- **Task ladder:** *What → Where (one) → What + Where (many) → Every pixel.*
- **R-CNN family:** *"Warp 2000 → Pool once → Propose with a net."*
- **Two-stage vs one-stage vs DETR:** *"Propose-then-refine / Look once / Ask N questions."*
- **YOLO versions:**
  - v2 "**K**-means + **B**N"
  - v3 "**3** scales, **sigmoid**"
  - v4 "**Bags** (freebies / specials) + **CIoU**"
  - v7 "**Aux** head"
  - v8 "**Anchor-free** + DFL"
  - v10 "**No NMS**"
  - v12 "**Attention**"
- **PAN vs FPN:** *FPN sends "what" down; PAN sends "where" back up.*
- **Precision vs recall:** *Precision = "don't cry wolf"; Recall = "don't miss the wolf".*
- **Attention:** *Query asks, Key answers, Value delivers.*

## 12.4 Last-night checklist (tick each one)

- [ ] Draw the task ladder and the segmentation types with an example each
- [ ] Explain why not segment everything (compute, annotation, latency)
- [ ] HOG pipeline (pixels → cells → blocks) + compute 3,780
- [ ] R-CNN → Fast → Faster: what each one fixed (table 2.10)
- [ ] RoI pooling: 512 × 7 × 7 and why it keeps the FC weights usable
- [ ] RPN: 3×3 conv → 512-d vector; k = 9 anchors; 0.7/0.3; the loss with λ and p*
- [ ] SSD: 6 maps → 8,732; scales; hard negatives 3:1
- [ ] YOLOv1: grid, responsibility, 7 × 7 × 30, confidence formula, 4-term loss, λ's, limitations
- [ ] YOLO v2–v12 table (one idea each)
- [ ] Anchor-free decode + DFL
- [ ] IoU, NMS (code + hand trace)
- [ ] P/R/F1 + when to prefer each (cancer vs lane change)
- [ ] AP 11-point & all-point on the 5-detection example (0.591 / 0.55)
- [ ] mAP, COCO AP, AP₅₀, AP_S/M/L
- [ ] Attention formula, shapes, √d_k, multi-head, LayerNorm vs BatchNorm
- [ ] ViT 7 steps; 196 / 197 tokens; ViT-B numbers
- [ ] DETR architecture, object queries, Hungarian matching, no NMS, results table
- [ ] Video: 4D tensor, late fusion vs 3D CNN, two-stream 2(T−1), I3D inflation, divided attention, tubelets
- [ ] Segmentation: FCN, U-Net, Mask R-CNN (RoIAlign), Dice / mIoU / PQ
- [ ] Case-study 7-line template
