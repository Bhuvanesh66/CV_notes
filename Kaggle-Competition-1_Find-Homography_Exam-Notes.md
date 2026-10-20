# Kaggle Competition 1 — "Find Homography So Dope": Exam Notes

**Course:** SST Computer Vision (Scaler) · **Assignment:** Kaggle Competition-1 (homography estimation) · **Your submission:** v6, public score **99.55433**

**Built from:** your own code and reports in `Kaggle Competition-1/` (`solution_v6.py`, `geom.py`, `REPORT.md`, `REPORT_V4.md`, the submission notebook), plus numbers recomputed from your CSVs. Deeper walk-through: `Kaggle Competition-1/STUDY_GUIDE.md`.

> **Preview tip (VS Code):** press `Ctrl+Shift+V` to render the tables.

---

## 0. What sir said, and how to read it

> **Sir (as you told me):** *"Will just test the approach you used for assignment and Descriptive and mcq."*

My reading of that (an **inference**, not something sir said word for word):

| Point | What it means for you |
|---|---|
| "the approach you used" | Questions are about **your submitted approach (v6)**: what each step does, why it is there, what it cost, where it fails. Not about generic homography theory in the abstract. |
| "Descriptive and MCQ" | Expect **Section A** (single-correct, 1 mark), **Section B** (multi-select, 2 marks, all-or-nothing) and **Section C** (descriptive, ≤ 100 words, 2 marks). **No coding (D) or case-study (E)** from the Kaggle part, as I read it. |
| Numbers | Small calculations may still appear inside an MCQ (e.g. error → leaderboard score). Know the formula. |

**Not known:** how many marks or questions come from Kaggle. Check `classes/question-pattern.pdf` and ask sir if you can.

### Authorship line (read this once, carefully)

"The approach you used" is the **v6 cascade**. Its structure (four candidate matchers, the `10·n − rmse + area/1e6` score, the ECC acceptance gate, the numeric settings) matches your friend's reference notebook. What is **yours**: the Huber-LM refit (hand-written in `geom.py`), the identity snap, and the whole v1–v5 pipeline and analysis. Details and a truthful one-paragraph answer are in `STUDY_GUIDE.md` §0. Do not say "I designed the cascade" or "it is not a copy". Check your course's collaboration policy.

---

## 1. Expected question types (my prediction, ranked by likelihood)

| # | Type | Likely format | Example | What earns the mark |
|---|---|---|---|---|
| 1 ★ | **Describe your approach** | C | "Explain the approach you used for the Kaggle assignment." | Names every stage in order, one reason each, one number, one limitation. |
| 2 ★ | **Justify a design choice** | C | "Why MAGSAC++ and not RANSAC?" "Why an identity snap?" | Choice → reason → **evidence** (a measured number). |
| 3 ★ | **Role of a component / what runs when** | A, B | "LoFTR runs only when…" "ECC is accepted only if…" | Exact rule and threshold. |
| 4 ★ | **Metric reading** | A, C | "Error 0.004 → score?" "Why probe corners?" | `100·(1 − err/0.2)`; corners are extrapolated. |
| 5 | **Core concepts** | A, B | DOF of H, minimum correspondences, ratio test, CLAHE, ECC, Huber | One-line definitions (Part 3 glossary). |
| 6 | **Limitations and failure modes** | B, C | "Name limitations of your approach." | Grazing angle, low overlap, repetitive texture, runtime, selection weakness. |
| 7 | **Validation and generalisation** | A, C | "Why validate by scene?" "Why did train fall but leaderboard rise?" | Scene-level leakage; train and test scene mixes differ; honest caveat. |
| 8 | **Alternatives compared** | B, C | SIFT vs ORB vs SuperPoint/LightGlue vs LoFTR | Strength, weakness, cost. |
| 9 | **What failed / what you would change** | C | "Two experiments that failed." "How to speed it up?" | Specific experiment + number + lesson. |

**Answer recipe (≤ 100 words):** *what it is → why you used it → one measured number → one limitation.* Marks go to justifications ("because…"), not to long lists.

---

## 2. Your approach at three sizes

### One sentence
For each image pair I build several candidate homographies (SIFT, CLAHE-SIFT, SuperPoint+LightGlue, LoFTR as a fallback), keep the one with the most geometric support, polish it with a robust Huber-LM refit, and snap near-identity results to the exact identity.

### The pipeline (memorise the order)

```
grayscale, FULL resolution
A  SIFT (8000 feats) → FLANN → Lowe ratio 0.78 → MAGSAC++ (1.5 px, best of 3) → gated ECC
B  same on CLAHE images (clip 2.0, 8×8 tiles), no ECC
C  SuperPoint + LightGlue (side ≤ 2048, 2048 kpts) → MAGSAC++ (1.0 px) → gated ECC
D  ONLY if best candidate has < 40 inliers → LoFTR (side 840, conf ≥ 0.3) → MAGSAC++ (1.5 px)
winner = max  score = 10·n_inliers − rmse + min(hull_area, 1e7)/1e6     (inliers counted at 3 px)
winner → Huber-LM refit on its own inliers (Huber δ = 2 px)
if mean probe shift from identity < 0.010 → return exactly eye(3)
```

### Version history (your own work, v1 → v6)

| Ver | What changed | Train score | Public LB | Time/pair |
|---|---|---|---|---|
| v1 | RootSIFT + mutual matching + MAGSAC | 90.36 | — | — |
| v2 | + warp/rematch refine, ASIFT escalation with colour NCC, identity snap | 98.94 | — | — |
| v3 | + Huber-LM, gated ECC, regimes A/B | 99.21 | 98.439 | 0.9 s |
| v4 | + ORB disagreement and median fusion | 99.215 | 98.443 | 1.8 s |
| v5 | + LightGlue in the escalation pool | 99.222 | 98.446 | — |
| **v6** | **candidate cascade + snap + Huber-LM (submitted)** | **96.88** | **99.554** | **13.4 s (CPU)** |

---

## 3. Glossary and component facts (1-mark material)

| Term | One line | Key fact for MCQs |
|---|---|---|
| **Homography** | 3×3 matrix mapping points between two views of a plane (or two views from a rotating camera) | **8 DOF** (h33 = 1; defined up to scale); needs **≥ 4** correspondences (2 equations each) |
| **DLT** | Linear solve for H via SVD null-space | Hartley normalisation first, for numerical stability |
| **SIFT** | Scale/rotation-invariant keypoints from DoG scale-space | Descriptor = **128-D** (4×4 cells × 8 orientation bins) |
| **FLANN** | Fast *approximate* nearest-neighbour search | KD-trees (5), 64 checks in your code |
| **Lowe ratio test** | Keep a match only if best distance < ratio × second-best | Lowe's paper suggests 0.8 (removes ~90 % of false matches, loses < 5 % of true ones); yours: **0.78** |
| **RANSAC** | Fit H to random minimal samples (4 matches), keep the model with the largest consensus | Needs a hard pixel threshold |
| **MAGSAC++** | RANSAC family; weights points over a range of noise scales | Less threshold-sensitive; `cv2.USAC_MAGSAC`; 1.5 px (1.0 px for LightGlue); run **3×**, keep the most inliers |
| **CLAHE** | Contrast-limited adaptive histogram equalisation | Local contrast; second chance for low-contrast pairs |
| **SuperPoint** | Self-supervised CNN keypoint detector + descriptor | |
| **LightGlue** | Attention-based matcher for SuperPoint keypoints | Adaptive depth (stops early on easy pairs) |
| **LoFTR** | **Detector-free** dense transformer matcher | Slowest; only a **fallback (< 40 inliers)** |
| **ECC** | Direct image alignment maximising enhanced correlation coefficient | Starts from the feature-based H |
| **Gated ECC** | Accept ECC only if sparse-inlier **RMSE improves ≥ 2 %** and **≥ 70 % of inliers** survive | Skipped when median displacement < 1.5 px (near-identity) |
| **NCC** | Normalised cross-correlation image similarity | Could **not** rank close candidates (your experiments) |
| **Huber loss** | Quadratic for small residuals, linear beyond δ | δ = 2 px; LM = Levenberg–Marquardt on the 8 parameters |
| **Identity snap** | Return `eye(3)` if probes move < **0.010** from identity | Illumination-only scenes have **exactly identity** truth |
| **LOSO** | Leave-one-scene-out validation | Images in a scene are near-duplicates |

### The metric (★ numerical)

1. Probes: (0,0), (1,0), (1,1), (0,1), (0.5,0.5) of image 1, scaled to pixels.
2. Warp with predicted H and true H; divide by **image-2** width/height.
3. Error = mean Euclidean distance over the 5 probes.
4. **Score = 100 · max(0, 1 − error/0.2).**

| Shortcut | Value |
|---|---|
| 1 px off on a 1000 px image | error 0.001 = **0.5 points** |
| error 0.004 | 100·(1 − 0.02) = **98.0** |
| error 0.005 | **97.5** |
| Your final 99.554 | error ≈ **0.00089** (≈ 0.9 px per 1000 px) |
| v3's 98.439 | error ≈ **0.00312** |

### Key numbers

| Fact | Number |
|---|---|
| Train / test | 28 scenes (140 pairs) / 12 scenes (60 pairs); 6 images per scene, pairs (1, k), k = 2..6 |
| Identity scenes | 14 of 28 train scenes (70 pairs) exactly identity; 6 test scenes (30 of 60 pairs) snapped |
| Estimator comparison (same matches, mean train error) | MAGSAC 0.0200 ≤ USAC_ACCURATE 0.0203 < RANSAC 0.0277 ≪ LMEDS 0.0579 |
| Resolution experiment | max side 900 px: 0.0176 vs 0.0042 (≈ 4× worse); 1600 px: no gain at ~3× cost |
| ORB vs SIFT pipeline | mean error 0.0334 vs 0.0016 |
| Snap tolerance 0.010 | unsnapped identity-pair noise ≤ ~0.006; closest real non-identity pair 0.034 |
| Snap worth | ≈ 0.4 points (excluding one catastrophic pair), up to ≈ 2.9 |
| Huber-LM gain | small: ≈ −0.00002 mean error |
| v6 train self-check | 96.88; 78 % of its error is one scene (`scene_033`) |
| v6 runtime | 13.4 s/pair mean on CPU (Ryzen 5 5600H, no GPU); v3 was 0.9 s |

---

## 4. Section A — Single-correct MCQs (1 mark each)

**A1.** A homography with h33 fixed to 1 has how many free parameters?
A. 6  B. 8  C. 9  D. 4

**A2.** Minimum number of point correspondences needed to solve a homography?
A. 2  B. 3  C. 4  D. 8

**A3.** Why does the competition warp 5 probe points instead of comparing the matrix entries?
A. Matrix entries must be integers
B. H is defined only up to scale, and similar-looking entry differences can mean very different visual misalignment
C. Kaggle cannot read float columns
D. Probe points are faster to compute

**A4.** A submission has mean reprojection error 0.004. Its leaderboard score is:
A. 99.6  B. 98.0  C. 96.0  D. 99.2

**A5.** For an illumination-only pair, whose ground truth is exactly the identity, returning exactly `eye(3)` gives an error of:
A. about 0.001  B. exactly 0  C. depends on image size  D. 1

**A6.** The Lowe ratio test:
A. removes keypoints near the image border
B. discards matches whose best descriptor distance is not clearly smaller than the second-best
C. normalises SIFT descriptors
D. chooses the RANSAC threshold

**A7.** Why is a RANSAC-family estimator used rather than plain least squares on all matches?
A. Least squares cannot handle 8 unknowns
B. Matches contain outliers that would corrupt a least-squares fit
C. RANSAC is always more accurate than any refit
D. Least squares needs ground truth

**A8.** CLAHE is applied in the pipeline to:
A. resize images
B. equalise local contrast, giving SIFT a second chance on low-contrast pairs
C. convert images to colour
D. detect keypoints

**A9.** In v6, LoFTR is run:
A. on every pair
B. only on illumination pairs
C. only when the best candidate so far has fewer than 40 inliers
D. only on a GPU

**A10.** The ECC result is accepted only if:
A. its normalised cross-correlation is higher than the SIFT candidate's
B. the sparse inlier RMSE improves by at least 2 % and at least 70 % of inliers are kept
C. the image is larger than 1000 px
D. the median displacement is below 1.5 px

**A11.** The identity-snap tolerance of 0.010 means:
A. 1 % of pixels may differ
B. if the five probes move on average less than 0.010 (normalised) from identity, output exactly the identity
C. at most 10 matches may disagree
D. the RANSAC threshold is 0.010 px

**A12.** A Huber loss differs from squared error because it:
A. is linear beyond a threshold, so large residuals are down-weighted
B. ignores small residuals
C. is non-differentiable everywhere
D. requires ground truth

**A13.** Why was validation done leave-one-scene-out rather than with a random pair split?
A. It is faster
B. Images within a scene are near-duplicates, so a random split leaks and overstates performance
C. Test scenes were shuffled
D. Scenes have different file formats

**A14.** v6 had a lower training score than v5 but a higher leaderboard score. The best-supported explanation is:
A. v6 overfit the training set
B. Mean training error was dominated by one scene type (a repetitive facade at grazing angle) that is absent from test, while on test the cascade's learned matchers helped on a wide-baseline scene
C. The leaderboard is computed on the training set
D. v6 is faster

### Answers — Section A

| Q | Ans | Why the closest distractor is wrong |
|---|---|---|
| A1 | **B** | 9 entries, but scale is arbitrary and h33 = 1 removes it: 8. |
| A2 | **C** | Each correspondence gives 2 equations; 4 × 2 = 8 unknowns. |
| A3 | **B** | The others are invented. |
| A4 | **B** | 100·(1 − 0.004/0.2) = 98.0. |
| A5 | **B** | Same matrix as truth → zero error exactly. |
| A6 | **B** | It is about ambiguity of the best match, not borders or the threshold. |
| A7 | **B** | RANSAC is *not* always more accurate than a refit; that is why a Huber-LM refit follows it. |
| A8 | **B** | It is image preprocessing, not detection. |
| A9 | **C** | Gated by inlier count (< 40); it ran on CPU in your submission. |
| A10 | **B** | NCC was found unable to judge ECC. D is the *skip* rule, not the acceptance rule. |
| A11 | **B** | It is a probe-shift (metric-space) tolerance. |
| A12 | **A** | Quadratic near zero, linear beyond δ. |
| A13 | **B** | Same reason the organisers split train/test by sequence. |
| A14 | **B** | A is the common-sense answer your own analysis argues against. Caveat: you chose v6 after seeing the leaderboard. |

---

## 5. Section B — Multiple-correct MCQs (2 marks; all correct and no wrong option)

**B1.** Which are TRUE about a homography?
A. It has 8 degrees of freedom when h33 = 1
B. It exactly relates two views of a planar surface
C. It exactly relates two views from a purely rotating camera
D. It can represent any 3D scene exactly
E. It is defined only up to scale

**B2.** Which are TRUE about robust estimation in this project?
A. Classical RANSAC fits minimal samples (4 matches) and keeps the model with the largest consensus
B. MAGSAC++ reduces dependence on a single hard inlier threshold
C. LMEDS gave the lowest error in the estimator comparison
D. A final Huber-LM refit on inliers adds precision
E. MAGSAC++ was run three times and the model with the most inliers kept

**B3.** Which belong to the v6 candidate set?
A. SIFT + FLANN
B. SIFT on CLAHE-equalised images
C. SuperPoint + LightGlue
D. LoFTR (conditionally)
E. ORB (always)

**B4.** Which are TRUE about the identity snap?
A. It runs last, as a check on the final H
B. It could wrongly snap a true viewpoint pair that lies closer than 0.010 to identity
C. It needs ground truth at test time
D. It exploits that illumination-only sequences have identity truth
E. Its tolerance sits between estimator noise and the smallest real motion in train

**B5.** Why can corner probes have large error even when H fits the matches well?
A. Corners may lie outside the matched region
B. Extrapolation amplifies small parameter errors
C. Low-overlap pairs make this worse
D. More inliers always remove the problem
E. Strong perspective (grazing angle) makes it worse

**B6.** Which are TRUE about gated ECC?
A. It is dense: it maximises image correlation
B. It starts from the feature-based H
C. It is accepted unconditionally
D. It is skipped when the median displacement is below 1.5 px
E. Acceptance is judged by sparse-match RMSE

**B7.** Which are TRUE about the learned matchers?
A. SuperPoint is a CNN detector and descriptor
B. LightGlue matches keypoints with attention
C. LoFTR is detector-free
D. In your run they executed on CPU and dominated the ~13 s/pair runtime
E. They always beat SIFT

**B8.** Which are genuine limitations of the submitted approach?
A. The candidate score compares inlier counts from different match sets
B. The hull-area term is worth at most about one inlier
C. CPU runtime of ~13 s per pair
D. Much of the leaderboard gain rests on one test scene
E. It requires ground-truth H at test time

### Answers — Section B

| Q | Correct | Notes |
|---|---|---|
| B1 | **A, B, C, E** | D false: a general 3D scene is not a plane. |
| B2 | **A, B, D, E** | C false: LMEDS was worst (0.0579). |
| B3 | **A, B, C, D** | E false: ORB was a v4 side-channel experiment, not part of v6. |
| B4 | **A, B, D, E** | C false: it uses only the predicted H. |
| B5 | **A, B, C, E** | D false: inlier count says nothing about *where* they are. |
| B6 | **A, B, D, E** | C false: it is gated. |
| B7 | **A, B, C, D** | E false: SIFT is better on easy pairs and far cheaper. |
| B8 | **A, B, C, D** | E is nonsense: no ground truth exists at test time. |

---

## 6. Section C — Descriptive (2 marks, ≤ 100 words): model answers

**Structure:** *what → why → one number → one limitation.* Rewrite in your own words before the exam; the numbers are what examiners look for.

**C1. Explain the approach you used for the Kaggle assignment.**
> For each pair I build several candidate homographies: SIFT with FLANN and a ratio test, the same on CLAHE-equalised images, SuperPoint+LightGlue, and LoFTR only if fewer than 40 inliers remain. Each is fitted with MAGSAC++ and optionally refined by gated ECC. The candidate with the highest support score (10 × inliers − RMSE + hull-area bonus) wins, is polished by a Huber Levenberg–Marquardt refit, and is snapped to the exact identity if nearly identity, because illumination-only scenes have identity truth. The cascade follows a peer's reference; the snap and refit are mine.

**C2. What is a homography, and why does this task use one?**
> A homography is a 3×3 matrix with eight degrees of freedom (h33 = 1) that maps pixel coordinates between two views of a planar surface, or two views from a purely rotating camera. It acts on homogeneous coordinates, then divides by the third component. Because the scenes are approximately planar, one matrix per pair describes the geometry. At least four correspondences are needed, but many more are used so that noise averages out.

**C3. Explain the evaluation metric and why corner accuracy matters.**
> Five probe points, the four corners and the centre, are warped by the predicted and the true H, normalised by image-2 size, and the Euclidean distances are averaged. Kaggle displays 100·max(0, 1 − error/0.2). This measures what H does to pixels rather than raw entries, because H is defined only up to scale. Four of the five probes sit at the corners, so if matches cover only part of the image the fit is extrapolated there and small parameter errors become large.

**C4. Why classical matching plus a robust fit rather than a CNN regressing H directly?**
> The metric needs sub-pixel geometric precision. Matching plus a robust fit uses the actual image evidence and gives precise corners; a network regressing eight numbers would need far more than 28 training scenes and would typically be coarse. Robust fitting also tolerates wrong matches, is interpretable, and fails visibly (few inliers). Learned components are used where they help, as matchers (LightGlue, LoFTR) on hard pairs, not as end-to-end regressors.

**C5. Why MAGSAC++ rather than plain RANSAC?**
> Matches contain outliers, so a robust estimator is needed. RANSAC depends on a hard pixel threshold. MAGSAC++ weights points over a range of noise scales, so it is less threshold-sensitive and more accurate. On identical matches my mean training error was 0.0200 for MAGSAC++, 0.0277 for RANSAC and 0.0579 for LMEDS. I run it three times, keep the model with the most inliers, then refit it with Huber-LM.

**C6. Why a cascade of several matchers instead of one?**
> No single matcher wins on every pair. SIFT is precise on ordinary pairs, CLAHE-SIFT helps low-contrast images, SuperPoint+LightGlue handles wide baselines where descriptors break, and LoFTR rescues pairs where sparse methods starve. Each candidate is scored by geometric support and the best is kept. The slow LoFTR is gated by an inlier count below 40. The price is runtime, about 13 s per pair on CPU.

**C7. What is the identity snap and why is it justified?**
> Illumination-only sequences have exactly the identity matrix as ground truth: all 14 such training scenes. If the final H moves the five probes by less than 0.010 from identity, I return `eye(3)`, turning a small residual into exactly zero. The tolerance sits between estimator noise (about 0.006 at worst on training identity pairs) and the closest real viewpoint pair (0.034). It is a final check, so it never overrides a good non-identity estimate.

**C8. What is ECC and why was it gated?**
> Enhanced Correlation Coefficient alignment directly maximises image similarity over the eight homography parameters, starting from the feature-based estimate. It can sharpen fits when sparse matches cover little of the image, but ungated it drifts on repetitive texture. So its result is accepted only if the sparse inlier RMSE improves by at least 2 % while at least 70 % of inliers remain, and it is skipped for near-identity pairs. A geometric judge is used because NCC cannot rank ECC against feature fits.

**C9. v6 scored lower on training than v5 but higher on the leaderboard. Explain.**
> Mean error depends on scene mix. v6's training error was dominated by one repetitive facade scene seen at grazing angles, about 78 % of its total, a type absent from the test set. On test, v6's gain came mainly from one wide-baseline alley scene where learned matchers help. So the training mean was a misleading selection metric here. Caveat: I chose v6 after seeing the leaderboard, so the explanation is plausible but partly post-hoc, and one scene drives much of the gain.

**C10. State the main limitations of your approach and one improvement.**
> Limitations: grazing-angle and low-overlap pairs, where corners are extrapolated beyond the matched region; repetitive textures; selection by inlier count compares different match sets, and the hull-area bonus is worth at most one inlier; much of the leaderboard gain rests on one scene; CPU runtime of about 13 s per pair. Improvement: run SIFT first and skip the learned matchers when inlier support is strong and well spread, and use a GPU.

**C11. Why validate by whole scenes (leave-one-scene-out)?**
> The six images of a scene are near-duplicates, so a random pair split would put almost identical images in both training and validation and overstate performance. This is also why the organisers split train and test by whole sequences. Leave-one-scene-out holds out every pair of a scene while tuning on the rest, so each rule is judged on a genuinely unseen scene. In my experiments several tempting rules that improved the mean failed this test and were rejected.

**C12. Compare SIFT, ORB and learned matchers.**
> SIFT: float 128-D descriptors, scale and rotation invariant, accurate, moderately slow. ORB: binary descriptors, very fast but less accurate; in my study its mean error was 0.033 against 0.0016 and it collapsed on hard pairs. SuperPoint+LightGlue: learned keypoints and an attention-based matcher, better under wide baselines and appearance change. LoFTR: detector-free dense matching, good on low texture, slowest. Learned methods need PyTorch and are costly on CPU.

**C13. What does the Huber–LM refit do?**
> RANSAC-type methods fit from small random samples and a pixel threshold, so the result is slightly noisy. After the inliers are chosen, Levenberg–Marquardt minimises the geometric transfer error of all inliers over the eight homography parameters. The Huber loss is quadratic for small residuals and linear beyond 2 px, so stragglers are down-weighted rather than dominating. It is cheap, but the measured gain was small, about −0.00002 mean error.

**C14. Give two experiments that failed and what they taught you.**
> (1) Choosing among candidates with photometric similarity (NCC variants): error rose from 0.0021 to 0.0025–0.0034, because different homographies imply different overlap regions and ECC optimises the same objective NCC measures, so such scores cannot rank close candidates. (2) Downscaling images to 900 px: error rose about four-fold (0.0176 versus 0.0042), so resolution matters for corner precision. Lesson: test every idea on held-out scenes before adopting it.

---

## 7. Traps and "do not say"

| Trap | Why |
|---|---|
| "I designed / invented the cascade", "it is not a copy" | The structure and constants match your friend's reference. Credit it. |
| "I tuned ratio 0.78, the `10·n` score, the 2 % / 70 % ECC gate" | Inherited. You tuned **0.90** in your own v3 pipeline. |
| "The hull-area term protects the corners" | Capped at 10 points = one inlier. Tie-breaker only. |
| "Train error predicts leaderboard" | Your own results (v5 → v6) say otherwise. |
| "The learned matchers are always better" | They lose on easy pairs and cost 10×+ in runtime. |
| "Homography works for any scene" | Exact only for planes or pure rotation. |
| "Snap uses the ground truth" | It uses only the predicted H and the fact that illumination-only truth is identity. |
| Multi-select: "always / never / only" | Usually the wrong option. |
| Mentioning `submission_v7.csv` | No source or log; not what you submitted. |
| Claims about other leaderboard entries | Speculation. Say you did **not** look up the source dataset's published answers. |

## 8. Last-night checklist

- [ ] I can recite the pipeline order A → B → C → D → score → LM → snap.
- [ ] I know the five numbers: **0.78, 1.5 px, 40 inliers, 2 % / 70 %, 0.010**.
- [ ] I can compute score from error (and back) in 10 seconds.
- [ ] I can state one measured number for MAGSAC, resolution, ORB, snap tolerance.
- [ ] I can explain train 96.88 vs leaderboard 99.554 **with the caveat**.
- [ ] I can say truthfully which parts are mine and which came from the reference.
- [ ] I can give two limitations and one improvement.
