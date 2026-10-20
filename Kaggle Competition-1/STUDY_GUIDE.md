# Study guide — "Find Homography So Dope" (your submission, explained)

Everything here was checked against your files. Numbers marked **[re-checked]** were recomputed
from your CSVs / code while writing this guide, not just copied from your reports.

---

## The Kaggle question — what you were actually asked

**One sentence:** you are given two photos of the same scene, taken from different positions or under
different lighting. For every pair, predict the 3×3 **homography matrix H** that says where each pixel of
photo 1 lands in photo 2.

**What you were given** (competition README, counts **[re-checked]** on your files)

| Item | Detail |
|---|---|
| Train | 28 scenes × 6 images. Pairs are always (image 1, image k), k = 2..6, so 140 pairs, each with its true H in `train.csv` |
| Test | 12 scenes → 60 pairs in `test.csv`, **no answers** |
| Split | Whole scenes go to train *or* test, so near-duplicate photos can't leak between them |
| Image size | from about 520×420 up to 3088×2056 |
| Method | Anything allowed: classical (SIFT/ORB + RANSAC), learned (CNN), hybrid, pretrained or generative models |

**What you submit.** A `submission.csv` with one row per test pair:
`pair_id, h11, h12, h13, h21, h22, h23, h31, h32` (h33 is fixed to 1). You also send your instructor a copy
named `<roll_number>_submission.csv`. Separately, the course times your
`predict(image_1_path, image_2_path) -> 3×3 matrix` function with `benchmark.py` on its own machine.
**Runtime is a separate column and is not part of the score.**

**How you are scored.** The metric does not compare matrix entries. It warps 5 fixed points of image 1
(the four corners and the centre) with your H and with the true H, divides by image-2 size, and averages
the 5 distances. That is the **error**. Kaggle shows `100 · max(0, 1 − error/0.2)`, so higher is better.
Example: every probe lands 5 px off on a 1000 px image → error 0.005 → score 97.5.

**Why it is hard**
* Wrong matches: repeated bricks or windows look alike and fool the matcher.
* Extreme viewpoint and low overlap: the corner probes are *extrapolated* beyond what was matched.
* Lighting changes can break matching, yet the right answer for those scenes is still the identity matrix.
* Only 28 training scenes, which is far too few to train a network to output H directly.

**What the question is really testing:** the Module 1–3 toolkit (feature detection and matching, robust
fitting with RANSAC-type estimators, learned matchers), plus validating honestly on held-out scenes.
§1 below says the same thing in more detail, and §0 comes first because it records which parts are yours.

---

## 0. Read this first — what is yours and what is not

Your **submitted file** is `outputs/24bcs10134_submission.csv`. It is byte-identical to
`outputs/submission_v6.csv`, and the notebook cites its public score, **99.55433**. So the
submission is **v6** (`src/solution_v6.py`), not v3/v4/v5. **[re-checked]**

| Part | Whose work | Evidence |
|---|---|---|
| v1 → v5 (RootSIFT, warp-refine, regimes A/B, ORB study, LOSO, 13+ experiments) | **Yours** | `REPORT.md`, `REPORT_V4.md`, `outputs/*`. Public LB 98.44. Not the final. |
| v6 candidate cascade: SIFT-FLANN 0.78 + CLAHE-SIFT + SuperPoint/LightGlue + LoFTR(<40) + `10·n − rmse + area/1e6` + gated ECC (2 % / 70 %) | **Your friend's Colab notebook** (`data/friends-answer-for-reference/`) | Same constants, same formulas, same gating rules, line for line. |
| Huber-LM polish (`geom.lm_refit`, hand-written), identity snap, CPU port, the train-vs-test analysis | **Yours** | Snap and LM exist in your v2/v3 logs *before* v6 (12-Sep vs 13-Sep). |

**Why this matters for the exam.** The docstring in `solution_v6.py` says "Reimplementation (not a
copy)", and the notebook describes the cascade as your approach without crediting the reference.
The code is functionally the same as your friend's cascade, so please do **not** say "I designed
the cascade" or "it's not a copy". The safe, truthful line is:

> "My own pipeline (v1–v5) topped out at 98.44. A peer's reference notebook used a
> multi-candidate cascade that did better on the hidden set, so I adopted that architecture and
> added two things I had already validated: exact-identity snapping and a Huber-LM refit. I then
> analysed *why* train score and leaderboard score disagreed."

Check your course's collaboration policy; if it requires acknowledgement, add a credit line to the
notebook (I can do that for you — say the word). I can't tell from the files what the rules are.

Also: `outputs/submission_v7.csv` has **no source code, no log, and no mention in either report**
(28/60 pairs differ from v6, by ≤ 0.0034). It is not what you submitted. If asked, don't discuss it.

---

## 1. The task in plain words

Each scene = 6 photos of the same thing. Pairs are (image 1, image k), k = 2..6.
You output a 3×3 matrix **H** that tells you, for any pixel in image 1, where it lands in image 2.

* **Homography** = the transformation relating two views of a *flat* surface (or two views from a
  camera that only rotates). 8 unknowns (h33 is fixed to 1 because H only matters up to scale), so
  4 point matches are the minimum; more are used and the noise averaged out.
* Two kinds of scene exist (**[re-checked]** on train: 14 scenes, 70 pairs have *exactly* the identity
  matrix as truth; the other 14 scenes have none; no scene is mixed):
  * **Illumination scenes** — camera doesn't move, only lighting changes → true H = identity.
  * **Viewpoint scenes** — camera moves → real homography, sometimes extreme (grazing angles).
* README never names the dataset. "HPatches-style" in your notebook is your *inference* (6-image
  sequences, illumination vs viewpoint split, exact identity). Say "consistent with", not "is".

### The metric (memorise this)

Take 5 points of image 1: four corners + centre. Push them through your H and the true H.
Divide by image-2 width/height. Average the 5 Euclidean distances → **error**.
Leaderboard score = `100 · max(0, 1 − error/0.2)`.

* **0.001 of error = 0.5 points.** Your final 99.554 ⇒ mean error ≈ 0.00089 (≈ 0.9 px on a 1000-px image).
  v3's 98.439 ⇒ ≈ 0.00312.
* Consequence 1 — **corners are extrapolated.** If your matches only cover half the image, H can be
  perfect there and wrong at the far corners.
* Consequence 2 — **identity pairs are free.** If you return exactly `eye(3)` on an identity pair,
  error is exactly 0.

---

## 2. The final pipeline (v6), in the order it runs

```
for each pair (grayscale, FULL resolution):
  A  SIFT 8000 feats → FLANN → Lowe ratio 0.78 → MAGSAC++ (1.5 px, best of 3) → gated ECC
  B  same, on CLAHE-equalised images (no ECC)
  C  SuperPoint + LightGlue (side ≤ 2048) → MAGSAC++ (1.0 px) → gated ECC
  D  ONLY if best candidate has < 40 inliers: LoFTR (side 840) → MAGSAC++
winner = highest   score = 10·inliers − rmse + hull_area/1e6
winner → Huber-LM refit on its own inliers
if probe-shift from identity < 0.010 → return exact identity
```

| Step | What it is | Why it is there (the sentence to say) |
|---|---|---|
| SIFT | Scale/rotation-invariant keypoints + 128-D gradient descriptors | Classic, accurate localisation, handles scale & rotation; works for most pairs |
| FLANN | Fast approximate nearest-neighbour search (5 KD-trees, 64 checks) | Brute force on 8000×8000 descriptors is slow |
| Lowe ratio 0.78 | Keep a match only if best distance < 0.78 × second-best | Throws away ambiguous matches (repetitive texture) |
| MAGSAC++ | RANSAC-family robust estimator; instead of one hard inlier threshold it averages over a range of noise levels | Matches contain outliers; plain least squares would be wrecked. Less sensitive to the threshold |
| CLAHE | Local contrast equalisation (clip 2.0, 8×8 tiles) | Gives a second chance on low-contrast / lighting-changed pairs |
| SuperPoint + LightGlue | CNN keypoints + transformer matcher | Handles wide baselines where SIFT descriptors stop matching |
| LoFTR | Detector-free dense transformer matcher | Last resort when everything is starved (slowest, so gated) |
| Gated ECC | Dense photometric alignment (maximises image correlation), *accepted only if sparse-match RMSE improves ≥ 2 % and ≥ 70 % of inliers survive*; skipped if median displacement < 1.5 px | Sharpens the fit where sparse matches are thin; ungated it drifts on repetitive texture |
| Candidate score | `10·n_inliers − rmse + min(area,1e7)/1e6` | Pick the candidate with the most geometric support |
| Huber-LM | Levenberg-Marquardt on the 8 parameters, minimising geometric transfer error with Huber loss (δ = 2 px) | Turns a minimal-sample RANSAC answer into a precise all-inlier fit; Huber stops stragglers dominating |
| Identity snap | If the 5 probes move < 0.010 from where identity would put them → `eye(3)` | Half the pairs are exactly identity; snapping makes their error exactly 0 |

If SuperPoint/LightGlue/LoFTR can't be imported, the code catches the exception and falls back to
the SIFT candidates — it degrades instead of crashing.

---

## 3. Decision by decision: why, evidence, honest caveat

**D1. Identity snap, tolerance 0.010 — yours, and the best-validated decision.**
* Why: half the data is identity truth (14/14 train illumination scenes; 6 test scenes were snapped,
  30/60 pairs **[re-checked]**: scenes 008, 009, 013, 025, 039, 040, all 5 pairs each).
* Why 0.010 (**[re-checked]** today): on the 69 normal identity train pairs, the *unsnapped*
  estimate sits at most 0.0063 from identity; the *closest real non-identity pair* in train is
  0.034 from identity. 0.010 sits in the empty gap: ~1.6× above estimator noise, ~3.4× below the
  smallest real motion.
* How much is it worth? (**[re-checked]**: I ran the v6 SIFT→MAGSAC→LM path with no snap on the 70
  identity pairs.) Median error 0.0011, mean 0.0117 (one catastrophic pair: `scene_030_1_6`, 0.72, where
  matching fails under extreme lighting). That is ≈ 0.4 leaderboard points even ignoring the
  catastrophe, up to ≈ 2.9 with it. So "the snap is the single biggest win" **is** supported.
* Caveat: a genuine viewpoint pair that moved < 0.010 would be wrongly snapped, costing ≤ ~0.01
  error. None exists in train.
* Also: the snap runs last and only *checks* the final H; it never overrides a good non-identity answer.
  The cascade is what rescues `scene_030_1_6` (final v6 error on all identity scenes is exactly 0
  **[re-checked]**).

**D2. MAGSAC++ as the estimator — yours (experiment #6).** On identical matches:
MAGSAC 0.0200 ≤ USAC_ACCURATE 0.0203 < RANSAC 0.0277 ≪ LMEDS 0.0579 (mean error, train).
The 1.5 / 1.0 px thresholds themselves are inherited; your own sweep of 2 vs 4 px was "within noise".

**D3. Full resolution grayscale — supported by your experiment #3.** Downsizing to 900 px max
side → error 0.0176 vs 0.0042 (4× worse, "catastrophic"); 1600 gave no gain at 3× cost.

**D4. Ratio 0.78 — inherited.** Your own pipeline chose **0.90** (with mutual matching) after a
sweep; both work because the estimator downstream rejects outliers. Don't claim you tuned 0.78.

**D5. Learned matchers (LightGlue, LoFTR) — adopted from the reference, consistent with your findings.**
Your experiment #12: LoFTR wins on exactly one train pair and can't be chosen by NCC; 8–13 s/pair.
In v6 it is only a *fallback chosen by inlier count* (< 40), which sidesteps the "can't select it"
problem you found. LightGlue you had already tried as an extra candidate in v5 (`lg.py`).

**D6. Gated ECC — your finding, inherited gate.** Your #10/#11: ECC optimises the same objective NCC
measures, so NCC can't judge it; it only helped when gated. v6's gate (sparse-RMSE improves ≥ 2 %,
keeps ≥ 70 % inliers) is the reference's and uses a *geometric* judge instead — a good fit to your finding.

**D7. Candidate score — inherited, with real weaknesses you should know.**
* `10·n_inliers` dominates. The convex-hull term is capped at 1e7/1e6 = **10 points = one inlier's
  worth**, so it's a tie-breaker. The notebook's claim that it "protects the corners" is overstated.
* Inlier counts come from *different match sets* (SIFT yields many more raw matches than LoFTR), so
  "more inliers" partly means "more matches", not "better H". Your own experiments (#9, #10, patch
  similarity study) found that photometric scores cannot rank close candidates either — selection of
  candidates in the 0.005–0.07 error band is a genuinely open problem in your reports.

**D8. Huber-LM — yours, small effect.** Measured gain in your table: −0.00002 mean error (≈ 0.01
points). Kept because it's free and principled (minimises the thing the metric measures). Don't oversell it.

**D9. Choosing v6 over v5 although train score is worse (96.88 vs 99.22) — the key reasoning question.**
See §4.

---

## 4. Why v6 when train says v5 is better

Facts (**[re-checked]**):

| | train score | public LB |
|---|---|---|
| v3 | 99.2096 | 98.43944 |
| v5 | 99.2221 | 98.44643 |
| **v6** | **96.8821** | **99.55433** |

1. **Train error is dominated by one scene absent from test.** v6's train error: 0.873 total, of
   which `scene_033` (repetitive tiled facade, grazing angle) = **0.682 (78 %)**; pairs 033_1_6 =
   0.383 and 033_1_5 = 0.295. v3 on the same scene: 0.063. Per your regime analysis, the test set
   has no equivalent scene.
2. **The leaderboard gain is essentially one test scene.** v3 and v6 disagree most on
   `scene_023` (mean 0.031 per pair vs ≤ 0.002 elsewhere): scene_023 alone is ≈ 88 % of their
   total disagreement. The LB moved by ≈ 0.0022 in mean error; scene_023 alone can account for up
   to ≈ 0.0026. In your regime analysis, scene_023 resembles train `scene_006` (alley mural receding
   to grazing angle) — the case where learned features win.
3. **Lesson you drew:** train mean error is a weighted average with the wrong weights when train and
   test have different scene mixes; judge by regime-weighted performance on the regimes test contains.

Be honest about the limits of that argument:
* It is **partly post-hoc**: you picked v6 because the public leaderboard said so, then explained why.
  The explanation is plausible and consistent with the numbers, but you cannot verify scene_023
  without test labels.
* The README doesn't mention a public/private split. If the 99.55 is computed on all 60 test pairs,
  there's no hidden remainder to overfit; if there is a private split, expect some drift.
* The whole jump riding on one scene means variance is high — say so before an examiner does.

---

## 5. What you tried and rejected (shows scientific method — know 5–6 of these)

| # | Idea | Outcome | Lesson |
|---|---|---|---|
| 3 | Smaller images (900 px) | 4× worse | Resolution matters for corner accuracy |
| 6 | RANSAC / LMEDS / USAC_ACCURATE vs MAGSAC | MAGSAC best | Estimator choice, measured on identical matches |
| 9–10 | Pick between candidates with NCC (colour/gray/gradient, hi-res) | Worse (0.0025–0.0034 vs 0.0021) | Photometric scores can't rank fine candidates; ECC optimises the judge's own objective |
| 12 | LoFTR as a general candidate source | Wins 1 pair, unselectable, 8–13 s | Fine as a fallback, not as a default |
| ORB | ORB instead of SIFT | Mean 0.033 vs 0.0016 | Binary descriptors collapse on hard pairs; useful only as a *disagreement signal* |
| LOSO | Leave-one-scene-out validation of every new rule | Fusion picked 28/28; rescue/cluster/NCC-swap rejected | Hold out whole scenes (images within a scene are near-duplicates, same reason the README splits by sequence) |
| Final addendum | Patch-wise NCC arbiter, grazing-angle fitting, deep-pyramid ECC | All rejected | Remaining error = low-overlap corner extrapolation; no safe fix found |

---

## 6. Numbers to remember

* Metric: `score = 100(1 − err/0.2)`; 0.001 error = 0.5 pt.
* Data: 28 train scenes (140 pairs), 12 test scenes (60 pairs); 6 images per scene.
* Final: **public 99.55433**; train (self-check) 96.88 **[re-checked]**; 30/60 test pairs exactly identity.
* Your own best classical pipeline (v5): public 98.446, train 99.22.
* Runtime (your measurements): v3 0.9 s/pair; v4 1.8 s/pair; **v6 13.4 s/pair on CPU** (Ryzen 5 5600H, no GPU).
  README says runtime is a *separate column*, not part of the score. You chose accuracy over speed;
  the notebook names the easy speed-up (skip learned matchers when SIFT already has strong support).

---

## 6b. Quick concept glossary (one line each)

* **RANSAC** — repeatedly fit H to 4 random matches, keep the one most others agree with.
* **Inlier** — a match consistent with the fitted H within a pixel threshold.
* **DLT** — linear method to solve H from ≥ 4 matches (Hartley normalisation first, for stability).
* **RootSIFT** — L1-normalise SIFT then square-root; matches better than raw SIFT (used in v1–v5).
* **NCC** — normalised cross-correlation, a similarity score between two aligned images.
* **ECC** — enhanced correlation coefficient; direct image alignment, no keypoints.
* **Huber loss** — quadratic for small residuals, linear for large ones, so outliers pull less.
* **LOSO** — leave-one-scene-out cross-validation.

---

## 7. Hard questions and model answers

**"Which parts did you write?"** → see §0. Own pipeline v1–v5, the metric analysis, the identity snap,
the hand-written Huber-LM, the regime/LOSO analysis; the cascade structure came from a peer notebook.

**"Why is your train score lower than your earlier version but the leaderboard higher?"** → §4.

**"Why not just use deep learning end-to-end?"** → Metric is geometric and corner-based; classical
matching + robust fitting gives sub-pixel precision and is interpretable. Learned *matchers* are used
where classical descriptors fail; training a regressor on 28 scenes would overfit.

**"What is the snap, and could it hurt?"** → D1. It can only hurt if a true viewpoint pair is within
0.010 of identity; closest in train is 0.034, noise is ≤ 0.0063.

**"What would fail?"** → Grazing-angle / low-overlap pairs where corners are extrapolated;
repetitive texture; scene_023-type pairs; also candidate selection that compares inlier counts of
different match sets.

**"How would you speed it up?"** → Run SIFT first; skip LightGlue/LoFTR when it already has many,
well-spread inliers; use a GPU.

**"Did you use ground-truth leakage?"** → No. Your report says you deliberately did not look up the
published homographies of the source dataset. Say that. Don't make claims about what other
leaderboard entries did — that's speculation.

---

## 8. Things not to claim

* "I designed / invented the cascade" or "it's not a copy" (see §0).
* "I tuned ratio 0.78 / the 10·n score / the 2 %/70 % ECC gate" — those are inherited.
* "The hull-area term protects the corners" — it's worth at most one inlier.
* "Train error predicts leaderboard" — your own results show it doesn't here.
* Anything about v7.
