# Study guide: "Match the Dish to Its Rizz" (Competition-3, 24BCS10134)

How to use this: learn §1 and §2 until you can say them without notes. §3 explains the code step by step. §4 is every decision you made, with the reason, the evidence and what to say if asked. §6 lists the weak spots so the examiner can't surprise you. The exam-style questions (MCQ, multi-select, descriptive) are in `../notes/Kaggle-Competition-3_Dish-Rizz_Exam-Notes.md`.

Legend:
- **[re-run]**: I re-ran your code on your cached features and the number matched.
- **[log]**: taken from `outputs/experiments.csv`, not re-run.
- **[report]**: stated in your REPORT.md / HANDOFF.md and I couldn't check it, because I can't see the Kaggle page or your Kaggle account.
- **[inference]**: an explanation that fits the evidence but was never proven. Don't present an inference as a fact.

One thing to know about how this was produced: REPORT §6 says the dish descriptions in one *rejected* experiment (CuPL) were written with an LLM assistant. They are not in your final pipeline. I can't tell your course's rules on AI assistance. If you are asked how the work was produced, answer truthfully.

---

## 1. The Kaggle question, end to end

**One sentence:** you are given sentences like *"Show me pictures of potstickers."* and a pile of 2,000 unlabelled food photos. For each sentence, you must list the 20 photos most likely to show that dish, best first.

### What you are given

| File / folder | What's in it | Check |
|---|---|---|
| `train.csv` + `train_gallery/` | 2,000 labelled photos, 19–20 per dish. Labels contain some noise. | [re-run] |
| `validation_gallery.csv`, `validation_queries.csv`, `validation_ground_truth.csv` | 1,010 photos (exactly 10 per dish), 202 queries, and the answer key | [re-run] |
| `test_gallery.csv` + `test_gallery/` | 2,000 photos, no labels | [re-run] |
| `test_queries.csv` | 303 queries to answer, exactly 3 per dish | [re-run] |
| `train_queries.csv` | 505 example queries with their dish (5 per dish) | [re-run] |
| `category_metadata.csv` | For each of the **101 dishes**: name, a one-line definition, alternative names, and 5 paraphrase sentences | |
| `sample_submission.csv`, `validate_submission.py` | Format example and a format checker | |

Real examples from your data:
- Test queries: "A photograph of a croque madame." and "An Italian dish of very thinly sliced raw beef. The dish is called beef carpaccio."
- Validation query: "I want to see baklava."
- The dish *gyoza* has the alternative name *potstickers*. So "Show me pictures of potstickers" means gyoza.

### What you submit

A CSV with one row per test query: `query_id, image_ids`. The `image_ids` cell holds **20 photo ids separated by spaces, best first**. Your code writes exactly 20 (`common.write_sub`).

### How you're scored: R-precision

For a query whose dish has **R** relevant photos, look at your top **R** results. R-precision is the fraction of them that are relevant, then averaged over all queries.
- Validation: R = 10. Test: R = 19 or 20.
- Example: R = 20 and 19 of your first 20 are the right dish, so you score 0.95.
- Random ranking scores about R/N = 20/2000 = **0.01**.
- Order *inside* the top R doesn't matter. Only *which* photos made it in.
- The public leaderboard uses about **50% of the test queries** [report]. The rest is scored privately.
- Limit: 5 submissions per day.

### Rules that shaped your design [report]

| Allowed | Forbidden | Allowed if declared |
|---|---|---|
| Generic pretrained models and vision-language models (VLMs) | Checkpoints trained to classify these food categories; external copies of the images; matching test photos against outside collections; hand-labelling | Unsupervised use of the test gallery (rule 10b). You declare it in REPORT §4.3. |

### Why it's hard, and why it turned out easy

- **Hard:** 101 dishes, and many look alike (gyoza vs dumplings; steak vs filet mignon vs prime rib; panna cotta vs creme brulee vs cheesecake; bread pudding vs french toast). The 20 training labels per dish are noisy.
- **Easy once you see it:** every query names its dish in plain text, and big pretrained image–text models already know these dishes. So the real work is (1) reading the dish out of the query, (2) scoring photos against dish names, (3) squeezing out the last few percent using the gallery's structure.

---

## 2. The solution in 60 seconds

```
query text ──(rule-based name matcher)──► one of 101 dishes
                                              │   all 3 queries of a dish get the same list
gallery photo ─► SigLIP2-giant ┐
                 PE-Core-bigG  ┘  compare to each dish's TEXT prototype (cosine similarity)
                                  no training at all
                      │
                      ▼
      "each dish has the same number of photos" ─► Sinkhorn balancing + Hungarian assignment
                      │
                      ▼
      for each dish: photos assigned to it first, then the rest by score → top 20 ids
```

1. **Read the dish out of the query** with simple text matching. This works 100% on every labelled query set. Now each query is just "show me dish #k".
2. **Score every photo against every dish name** with two big image–text models, with no training. That gives about 96% top-1 accuracy on validation.
3. **Use the gallery's structure.** Every dish has about the same number of photos, so no dish may swallow too many. Balancing the scores and assigning each photo to exactly one dish (at most 20 per dish) lifted validation R-precision from 0.954 to **0.982**.
4. Public leaderboard: **0.98159** [report]. You submitted **once**.

**One sentence to remember:** *"The text side is solved exactly, so retrieval becomes classification; zero-shot vision–language models beat anything I could train on 20 noisy photos per dish; and the equal-count structure of the gallery fixes the model's class bias."*

---

## 3. The pipeline, step by step

### Step 0: features, computed on Kaggle GPUs (`kernels/feats/feats.py`)
- **Image embeddings** for all 5,010 photos (train + validation + test) from 6 CLIP-family models plus DINOv2.
- **Four views per photo** (test-time augmentation, TTA):
  - `c`: resize, then centre crop
  - `s`: squash the whole image to a square
  - `cf` and `sf`: the same two, flipped left-right
- **Text embeddings** for 8 prompt templates per dish name and per alternative name, for example "a photo of {}.", "a close-up photo of {}.", "a plate of {}.", "food photography of {}.", plus the bare name. They also cover each definition, paraphrase and example query.
- Everything is saved as `.npy` files in `features/`, so the experiments in `src/` run in seconds on a laptop.

### Step 1: query → dish (`common.match_query`)
1. Normalise the text: lowercase, strip accents, and treat "&" as "and".
2. **Erase any dish definition sentence found inside the query.** Definitions can mention other dishes. For example, poutine's definition contains "fries", which would wrongly match the *fries* dish.
3. Among the remaining names and alternative names (singular/plural both allowed), take the **longest** match.
4. Result [re-run]: 505/505 train queries correct, 202/202 validation queries correct, 0 unmatched test queries, exactly 3 test queries per dish.

### Step 2: photo → dish scores (`models.load`, `text_protos`, `zs_logits`, `standardize`)
1. **Image vector:** L2-normalise each of the 4 view embeddings, average them, and normalise again.
2. **Dish prototype:** the normalised average of that dish's text embeddings. Prompts used: 8 templates × (name and each alternative name), the definition, 5 paraphrases and 5 example train queries. Validation and test query texts are **not** used. This is prompt ensembling.
3. **Score:** cosine similarity between the image vector and each dish prototype gives a 101-number row per photo.
4. **Standardise:** per model and per photo, subtract the mean and divide by the standard deviation across the 101 dishes. This puts the two models on a common scale.
5. **Add** the two models' standardised scores.
- Nothing here is trained. Validation top-1 accuracy [re-run]: SigLIP2-g 0.9644, PE-G 0.9554, both together 0.9644.

### Step 3: use the equal-count gallery (`models.sinkhorn`, `assign`, `rank_scores`; `make_final.py`)
- **The problem.** A zero-shot model has a class bias. Some dish names "attract" too many photos. On validation, where every dish has exactly 10 photos, the raw top-1 predictions gave between 8 and 12 photos per dish [re-run]. 22 dishes got too many and 21 too few.
- **Sinkhorn balancing:** repeatedly rescale the rows and columns of the photo × dish score matrix, so that every dish receives about the same total probability mass. This is a soft correction.
- **Hungarian assignment:**
  - Give each dish `cap` slots (`cap` = 10 on validation, 20 on test).
  - Let `scipy.optimize.linear_sum_assignment` find the cheapest way to put every photo in exactly one slot, with cost = −log-probability.
  - Result: every photo gets one dish, and no dish gets more than `cap` photos.
- **Final ranking for a dish:** photos assigned to it first (a +1000 bonus), then every other photo by its soft balanced score. Take the top 20.
- Because it uses the structure of the unlabeled test gallery, this step is **transductive**. It is allowed but must be declared (rule 10b), and your report declares it.

### Step 4: write the submission (`common.write_sub`)
Take the top 20 photo ids per query and write the CSV. `make_final.py` reproduces the whole thing from the cached features in under a minute. The notebook is a walkthrough of the same code.

---

## 4. Decision log

### D1. Validation that behaves exactly like the leaderboard
- **What:** `common.eval_val` scores your 202 validation queries against the 1,010-photo validation gallery with the same R-precision code the leaderboard uses. As a second check, the 2,000 train photos were used as a gallery with at most 20 per dish.
- **Why:** test labels are hidden, so you need a faithful stand-in. Two independent checks stop you from believing a gain of a few photos.
- **Evidence:** v1 gets 0.9822 on validation and 1,924 / 2,000 on train-as-gallery [re-run].
- **If asked:** "I never used test labels. I only trusted a change if it won on both validation and train-as-gallery."

### D2. Two baselines first (`src/baselines.py`)
- **b1, classical CV:** ORB keypoints → a 256-word bag of visual words (k-means) → plus an HSV colour histogram → logistic regression. Validation R-precision **0.0772** [log].
- **b2, transfer learning:** ImageNet ResNet-50 features → logistic regression. **0.4366** [log].
- **Why they matter:** they show the progression. Hand-crafted features can't separate 101 similar dishes (chance is about 0.01). ImageNet features help but need many labelled photos. Zero-shot vision–language models jump to 0.95+.
- **Syllabus:** b1 is Module 1; b2 is Module 2.

### D3. A rule-based query matcher, not a trained text model
- **Why:** each query names exactly one dish, and the metadata gives names and alternative names. Simple matching is exact and needs no training.
- **Evidence:** 505/505, 202/202, and 3 per dish on test [re-run].
- **Caveat:** it works because the queries are templated. A free-form query ("something crunchy with cheese") would break it.

### D4. Reduce retrieval to classification
- **Why:** relevance is defined per dish, so every query of a dish has the same relevant set. So you just need each gallery photo's dish.
- **Consequence:** the 3 queries for a dish get identical lists.
- **If asked:** it is why the pipeline is a classifier plus a ranker, and why detection or segmentation isn't needed.

### D5. Zero-shot vision–language models instead of training
- **What:** compare image embeddings with text prototypes and train nothing.
- **Why:**
  - With about 20 noisy labelled photos per dish you can't beat models trained on billions of web image–text pairs.
  - Evidence: ResNet-50 + logistic regression got 43% top-1. SigLIP2-g zero-shot got 96.4% [log].
- **Rejected few-shot heads** (prototypes from labelled photos, Tip-Adapter cache, logistic probe): validation 0.978–0.980 against 0.982 for v1 [log]. The noisy 20-shot labels add nothing beyond the text prototypes.
- **Important nuance:** "zero-shot" means the models were *not trained on your labels*. You still used validation labels to *choose* models and settings (see D6, §6).

### D6. SigLIP2-giant + PE-Core-bigG, not all six models
- **What:** six CLIP-family models were scored alone on validation (raw R-precision): SigLIP2-g 0.956, PE-G 0.944, SigLIP2-SO 0.938, SigLIP-SO 0.925, DFN-H 0.922, MetaCLIP2-H 0.911 [log]. Round 2 added three more, all weaker.
- **Why two:** the two best come from different labs and training data (Google WebLI vs Meta), so their errors partly differ. Adding weaker models diluted them: all six with equal weights gave 0.971 against 0.982 [log].
- **Honest caveat:** SigLIP2-g alone plus the balancing step already gets 0.9792 [log]. Adding PE-G buys about **3 photos out of 1,010**, which is within noise. See §6.

### D7. Prompt ensembling for the dish prototype
- **What:** average the text embeddings of many phrasings (8 templates × names, definition, paraphrases, example queries).
- **Why:** one phrasing is noisy. Averaging gives a steadier "centre" for the dish.
- **Rejected alternative, CuPL:** longer LLM-written visual descriptions. Validation 0.964–0.972, worse than plain templates [log].

### D8. Four-view test-time augmentation
- **What:** average the embeddings of centre-crop, squash and their flips.
- **Why:** food photos have different framing. The views make the embedding less sensitive to it.
- **Round 2 test:** adding zoomed-centre views gave the same validation score (0.9822) and +3 of 2,000 on train-as-gallery, which is noise. Not adopted.

### D9. Standardise each model's scores before adding
- **Why:** different models' cosine similarities sit on different scales. Without this, the model with the larger spread would dominate.

### D10. Balanced assignment (Sinkhorn + Hungarian): the biggest single gain
- **Evidence [re-run]:** raw ensemble 0.9535 → soft Sinkhorn alone (T = 0.3) 0.9812 → Sinkhorn + hard assignment **0.9822**. The assignment changed the dish for 31 of 1,010 validation photos: 22 became correct, 4 became wrong, and 5 changed from one wrong dish to another. On test it changes 38 of 2,000 photos.
- **What the ablation tells you [re-run]:**
  1. The big idea is *using the equal-count constraint*. Soft or hard, it is worth about **+2.8 points**. The hard step adds only about **+0.1** over soft Sinkhorn.
  2. **Hungarian alone, with no Sinkhorn, gives the identical assignment** (same 0.9822, same 18 errors). Neither the temperature nor the Sinkhorn step changed it, on validation or on test.
  3. **[inference]** Why Sinkhorn is redundant before the assignment: Sinkhorn only adds a constant per photo and a constant per dish to the score matrix. A per-photo constant can't change that photo's best dish. A per-dish constant adds the same total whoever is assigned, once every dish is filled to its cap. On validation 101 × 10 = 1,010, so every slot is filled. On test, 2,000 photos for 2,020 slots, it still changed nothing.
  4. So Sinkhorn's real job in your pipeline is the *soft balanced score* used to order the remaining photos. Hungarian is what makes the hard assignment.
- **If asked "what does Sinkhorn do?":** "It balances the score matrix so every dish gets about the same total probability mass, which removes the class-prior bias softly. The Hungarian step then enforces a hard capacity per dish."
- **Caveat on the temperature:** the soft version is sensitive to T (0.9683 at T = 0.02, 0.9812 at 0.3, 0.9743 at 1.0) [re-run]. The hard assignment isn't.

### D11. Capacity = ceil(N / 101)
- **What:** `cap` = 10 on validation (1,010 photos) and 20 on test (2,000 photos).
- **Why 20 and not exactly 19.8:** the test gallery has 19–20 photos per dish [report]. A cap of 20 leaves 20 spare slots (101 × 20 = 2,020), so a dish with a few extra photos isn't forced to give them up. The per-dish assigned count on test came out between 18 and 20 [re-run].
- **Caveat:** this depends on the competition really being balanced.

### D12. Ideas you tried and rejected (all logged)

| Idea | Val R-prec | Why it failed |
|---|---:|---|
| All six models, equal weights | 0.971 | Weak models dilute the strong ones |
| Weight search over six models | 0.982 | +4 of 3,010 photos: noise, not submitted |
| Few-shot heads from labelled photos | 0.978–0.980 | 20 noisy shots add nothing beyond text prototypes |
| kNN label propagation / centroid refinement | 0.959 / 0.972–0.980 | Raises raw accuracy but fights the balanced assignment |
| CuPL LLM descriptions | 0.964–0.972 | Plain templates are better |
| Qwen3-VL-8B re-scoring each photo's top 5 | 0.921–0.971 | Alone it gets only 86.4% top-1. Where the ensemble is wrong, the VLM is right only about 25% of the time. |
| Soft Sinkhorn instead of hard assignment | 0.981 | Hard is slightly better (+0.1) |
| Round 2: three more CLIP models, zoom views | 0.950–0.982 | Never better on both checks |

### D13. One submission only
- **Why:** no variant beat v1 by more than 4 of the 3,010 photos across validation and train-as-gallery. That is within noise, so nothing else was worth a leaderboard submission.
- **If asked:** "I used the leaderboard to confirm, not to search."

### D14. Declaring the transductive step
- The rules allow unsupervised use of the test gallery if declared. The balancing uses only the *count structure* of the unlabeled test photos. It uses no test labels, no hand labelling, and no outside image collection. Your REPORT §4.3 declares it.

---

## 5. Numbers to memorise

| Run | What | Val R-precision | Val top-1 acc |
|---|---|---:|---:|
| b1 | ORB BoVW + HSV + logistic regression | 0.0772 | 0.0822 |
| b2 | ImageNet ResNet-50 + logistic regression | 0.4366 | 0.4257 |
| zero-shot | SigLIP2-g alone | 0.9564 [re-run] | 0.9644 |
| zero-shot | PE-G alone | 0.9436 [re-run] | 0.9554 |
| zero-shot | SigLIP2-g + PE-G, no balancing | 0.9535 [re-run] | 0.9644 |
| + soft Sinkhorn (T = 0.3) | | 0.9812 [re-run] | |
| **v1** | **+ Sinkhorn + Hungarian** | **0.9822 [re-run]** | 0.9822 |

v1's other validation metrics: mAP@20 0.9891, P@5 0.9941, Hit@1 0.9901, MRR@20 0.9950 [re-run]; Hit@5 1.000 [report]. 18 of 1,010 validation photos are wrongly assigned [re-run]. Train-as-gallery: 1,924 / 2,000. Public leaderboard: **0.98159** [report].

What the 18 errors look like [re-run]: gyoza ↔ dumplings (4 of them); steak / prime rib / filet mignon (4); bread pudding ↔ french toast; scallops ↔ sushi; and a few chained errors among cannoli, ice cream, pancakes, cheesecake, creme brulee and panna cotta. The report says several are ambiguous photos, such as a scallop nigiri labelled *scallops*. Hard assignment can create **swap errors**: when one photo is forced out of its natural dish, another can be bumped.

---

## 6. Weak spots: know these before the examiner finds them

1. **You were not at the very top.** The public score 0.98159 sits just below a cluster at 0.986 (a gap of about 4 test photos, per your report), and the top entries are 1.000 [report]. I couldn't see your leaderboard rank. About half the test queries are public, so differences at this level are mostly noise. The report's own words: "the private ranking among those scores is largely noise."
2. **Ensembling two models is only weakly supported.** SigLIP2-g alone with balancing is 0.9792 against 0.9822 for the pair, about 3 photos of 1,010. Defend it as a reasonable choice (two different labs and training sets), not as a proven win.
3. **"Zero-shot" doesn't mean "no labels were used".** Validation labels chose the models, the prompt set, the capacity and the temperature. Train-as-gallery partly guards against overfitting to validation.
4. **Sinkhorn's role is smaller than the report's wording suggests.** See D10: the hard assignment is identical without Sinkhorn. If asked what lifted the score, say "using the equal-count structure", not "Sinkhorn specifically".
5. **The whole balancing trick relies on the gallery being balanced.** The "19–20 per dish" test count comes from the competition's description via your report [report]. If the hidden test were unbalanced, forcing balance would hurt. This is also why it is a *declared* transductive step.
6. **The query matcher relies on templated queries.** It scores 100% here, which says nothing about free-form language.
7. **Label noise is asserted, not measured.** The report says the training labels are noisy and gives examples (a taiyaki labelled *ice cream*). I haven't looked at the photos.
8. **The experiment log has quirks.**
   - The first v1 row shows Hit@1 = 1.0 and mAP 0.9886, but the report and my re-run give 0.9901 and 0.9891. The log row came from an older tie-ordering. R-precision, the leaderboard metric, is the same (0.9822).
   - Two rows are duplicates, and some rows are partly empty (HANDOFF §7).
9. **The notebook has no saved outputs** (HANDOFF §3). Run it once in Jupyter before you show it.
10. **Reproducing the features needs Kaggle GPUs** (about 3 hours). Reproducing the final submission from the cached features is a one-minute CPU job.
11. **Private leaderboard score is unknown** until the competition ends [report].

---

## 7. Concept cheat sheet (plain words)

| Term | Plain meaning | Where |
|---|---|---|
| Text-to-image retrieval | Given words, rank images | the task |
| R-precision | Fraction of the top-R results that are relevant, where R is the number of relevant items | metric |
| Embedding | A vector a pretrained network produces to summarise an image or sentence | D5 |
| Cosine similarity | Angle-based closeness of two vectors, between −1 and 1 | Step 2 |
| CLIP / SigLIP | Models trained so a photo and its caption land close together. SigLIP uses a sigmoid loss on each image–text pair instead of CLIP's softmax over the batch. | D5 |
| Zero-shot | Using a pretrained model on your task with no training on your labels | D5 |
| Prototype | One representative vector per class, here the averaged text embedding of its prompts | D7 |
| Prompt ensembling | Averaging many phrasings of the same prompt | D7 |
| TTA (test-time augmentation) | Embedding several altered versions of one photo and averaging | D8 |
| Class-prior bias | The model favours some classes regardless of the photo | Step 3 |
| Inductive vs transductive | Inductive: learn from training data, then judge each test item alone. Transductive: also use the structure of the unlabeled test set. | D10 |
| Sinkhorn | Alternately rescale rows and columns until the matrix has the wanted row and column sums. It is entropy-regularised optimal transport. | D10 |
| Hungarian algorithm | Finds the cheapest one-to-one assignment. Also used inside DETR's loss (Module 4 notes). | D10 |
| ORB / bag of visual words | Binary keypoint descriptors, clustered with k-means into "words" whose counts form the image vector | D2 |
| Transfer learning | Reusing ImageNet-trained features with a small classifier | D2 |
| Label propagation | Smoothing class predictions over a kNN graph of similar images | D12 |
| VLM (Qwen3-VL) | Generative vision–language model that answers questions about images | D12 |

**Syllabus mapping (from your report).** Module 1: ORB, bag of visual words, colour histograms (b1). Module 2: ImageNet transfer learning and why 20 noisy labels overfit (b2, few-shot heads). Module 3: embeddings, cosine similarity, retrieval, contrastive image–text models. Module 6: foundation models used zero-shot, and a generative VLM. Beyond the syllabus: Sinkhorn, Hungarian assignment, prompt ensembling.

---

## 8. Next step

The MCQ, multi-select and descriptive questions I expect, with model answers, are in `../notes/Kaggle-Competition-3_Dish-Rizz_Exam-Notes.md`.
