# Kaggle Competition 3 — "Match the Dish to Its Rizz": Exam Notes (expected questions)

**Course:** SST Computer Vision (Scaler) · **Covers:** Competition-3 in full, plus a shorter bank for Competition-2 ("Cooked or Not") in Part 6. Competition-1 has its own file: `Kaggle-Competition-1_Find-Homography_Exam-Notes.md`.

**Built from:**
- `Kaggle Competition-3/STUDY_GUIDE.md` and `Kaggle Competition-2/STUDY_GUIDE.md` (every number was re-run on your files; see those guides for the evidence)
- `classes/question-pattern.pdf` (the exam format, which I re-read for this)
- what your teacher said about the Kaggle work (below)

> **Preview tip (VS Code):** press `Ctrl+Shift+V` to render the tables.

---

## The Kaggle question — what you were actually asked

**One sentence:** you are given a sentence like *"Show me pictures of potstickers."* and a pile of
2,000 unlabelled food photos. For each sentence, you list the **20 photos** most likely to show that dish,
best first.

**What you were given** (counts re-checked on the CSV files)

| Item | Detail |
|---|---|
| Dishes | **101** (`category_metadata.csv`: name, one-line definition, alternative names, 5 paraphrases). Each query names exactly one dish. |
| Train | 2,000 labelled photos (19–20 per dish, some label noise) and 505 example queries (5 per dish) |
| Validation | 1,010 photos (exactly 10 per dish), 202 queries, and the answer key (`validation_ground_truth.csv`) |
| Test | 2,000 photos with **no labels** and **303 queries** (exactly 3 per dish) |
| Example | Query "A photograph of a croque madame." *Gyoza* has the alternative name *potstickers*, so "Show me pictures of potstickers" means gyoza. |

**What you submit.** A CSV `query_id, image_ids`: one row per test query, with **exactly 20 distinct
test-photo ids separated by spaces, best first** (`validate_submission.py` rejects anything else).
Limit: 5 submissions per day.

**How you are scored: R-precision.** For a query whose dish has R relevant photos, look at your top R
results. The score is the fraction that are relevant, averaged over queries. R is 10 on validation and 19–20
on test. Example: R = 20 and 19 of your top 20 are right → 0.95. A random ranking scores about
R/N = 20/2000 = **0.01**. Order *inside* the top R doesn't matter. The public leaderboard uses about
half of the test queries, and the rest are scored privately [report].

**Rules that shaped the design** [report: I can't see the Kaggle page]

| Allowed | Forbidden | Allowed if declared |
|---|---|---|
| Generic pretrained models and vision-language models | Checkpoints trained to classify these food categories; outside copies of the images; matching test photos against outside collections; hand-labelling | Unsupervised use of the test gallery (rule 10b) |

**Why it is hard:** 101 dishes, and many look alike (gyoza vs dumplings; steak vs filet mignon vs prime
rib; bread pudding vs french toast), and the 20 training photos per dish are noisy.
**Why it turned out easy once you saw it:** every query names its dish in plain text, and big pretrained
image–text models already know these dishes. So the work is (1) read the dish out of the query,
(2) score photos against dish names, (3) use the gallery's equal-count structure for the last few percent.

**What the question is really testing:** whether you can pick a sound approach for text-to-image retrieval
(classical vs transfer learning vs zero-shot), justify it with numbers, and follow the rules about what data
may be used.

---

## What sir said, and what it means

> "Will just test the approach you used for assignment and Descriptive and mcq"

How I read it:
1. The Kaggle work is examined through **questions about the approach you used**, not through a new Kaggle task.
2. The formats named are **MCQ** and **descriptive**. That maps to **Section A** (single MCQ, 1 mark), **Section B** (multi-select, 2 marks, all-correct-only) and **Section C** (descriptive, 2 marks, ≤ 100 words).
3. He did not mention coding (D) or numericals (F). A case study (E) is possible because the pipeline is a natural "problem → task → data → model → metric → failure mode" chain, but **that part is my guess, not something he said**. Prepare E only after A–C are solid.
4. Because the questions are about *your* approach, a generic textbook answer loses marks. Answer with your own pipeline, your own reason and one of your own numbers.

| Section | Format | What a Kaggle question looks like | Where to practise |
|---|---|---|---|
| **A** Single MCQ (10 × 1) | One correct option | "Which technique ranked the photos?", "What does the Hungarian step do?", "What R-precision did the ORB baseline get?" | Part 3 |
| **B** Multi-select (10 × 2) | All correct and none wrong. Watch for "always" and "never". | "Which statements about your final pipeline are true?" with traps like *trained*, *used test labels*, *used ORB* | Part 4 |
| **C** Descriptive (10 × 2, ≤ 100 words) | Define → one reason, distinction or example → stop | "Why zero-shot instead of training?", "Explain the balancing step", "What went wrong and how would you improve it?" | Part 5, Part 6 |
| **E** Case study (15), *possible* | Problem → task → data → model → metric → failure mode | "Adapt your approach to retrieving paintings by title" | Part 5 (C16), Part 7 |

---

# PART 1 — The nine question types to expect

| # | Type | Typical wording | Shape of a full-marks answer |
|---|---|---|---|
| 1 ★ | **Describe your approach end to end** | "Explain the approach you used for the Kaggle assignment." | Every stage in order, one reason each, one number, one limitation, in ≤ 100 words. C1 is the model answer. |
| 2 ★ | **Name the approach** | "Which technique was used to …?" | Pick the exact method. Distractors are things you tried and rejected (ORB, ResNet probe, Qwen, label propagation). |
| 3 ★ | **Why this, not that** | "Why zero-shot instead of fine-tuning?" | Claim → one number as evidence → why the alternative was weaker. |
| 4 ★ | **True / false about your pipeline** | "Which statements are true?" | Judge each option alone. Traps: "trained", "test labels", "all six models", "ORB in the final". |
| 5 ★ | **Concept behind a component** | "What does Sinkhorn / Hungarian / TTA / a prototype do?" | Define precisely, then say what it did in your pipeline. |
| 6 | **Read the result** | "What does ORB 0.077 vs ResNet 0.437 vs zero-shot 0.956 show?" | State the trend, then the reason (representation quality, labels needed). |
| 7 ★ | **Limitations and improvements** | "Where does it fail? What next?" | Name the look-alike dishes and label noise, then 2 concrete fixes. Admit gains would be within noise. |
| 8 | **Rules and validity** | "Is using the test gallery allowed?" | Allowed if unsupervised and declared. No test labels, no outside images, no hand labelling. |
| 9 | **Transfer to a new problem** | "A museum wants to retrieve paintings by title…" | Reuse the chain: text → class, pretrained embeddings, prototypes, balance if counts are known, validate with the same metric. |

**A descriptive answer in 100 words (from the exam PDF):** (1) define accurately, (2) add one distinction, reason or example, (3) stop. No long introduction. Don't list terms without explaining them.

---

# PART 2 — Competition-3 fact sheet (30-second recall)

| Item | Fact |
|---|---|
| Task | Text query ("Show me pictures of potstickers.") → rank 2,000 test photos; list the 20 best |
| Dishes | 101. Each query names exactly one dish. Test: 303 queries, 3 per dish. |
| Data | Train 2,000 photos (19–20 per dish, noisy labels). Validation 1,010 photos (10 per dish), 202 queries. Test 2,000 photos. |
| Metric | **R-precision**: fraction of the top R that are relevant, R = # relevant photos (10 on validation, 19–20 on test). Random ≈ 0.01. |
| Rules | Generic pretrained models allowed. Food-trained checkpoints, outside images, hand labelling forbidden. Unsupervised use of the test gallery allowed **if declared** (rule 10b). 5 submissions/day. |
| Final model | **Zero-shot** SigLIP2-giant + PE-Core-bigG. **Nothing trained.** Then Sinkhorn + Hungarian on the gallery. |
| Query → dish | Rule-based text matcher. Erases the dish's own definition first. 505/505 and 202/202 correct. |
| Photo → dish | Mean of 4 views (centre crop, squash, both flipped) → cosine similarity to a per-dish text prototype → standardise per photo → add the two models |
| Prototype | Mean text embedding of 8 templates × names/alt names, definition, 5 paraphrases, 5 example train queries |
| Balancing | Equal photos per dish → Sinkhorn (soft) + Hungarian with capacity 10 (val) / 20 (test) |
| Numbers | b1 ORB+HSV 0.077 · b2 ResNet-50 0.437 · SigLIP2-g 0.956 · pair raw 0.9535 · soft Sinkhorn 0.9812 · **v1 0.9822** · public LB **0.98159** · train-as-gallery 1,924/2,000 |
| Rejected | All six models, weight search, few-shot heads, label propagation, CuPL, Qwen3-VL-8B, more CLIPs, zoom views |
| Submissions | **One** (v1). Alternatives were within noise (≤ 4 of 3,010 photos). |

---

# PART 3 — Section A: single-correct MCQs

**A1 ★.** What does R-precision measure in this task?
- A. The fraction of queries whose first result is correct
- B. The fraction of relevant photos found in the top R results, where R is the number of relevant photos for that query
- C. The average position of the first relevant photo
- D. The fraction of all gallery photos labelled correctly

**Answer: B.** A is Hit@1. C resembles MRR (mean reciprocal rank). D is plain accuracy.

**A2 ★.** Why could text-to-image retrieval be solved as image classification here?
- A. The test photos came with labels
- B. Each query names exactly one dish and relevance is defined per dish, so all queries of a dish share one ranking
- C. The metric is accuracy
- D. Queries were given as images

**Answer: B.** A is false (test is unlabelled). C is false (the metric is R-precision). D is false (queries are sentences).

**A3 ★.** How many labelled food photos were used to fit model parameters in the final v1 pipeline?
- A. 2,000
- B. 1,010
- C. 3,010
- D. None

**Answer: D.** v1 is zero-shot. Validation labels were used only to *choose* models and settings, not to fit parameters.

**A4 ★.** What does the Hungarian algorithm do in the pipeline?
- A. It extracts image features
- B. It finds the minimum-cost one-to-one assignment of photos to dish slots, so no dish gets more than its capacity
- C. It clusters photos with k-means
- D. It generates prompts

**Answer: B.** Each dish gets `cap` slots (10 on validation, 20 on test) and `linear_sum_assignment` fills them at least cost.

**A5.** Why is a dish's own definition text erased from a query before name matching?
- A. To shorten the query
- B. Because a definition can mention other dishes (poutine's mentions "fries") and cause a wrong match
- C. Because definitions are not English
- D. To speed up the code

**Answer: B.**

**A6 ★.** Roughly what validation R-precision did the classical ORB bag-of-words + HSV baseline (b1) reach?
- A. 0.08
- B. 0.44
- C. 0.96
- D. 0.98

**Answer: A (0.0772).** B is the ResNet-50 baseline (0.4366). C is a single zero-shot model. D is v1.

**A7.** What is test-time augmentation (TTA) as used here?
- A. Training on extra augmented photos
- B. Averaging the embeddings of several views of the same photo (centre crop, squash, and their flips)
- C. Rotating the text prompts
- D. Averaging model checkpoints

**Answer: B.**

**A8 ★.** What is the "text prototype" of a dish?
- A. The embedding of one prompt, "a photo of {dish}"
- B. The normalised average of the text embeddings of many prompts for that dish
- C. The average image embedding of its training photos
- D. A learned linear layer

**Answer: B.** C would be a few-shot (image) prototype, which was tried and rejected.

**A9.** Why are each model's scores standardised per photo before the two models are added?
- A. Because cosine scores of different models have different scales and spreads
- B. To make the scores positive
- C. To remove duplicate photos
- D. To reduce the number of dishes

**Answer: A.**

**A10 ★.** Why did few-shot heads built from the labelled photos not beat the zero-shot text prototypes?
- A. The photos were too large
- B. About 20 noisy shots per dish add nothing beyond rich text prototypes
- C. Few-shot heads cannot use embeddings
- D. They need test labels

**Answer: B.** They scored 0.978–0.980 against 0.982.

**A11 ★.** What property of the test gallery does the transductive step exploit?
- A. Every dish has about the same number of photos
- B. Photos are sorted by dish
- C. The test labels are in the file names
- D. Photos are all taken in daylight

**Answer: A.** It must be declared under rule 10b.

**A12.** "Transductive" inference means:
- A. Training a new model for each test photo
- B. Using the structure of the unlabeled test set when predicting, not only a model learned from training data
- C. Using test labels
- D. Predicting each test item completely independently

**Answer: B.** D describes inductive inference.

**A13 ★.** Which kind of model was clearly weaker than the CLIP-family models at fine-grained food recognition?
- A. SigLIP2-giant
- B. PE-Core-bigG
- C. The generative VLM Qwen3-VL-8B (86.4% top-1 on its candidates)
- D. DFN-H

**Answer: C.**

**A14.** What is the main source of the remaining validation errors?
- A. Corrupted images
- B. Look-alike dishes (gyoza vs dumplings; steak vs prime rib vs filet mignon; custard desserts)
- C. Wrong query matching
- D. Too few gallery photos

**Answer: B.** The query matcher is 100% correct on all labelled queries.

**A15.** About what R-precision would a random ranking get on the test gallery (R = 20, N = 2,000)?
- A. 0.01
- B. 0.10
- C. 0.50
- D. 0.20

**Answer: A.** R/N = 20/2000 = 0.01 (about 1/101).

**A16.** Why was only one submission made?
- A. The leaderboard was closed
- B. Every alternative differed from v1 by at most 4 of 3,010 photos, which is within noise
- C. The other models crashed
- D. The rules allow only one

**Answer: B.** The limit is 5 per day, so D is false.

**A17.** How does SigLIP's training loss differ from CLIP's?
- A. SigLIP uses a sigmoid loss on each image–text pair; CLIP uses a softmax contrastive loss over the batch
- B. SigLIP is trained only on images
- C. SigLIP needs class labels
- D. SigLIP is generative

**Answer: A.**

**A18.** Which statement about Sinkhorn balancing is correct?
- A. It produces a hard one-photo-one-dish assignment
- B. It repeatedly rescales rows and columns so each dish receives about the same total probability mass
- C. It trains the image encoder
- D. It removes duplicate photos

**Answer: B.** A describes the Hungarian step.

---

# PART 4 — Section B: multi-select (all correct, none wrong)

**B1 ★.** Which statements about the final v1 pipeline are true?
- A. It fits no model parameters on labelled food photos
- B. It uses ORB bag-of-visual-words features
- C. It maps queries to dishes with a rule-based text matcher
- D. It uses test labels to tune the assignment
- E. It exploits the equal number of photos per dish

**Answer: A, C, E.** B: ORB was only baseline b1. D: no test labels exist or were used.

**B2.** Which statements about Sinkhorn balancing as used are true?
- A. It rescales rows and columns of the score matrix so each dish gets about equal total mass
- B. Its output is a hard one-photo-one-dish assignment
- C. It uses the labels of the test photos
- D. It can be viewed as entropy-regularised optimal transport

**Answer: A, D.** B is the Hungarian step.

**B3 ★.** Which of these were tried and rejected?
- A. Few-shot heads from labelled photos (prototype, Tip-Adapter, logistic probe)
- B. kNN label propagation over the gallery
- C. LLM-written CuPL descriptions
- D. The two-model SigLIP2-g + PE-G zero-shot ensemble
- E. Qwen3-VL-8B re-scoring of each photo's top five dishes

**Answer: A, B, C, E.** D is the final model.

**B4 ★.** Which are valid reasons zero-shot vision–language models beat the ResNet-50 + logistic regression baseline?
- A. They were trained on web-scale image–text pairs that include many food photos and dish names
- B. A text prototype per dish needs no labelled photos
- C. ResNet-50 features cannot be L2-normalised
- D. With about 20 noisy labelled photos per dish, a classifier on generic ImageNet features separates 101 similar dishes poorly

**Answer: A, B, D.**

**B5.** Which statements about R-precision are true?
- A. R can differ between queries (10 on validation, 19–20 on test)
- B. It equals precision at R
- C. It penalises the order of photos *inside* the top R
- D. It equals recall at R

**Answer: A, B, D.** C is false. Order inside the top R doesn't matter. Precision@R equals recall@R because there are exactly R relevant items.

**B6.** Which statements describe the Hungarian assignment step here?
- A. Capacity per dish is ceil(N/101)
- B. Cost is the negative log-probability of photo → dish
- C. A photo may be given to several dishes
- D. Each dish is repeated `cap` times as columns

**Answer: A, B, D.** C is false. Each photo goes to exactly one slot.

**B7 ★.** Which statements about the remaining errors are true?
- A. Most involve visually similar dishes
- B. Some photos are ambiguous or arguably mislabelled
- C. Hard assignment can create swap errors, where forcing one photo out of its dish bumps another
- D. They are mostly caused by wrong query matching

**Answer: A, B, C.** D is false (505/505 and 202/202 queries matched).

**B8 ★.** Which are consistent with the competition rules as described in your report?
- A. Using generic pretrained models and VLMs
- B. Using a checkpoint trained to classify these food categories
- C. Unsupervised use of the test gallery, if declared
- D. Matching test photos against an outside image collection
- E. Hand-labelling test photos

**Answer: A, C.** B, D, E are forbidden.

**B9.** Which statements about the validation setup are true?
- A. The validation gallery and queries are scored exactly like the leaderboard
- B. Test labels were used to tune settings
- C. A second check used the 2,000 train photos as a gallery (cap 20)
- D. Validation was used to choose models and settings

**Answer: A, C, D.** B is false.

**B10.** Which texts go into a dish's prototype?
- A. 8 templates × the dish name and each alternative name
- B. The dish definition
- C. The 5 paraphrase sentences
- D. The example train queries
- E. The texts of validation and test queries

**Answer: A, B, C, D.** E is not used (excluded in code).

---

# PART 5 — Section C: descriptive questions with model answers (≤ 100 words)

**C1 ★. Explain how your final pipeline turns a text query into a ranking of the 2,000 test photos.**

> **Answer:** Each query is matched to one of 101 dishes by rule-based text matching, so all queries of a dish share one ranking. Two pretrained vision–language models, SigLIP2-giant and PE-Core-bigG, embed every photo, averaged over four views. Each photo is scored by cosine similarity to a per-dish text prototype, with no training. Because every dish has about the same number of photos, I balance the scores with Sinkhorn and assign photos to dishes with a capacity of 20 using the Hungarian algorithm. A dish's list is its assigned photos first, then the rest by score; the top 20 are submitted.

**C2 ★. Why zero-shot models instead of training a classifier on the labelled photos?**

> **Answer:** With about 20 noisy labelled photos per dish, a trained classifier learns from a small, partly wrong set. Web-scale image–text models like SigLIP2 already know these dishes. Comparing a photo with text prototypes gave 96.4% top-1 on validation with no training, while an ImageNet ResNet-50 with logistic regression reached 43%. Adding few-shot heads from the labelled photos scored 0.978–0.980 against 0.982, so they added nothing. Zero-shot also avoids overfitting the label noise.

**C3 ★. What does the balancing step (Sinkhorn + Hungarian) do and why did it help?**

> **Answer:** Zero-shot scores have a class bias: some dish names attract too many photos. On validation, raw top-1 predictions gave between 8 and 12 photos per dish instead of 10. Every dish has the same number of gallery photos, so I balance the score matrix with Sinkhorn and then assign each photo to exactly one dish with a per-dish capacity, using the Hungarian algorithm. Validation R-precision rose from 0.954 to 0.982. It uses no test labels, only the count structure of the unlabeled gallery, which I declared.

**C4. Why did the classical ORB bag-of-words baseline do so badly (R-precision 0.077)?**

> **Answer:** ORB keypoints describe small local patches, and a bag of 256 visual words discards where they are; HSV histograms keep only colour. Many of the 101 dishes share colours and textures, such as gyoza and dumplings or different steak cuts, and food has no rigid shape or reliable corners. With about 20 photos per dish the logistic regression has little to learn from. Chance is about 0.01, so 0.077 is only slightly better. Classical features suit geometry, not fine-grained semantic categories.

**C5. What is test-time augmentation and which views did you use?**

> **Answer:** Test-time augmentation embeds several altered versions of the same photo and averages the vectors, so the result is less sensitive to framing. I used four views: a centre crop, the whole image squashed to a square, and the left-right flips of both. Each view's embedding is L2-normalised, averaged, and normalised again. It costs four forward passes per photo and no training. Adding zoomed-centre views gave the same validation score, so extra views gave no further gain.

**C6 ★. What does "retrieval became classification" mean?**

> **Answer:** Every query names exactly one dish, and relevance is defined per dish. So all queries for a dish have the same set of relevant photos and receive the same ranking. The problem reduces to deciding which of the 101 dishes each gallery photo shows, which is 101-class classification, followed by listing a dish's photos. That is why a query matcher plus a photo classifier solves it, and why detection or segmentation was not needed.

**C7. Why is a rule-based query matcher acceptable, and what is its trick?**

> **Answer:** The queries are templated and always name the dish, sometimes by an alternative name, so exact text matching is enough and needs no training. A trick avoids false matches: a dish's own definition can mention other dishes (poutine's mentions "fries"), so definition text is erased from the query first, then the longest name match wins. It matched 505/505 train and 202/202 validation queries and gave exactly 3 test queries per dish. It would fail on free-form language.

**C8. Why combine two models and not all six?**

> **Answer:** I scored six CLIP-family models alone on validation. SigLIP2-giant (0.956) and PE-Core-bigG (0.944) were clearly best and come from different labs and data, so their errors partly differ. Averaging all six with equal weights lowered the score to 0.971 because weaker models dilute the strong ones. Searching weights gained only 4 photos of 3,010, which is noise. So I kept the simple two-model ensemble, though its advantage over SigLIP2-giant alone is small, about 3 photos of 1,010.

**C9. Why did re-scoring with Qwen3-VL-8B fail?**

> **Answer:** I asked Qwen3-VL-8B to choose among each photo's top five dishes. Alone it reached only 86.4% top-1, against 96.4% for the CLIP-style ensemble. Where the ensemble was wrong, Qwen was right only about 25% of the time, so fusing it added more errors than it fixed: validation 0.921–0.971, below 0.982. A generative VLM is much weaker than contrastively trained image–text embeddings at fine-grained food recognition, so it was rejected.

**C10 ★. What is transductive inference? Is your use of it allowed, and what is the risk?**

> **Answer:** Transductive inference uses the unlabeled test set itself, not only a model learned from training data. I used the fact that every test dish has about the same number of photos to balance scores and cap each dish at 20 photos. It uses no test labels, hand labelling or outside images. The rules allow unsupervised use of the test gallery if declared, and my report declares it. The risk is the assumption of a balanced gallery: if it were not, forcing balance would hurt.

**C11 ★. Where does your solution still fail, and how would you improve it?**

> **Answer:** The 18 remaining validation errors are mostly look-alike dishes (gyoza/dumplings, steak/prime rib/filet mignon, custard desserts, bread pudding/french toast), and some photos are ambiguous or mislabelled. Hard assignment can also cause swap errors. I would try light fine-tuning (LoRA) of SigLIP2-giant with weights interpolated back toward zero-shot, a much larger VLM applied only to low-margin photos, and better prompts for the known confusion pairs. Expected gains are a few photos, within leaderboard noise.

**C12 ★. Compare the three approaches you tried: classical, transfer learning and zero-shot.**

> **Answer:** Classical (ORB bag of words plus colour histogram) uses hand-crafted local features and scored 0.077, because dishes share colours and textures. Transfer learning (ImageNet ResNet-50 plus logistic regression) learns generic features but needs labelled photos per dish; with about 20 noisy ones it reached 0.437. Zero-shot SigLIP2/PE embeddings place photos and dish names in one shared space learned from web-scale pairs and need no labelled photos: 0.956. Each step swaps a hand-designed or label-hungry representation for a richer pretrained one.

**C13 ★. Explain zero-shot classification with a vision–language model and the prototype idea.**

> **Answer:** A vision–language model maps photos and sentences into one vector space where matching pairs are close. For each dish I embed many prompts (templates like "a photo of {}.", alternative names, the definition, paraphrases) and average them into a prototype. A photo's score for a dish is the cosine similarity between its embedding and that prototype, and the highest-scoring dish is the prediction. No parameters are trained on my labels, which is why it works with only 20 noisy photos per dish.

**C14. How did you decide what to submit?**

> **Answer:** I made one leaderboard submission. Every later variant (more models, zoom views, weight search, VLM fusion) differed from v1 by at most 4 photos out of 3,010 across validation and train-as-gallery, which is within noise. A variant had to win clearly on both checks. The public leaderboard covers only about half the test queries, so chasing tiny differences would overfit it. I used validation plus train-as-gallery for decisions and the leaderboard only to confirm.

**C15. What does the Hungarian algorithm do in your pipeline?**

> **Answer:** The Hungarian algorithm finds the minimum-cost one-to-one assignment. Here each dish gets 20 slots and each photo must take one slot; the cost is the negative log-probability that the photo shows that dish. The result assigns every photo to exactly one dish with at most 20 photos per dish. This corrects the model's class bias, because no dish can take too many photos. DETR uses the same algorithm to match predicted boxes to ground-truth objects.

**C16 ★. (Transfer) A museum wants to retrieve paintings by title. How would you adapt your approach?**

> **Answer:** I would map each query to a title with text matching, embed paintings and titles with a pretrained vision–language model, and score by cosine similarity to per-title prototypes built from several prompts, with no training. If each title has a known number of images, I would balance scores and assign with per-title capacities. I would validate on a held-out gallery with the same metric, and watch for look-alike titles, label noise and the risk that the balance assumption is wrong.

**C17 ★. Define R-precision and say why it suits this task.**

> **Answer:** For a query with R relevant photos, R-precision is the fraction of the top R results that are relevant, averaged over queries. R is 10 on validation and 19–20 on test. It equals precision and recall at R, ignores order inside the top R, and a random ranking scores about R/N = 0.01. Unlike precision@5, it adapts to the number of relevant items, so it is fair here, where each query has a fixed set of relevant photos.

**C18. Why standardise each model's scores before adding them?**

> **Answer:** Cosine similarities from different models have different ranges and spreads. Before adding SigLIP2 and PE scores, I subtract each photo's mean score over the 101 dishes and divide by its standard deviation, separately per model. Then neither model dominates just because its numbers are larger, and the sum behaves like an equal-weight vote. Without this, the model with the larger spread would decide most rankings.

---

# PART 6 — Competition-2 ("Cooked or Not", crash detection): quick bank

**Fact sheet.**
- Predict whether a crash happens in a 30-frame dashcam clip. Metric: ROC-AUC.
- Train: 3,000 clips in 1,500 pairs. Each pair is one crash clip and one pre-crash clip from the same video. Test comes from a different collection, about 1/3 crashes, with normal-driving negatives.
- **Official final = v5**: rank blend of 0.35 within-clip anomaly LightGBM + 0.25 classical-CV LightGBM + 0.40 CLIP ViT-L/14 zero-shot. Public LB **0.74272**, rank 130/211.
- Key numbers: classical CV 0.792 / LB 0.727; Swin3D + CLIP embeddings CV **0.992** but LB **0.614**; adversarial AUC 1.000.
- Late, unranked: Qwen3-VL-4B zero-shot 0.829; blend 0.847.

**M1 ★ (single).** Why was GroupKFold on `group_id` used?
- A. To balance the classes
- B. Because the two clips of a pair overlap in time, so they must not be split between train and validation
- C. To speed up training
- D. To remove the frame-rate artifact

**Answer: B.**

**M2 ★ (single).** An adversarial-validation AUC close to 1.0 means:
- A. The features are very predictive of crashes
- B. Train and test are easy to tell apart, so there is a strong distribution shift
- C. The model is well calibrated
- D. The labels are noisy

**Answer: B.**

**M3 ★ (multi-select).** Which are true for Competition-2?
- A. Swin3D + CLIP embeddings had CV 0.992 but LB 0.614
- B. Zero-shot CLIP was trained on the crash labels
- C. Test negatives were ordinary driving, not pre-crash footage
- D. Rank averaging was used because AUC depends only on ordering and component scores had different scales
- E. The public leaderboard is computed on all test clips

**Answer: A, C, D.** B is false (zero-shot means no training on the labels). E is false (30% of test).

**M4 ★ (descriptive).** Your CV was 0.99 and your leaderboard score 0.61. Why?

> **Answer:** Pretrained deep features separated crash from pre-crash clips almost perfectly inside the training collection (CV 0.99), probably using cues specific to that collection. Test came from a different collection: a classifier separated train from test perfectly with those features (adversarial AUC 1.0), and the model flagged only about 7–12% of test clips as crashes when about a third were expected. The leaderboard AUC was 0.614. Validation measured similarity to the training clips, which test clips lacked.

**M5 (descriptive).** What is adversarial validation and how did you use it?

> **Answer:** Adversarial validation trains a classifier to tell training clips from test clips using the features. An AUC near 0.5 means no shift; near 1.0 means the features mostly encode which collection a clip comes from. I used it to reject feature sets (deep embeddings scored 1.0), to design better ones (classical 0.905, then the within-clip features 0.762) and to drop individual features that shifted most. It needs no test labels, only the unlabeled test clips, which the rules allow.

**M6 (descriptive).** Why "within-clip relative" features?

> **Answer:** A crash is an abrupt change compared with that clip's own normal motion. For each time series I use only comparisons against the clip itself: the peak versus the median in units of MAD, the biggest jump versus the typical jump, and the level shift between halves. These are independent of camera, speed and scale, and avoid absolute levels that differ in test. I also averaged adjacent frame pairs and used even frames only, to cancel the frame-rate alternation seen in 82% of test clips.

**M7 (descriptive).** Why did zero-shot models transfer better than trained ones?

> **Answer:** A zero-shot model never sees the training labels, so it cannot learn shortcuts specific to the training collection. CLIP ViT-L/14 scored 0.638 on train but 0.675 on the leaderboard; Qwen3-VL-4B scored 0.748 on train and 0.829 on the leaderboard (late). A plausible reason for the higher test scores is that test negatives (ordinary driving) are easier than the train negatives (just before a crash). Their errors are also nearly unrelated to the motion models', which is why blending helped.

**Competition-1 ("Find Homography So Dope").** A different task (estimating a homography between image pairs). Its question bank is in `Kaggle-Competition-1_Find-Homography_Exam-Notes.md`, with the deeper walk-through in `Kaggle Competition-1/STUDY_GUIDE.md`. **Read the authorship line in section 0 of those notes first**: it records which parts of the final pipeline are yours and which match a peer's reference notebook, and how to describe that truthfully.

---

# PART 7 — Themes that cross both competitions (for generic questions)

| Theme | Competition-2 (crash) | Competition-3 (dishes) |
|---|---|---|
| What the pretrained model brought | CLIP / Qwen zero-shot signal that doesn't learn collection shortcuts | SigLIP2 / PE zero-shot: 96.4% top-1 with no training |
| Validation honesty | Grouped CV plus adversarial validation, because plain CV lied (0.99 vs 0.61) | Validation scored exactly like the leaderboard, plus train-as-gallery as a second check |
| Using the unlabeled test set (allowed) | Adversarial validation, feature filtering, frame-rate analysis | Sinkhorn + Hungarian balancing (declared) |
| Biggest lesson | CV only helps if validation resembles test | Using the structure of the problem (equal counts, per-dish relevance) beat adding more models |
| Where it was weak | Final 0.743 was below the leaderboard median (0.78) | Gains between variants were within leaderboard noise |
| Syllabus modules | 1 (ORB, RANSAC, flow), 2, 3 | 1 (ORB, BoVW), 2, 3, 6 |

**One-line answers for generic questions:**
- *Why not just train a big model on the labelled data?* In both tasks the labelled data (3,000 clips; 20 noisy photos per dish) was small, or from a different distribution than the test.
- *Why report more than one metric?* R-precision is the leaderboard metric; mAP@20, P@5, Hit@k and MRR describe the ranking from other angles. For Competition-2 only ROC-AUC counted.
- *What if validation and test differ?* Check with label-free diagnostics (predicted positive rate, adversarial AUC) before trusting validation.

---

# PART 8 — Traps and wording to avoid

| Trap | The correct statement |
|---|---|
| "My final model was trained / fine-tuned." | It was **zero-shot**. Nothing was trained on food labels. |
| "Zero-shot means I used no labels anywhere." | No labels fit the model, but validation labels chose models and settings. |
| "Sinkhorn gives the final assignment." | The **Hungarian** step gives the hard assignment. Sinkhorn gives soft balanced scores. |
| "Sinkhorn is what improved the score." | Safer: *using the equal-count constraint* (soft or hard) improved it. Hungarian alone gave the identical assignment. |
| "I used ORB / ResNet in the final." | They were baselines b1 and b2 only. |
| "I used all six CLIP models." | Final = SigLIP2-giant + PE-Core-bigG. |
| "The accuracy on test is 98.2%." | 0.9822 is validation **R-precision**. Public leaderboard is 0.98159. |
| "R-precision is precision@5." | Precision at R, where R = number of relevant photos. |
| "I tuned on the leaderboard." | One submission only; decisions came from validation and train-as-gallery. |
| "I was first / near the top." | 0.98159 sits just below a 0.986 cluster; top entries are 1.000 [report]. Check your rank on Kaggle before claiming anything. |
| "All errors are look-alike dishes." | *Most* are. Some are ambiguous photos and chained swap errors from hard assignment. |
| "Hungarian is only used here." | It is also the matching step inside DETR (Module 4 notes, Part 4.5). |

---

# PART 9 — Last-night checklist

1. Say the one-sentence story: *text side solved → retrieval is classification → zero-shot SigLIP2 + PE → equal-count balancing → 0.982.*
2. Name the **rejected** list (few-shot, label propagation, CuPL, Qwen, six models, weight search, zoom views) and the one-line reason: *noise-level or worse.*
3. Know the numbers: **0.077, 0.437, 0.956, 0.9535, 0.9812, 0.9822, 0.98159, 1,924/2,000, 505/505, 202/202.**
4. Say what R-precision is and why random ≈ 0.01.
5. Explain Sinkhorn (soft, equal mass) vs Hungarian (hard, capacity) in one sentence each.
6. Say why the transductive step is allowed and what could go wrong.
7. Know that the final model was **not trained**, and that validation labels still chose the settings.
8. Admit the weak spots: gains within noise, two-model ensemble marginal, look-alike dishes.
9. Competition-2 in one line: *CV 0.99 vs LB 0.61 → domain shift → adversarial validation → within-clip features + zero-shot → 0.743.*
10. Practise Parts 3–5 aloud, with the answers covered.
