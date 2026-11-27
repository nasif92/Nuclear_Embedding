# Set Transformer vs MLP for nucleus classification: what we did, how the models learn, what we found

Sections 1-12 are Ki67 only; sections 13-14 extend to ER and PR and to Nuc2Vec pretraining. Two classes (Tumor / Non Tumor). Training slides: 28 Ki67 slides. Reference (test) slides: 6 held-out slides (CIS24-000938, FHS24-019049, MIS24-007597, PLS24-009859, RDS24-012921, RVS24-009627). Experiments run 2026-11-23 to 2026-11-26.

---

## 1. The question

The baseline classifies each nucleus **alone**: one row of QuPath features in, one class out (an MLP). The idea taken from the CoNIC notebook was a **Set Transformer**: give the model a *group* of nuclei at once so that each nucleus can "look at" the others before deciding. A pathologist does something like this: a nucleus is judged partly by what surrounds it.

Three things could make a set model better than the MLP, and they are different claims:

1. **Neighbourhood context**: nearby nuclei carry information (tumour nuclei sit among tumour nuclei).
2. **Slide calibration**: comparing a nucleus with other nuclei *from the same slide* cancels out slide-wide shifts (stain intensity, scanner colour, staining batch).
3. **A shortcut**: something about how the groups are built tells the model the answer without it learning anything useful for deployment.

Most of the work below was about telling these apart.

---

## 2. Data: where the labels and features come from

**Detections and features.** QuPath detected every nucleus in each slide and exported a detection GeoJSON (`*-feat-no_cells.geojson.gz`) with about 108 measurements per nucleus (`all_derived` feature set): shape, size, haematoxylin/DAB optical densities, texture, and QuPath's own neighbourhood smoothing features.

**Labels.** A pathologist drew annotation polygons (Tumor / Non Tumor regions). A nucleus gets a label if its **centroid falls inside** an annotation polygon (point-in-polygon). Nuclei outside every polygon have no label. This means:

- labels are **region labels**, not per-nucleus labels: everything inside a tumour polygon is called Tumor;
- annotated regions of different classes rarely touch. In CV about 22% of held-out nuclei were within 20 µm of an annotated nucleus of the other class (assuming 0.25 µm/px), and in one fold only 0.1%.

**Joining features to labels and embeddings.** Each nucleus is identified by `subtype|wsi_name|det_index`, where `det_index` is its position in the original detection file. Features come from that row; LEMON embeddings are looked up with the same key.

**Two test sets on the reference slides.**

| test set | what it is | caveat |
|---|---|---|
| Annotated reference | nuclei inside the pathologist's polygons on the 6 reference slides | region labels; the main test |
| Labelled-binary | ~3k nuclei per slide, labelled by an existing classifier's own calls | not full slides, not pathologist labels; a different class mix |

**Feature scaling.** Every feature is z-scored with the **training** mean and standard deviation, then clipped to ±20. Missing values are filled with the training median. The reference slides never contribute to any statistic.

**Outlier slides.** FHS24-019049 and PLS24-009859 behave very differently from the rest, so every table reports results with and without them.

---

## 3. The baseline MLP (`train.py`)

### Architecture

```
108 features → [Linear → BatchNorm → ReLU → Dropout 0.5] × 5 (96, 48, 24, 12, 6 units) → Linear → 2 logits
```

Each nucleus is processed independently. Nothing about its neighbours enters the model except what QuPath already put in the features.

### How it learns

- **Loss:** cross-entropy between the 2 logits and the true class. For one nucleus with predicted probability *p* for its true class, loss = −log *p*. Confident and right → near 0; confident and wrong → large.
- **Class imbalance:** handled by **weighted sampling**. Each training nucleus is drawn with probability ∝ 1 / (count of its class)^0.5, so the rarer class is seen more often, but not fully balanced (power 0.5 is a compromise between "as in the data" (0) and "fully balanced" (1)).
- **Optimiser:** Adam, learning rate 1e-4, weight decay 1e-3, batch 10,000 nuclei, 200 epochs, cosine learning-rate decay to zero, gradient clipping at norm 5.
- **Model selection:** none. The **final epoch** is kept, so the reference slides never influence training. This is the rule for every model here.

---

## 4. The Set Transformer

### 4.1 Core idea: attention over a set

A **set** is a group of nuclei with no order. Each nucleus is first turned into a 64-number vector (its *embedding*). Then **self-attention** lets every nucleus update its vector using a weighted mix of all the others' vectors:

1. Each nucleus produces a *query* ("what am I looking for?"), a *key* ("what do I contain?") and a *value* ("what do I pass on?"), each a linear projection of its embedding.
2. Nucleus *i* scores every nucleus *j* by how well *i*'s query matches *j*'s key (dot product, scaled, softmaxed). These scores are the **attention weights**; they sum to 1.
3. Nucleus *i*'s update is the attention-weighted sum of the values.

With **4 heads**, this happens 4 times in parallel with different projections, so the model can attend for different reasons (e.g. "similar stain" and "similar size").

Two properties matter:

- **Permutation equivariance.** Shuffling the nuclei in a set shuffles the outputs the same way. There is no position input: the model does not know where nuclei are, only *which nuclei are in the same set*.
- **Padding mask.** Sets have up to N nuclei; short sets are padded. The mask stops attention to padding and the loss ignores padded slots.

### 4.2 Architecture (port of the notebook's `NucleusSetTransformer`)

```
features (d) → Linear(d→64) → GELU                         # per-nucleus embedding
→ SetBlock × 2:                                            # pre-norm transformer block
     x = x + MultiheadAttention(LayerNorm(x))              # 4 heads, dropout 0.1
     x = x + FeedForward(LayerNorm(x))                     # 64 → 128 → GELU → 64
→ LayerNorm → Linear(64→2)                                 # a prediction for EVERY nucleus in the set
```

About 74k parameters. The "+ x" terms are **residual connections**: each block adds a correction to the nucleus's own representation instead of replacing it, so a nucleus can always fall back on its own features.

### 4.3 How the sets are formed

Two versions were tried.

**(a) Per-nucleus sets (`train_set.py`, first attempt).** For each nucleus, the set is that nucleus plus its K nearest annotated neighbours; only the centre nucleus's prediction is used. This did **not** beat the MLP, nor its own controls (K = 0, shuffled neighbours), so it was dropped. The user also pointed out that it wasn't really the notebook's model.

**(b) Partition sets (`train_partition.py`, the notebook's design).** Each slide's annotated nuclei are split into **non-overlapping, spatially compact groups** of at most N nuclei:

- `kd_groups`: recursively cut the point cloud at the median along its wider axis until every piece has ≤ N nuclei. Only centroids are used, never labels.
- Every epoch the coordinates are **randomly rotated** before cutting, so group borders move and the model sees each nucleus with different neighbours.
- The whole group goes through the model at once and **every** nucleus in it gets a prediction and a loss.
- At test time, predictions can be averaged over several rotated partitions (`--tta`; not used in the reported runs, i.e. 1 partition).

### 4.4 How it learns

- **Loss:** cross-entropy over **every real nucleus in every set**, padding ignored (`ignore_index = -100`).
- **Class imbalance:** handled by **class weights in the loss** (weight ∝ 1/count^0.5, normalised to mean 1), not by weighted sampling as in the MLP. Same strength of correction, different mechanism. This is a small difference between the two pipelines to remember when comparing them.
- **Batches:** `--batch_sets` groups per step, chosen as 3072 / N so every model sees about 3,072 nuclei per step regardless of set size (otherwise size 1 would take far more, far smaller steps).
- **Optimiser:** AdamW, learning rate 3e-4, weight decay 1e-3, 5-epoch linear warm-up then cosine decay, 100 epochs, gradient clipping at 5. Final epoch kept.

### 4.5 Controls (same script, one thing changed)

| control | what changes | what it isolates |
|---|---|---|
| `--set_size 1` | every nucleus alone | the transformer layers/optimiser without any context |
| `--shuffle_sets` | groups are random nuclei from the **same slide**, not neighbours | spatial neighbourhood vs "other nuclei from this slide" |
| MLP | different model entirely | the reference everyone must beat |

All configurations were run with **3 seeds** (42, 43, 44). Comparisons are **paired**: same seed against same seed, then averaged, which removes some run-to-run noise. "Slides better" counts on how many of the 6 reference slides the config beat the reference on average.

### 4.6 Context from all detections (`train_context.py`)

Annotated nuclei sit inside polygons, so annotated-only sets are unrealistically clean. At deployment, a set would contain **every** detection in an area. So a second script loads all detections within 100 µm of annotated nuclei, builds sets from all of them, and computes the loss **only on annotated nuclei** (unlabelled ones provide context only, label −100). QuPath features only: context detections have no LEMON embeddings.

---

## 5. The "annotation layout leak" (why annotated-only sets are suspicious)

Annotations are drawn as regions, so a spatial group of annotated nuclei is almost always **all one class**. A set model can then learn "decide for the whole group", e.g. pool evidence from 48 nuclei and give them all the majority answer. That is a strong signal **only because the groups were cut from label-homogeneous regions**. Also, the edge of an annotated region is where context ends, which itself carries information. Neither exists in a real slide, where a group may straddle tumour and stroma and unannotated nuclei fill the gaps.

So: a gain with annotated-only sets that disappears with all-detection context is probably this shortcut. This is a hypothesis, not proven: one diagnostic (mixed vs pure groups, section 7) did not behave as the shortcut would predict.

---

## 6. Tuning the set size with cross-validation

To pick N without touching the reference slides, the 28 training slides were split into **5 folds by slide** (fixed shuffle). For each fold: train on 4/5 of the slides, score the held-out 1/5. The summary averages per-slide accuracy over all 28 held-out slides (seed 42, annotated-only partition model).

| set size | CV slide-mean | fold sd | near other class | far from other class |
|---|---|---|---|---|
| 1 | 85.14 | 6.24 | 71.0 | 86.3 |
| 8 | 87.48 | 6.09 | 67.6 | 88.0 |
| 16 | 88.49 | 5.95 | 71.2 | 88.8 |
| **32** | **90.35** | 4.72 | 72.7 | 89.6 |
| 48 | 90.12 | 5.98 | 74.6 | 89.7 |
| 96 | 89.78 | 5.86 | 77.8 | 90.2 |

- "Near other class" = within 20 µm of an annotated nucleus of the other class (about 22% of held-out nuclei on average). That is the tumour-next-to-stroma case. It is much harder (68–78%) than the interior (86–90%).
- Context helps up to ~32, then flattens. 32, 48, 96 are a tie given fold sd of 5–6 points. **32 was chosen in advance** as the cheapest.
- Training-slide CV scores (~90%) are far above reference-slide scores (~80–86%). That gap between training slides and reference slides is bigger than any model difference in this document.

---

## 7. Results on the reference slides

Annotated reference, slide-mean accuracy %, mean ± sd over 3 seeds.

| config | what it is | all slides | w/o FHS+PLS | paired vs MLP | slides better |
|---|---|---|---|---|---|
| mlp | baseline | 83.06 ± 0.37 | 87.33 | n/a | n/a |
| set_ctx0 / set_ctx32 | per-nucleus sets, K = 0 / 32 | 82.64 / 83.14 | 84.95 / 85.34 | −0.42 / +0.08 | 2/6 |
| part_set1 | partition, size 1 (no context) | 80.89 ± 0.75 | 83.92 | −2.17 | 1/6 |
| **part_set32** | partition, size 32 (CV choice) | **83.03 ± 0.56** | 85.35 | **−0.03** | 2/6 |
| part_set48 | partition, size 48 | 84.38 ± 0.56 | 86.92 | +1.32 | 4/6 |
| part_shuffle48 | random same-slide groups of 48 | **86.39 ± 1.33** | 87.92 | **+3.33** | 5/6 |
| ctx_set1 | all-detection context, size 1 | 81.52 ± 0.42 | 84.23 | −1.54 | 1/6 |
| ctx_set48 | all-detection context, size 48 | 81.31 ± 0.78 | 83.27 | −1.75 | 1/6 |
| ctx_shuffle48 | all-detection, random groups | 83.91 ± 1.06 | 85.04 | +0.85 | 3/6 |
| mlp_sncenter | MLP + per-slide centring | 85.73 ± 0.04 | 86.70 | +2.67 | 4/6 |
| mlp_snz | MLP + per-slide z-score | 85.79 ± 0.13 | 87.50 | +2.74 | 3/6 |
| fus_mlp | MLP, QuPath + LEMON (PCA 64) | 84.15 ± 0.26 | 87.29 | +1.09 | 5/6 |
| pfus_set1 | partition size 1, fusion | 82.28 ± 0.27 | 84.37 | −0.78 | 2/6 |
| pfus_set32 | partition size 32, fusion | 83.93 ± 1.07 | 85.53 | +0.87 | 5/6 |
| pfus_shuf48 | random groups 48, fusion | 84.47 ± 0.31 | 85.92 | +1.42 | 4/6 |

Labelled-binary (where available): MLP 87.26, part_set32 87.19, part_set48 88.83, part_shuffle48 89.54, mlp_sncenter 84.03, mlp_snz 85.12. Fusion runs have no labelled-binary score (no LEMON embeddings for those nuclei).

### What each comparison says

1. **Context vs no context (same network).** part_set32 beats part_set1 by +2.1 on 5/6 slides. Inside the set model, seeing other nuclei clearly helps.
2. **The set model vs the MLP.** The CV-chosen size 32 only **ties** the MLP (−0.03), and is ~2 points worse without the outlier slides. The set network's size-1 version is 2 points *below* the MLP, so the context gain is mostly making up for a weaker per-nucleus model.
3. **CV did not predict the reference ranking.** CV said 32 ≈ 48; on the reference slides 48 is 1.35 ahead. Choosing 48 now would be choosing on the test set, so 32 stands.
4. **Random groups beat spatial groups.** part_shuffle48 (86.4) > part_set48 (84.4) > MLP (83.1). If neighbourhood context were the source, spatial groups should win. They don't. Mixed-label spatial groups also scored *higher* than pure ones in CV (92 vs 88), which doesn't fit the layout-leak story either, so that story is not settled for annotated-only sets.
5. **All-detection context removes the spatial gain.** ctx_set48 ≈ ctx_set1 (81.3 vs 81.5). With realistic context, spatial sets add nothing.
6. **Slide normalisation explains most of the random-group gain.** An MLP whose features are centred on each slide's own mean gets +2.7 of the +3.3 points that random same-slide groups give. The remaining −0.6 gap swings from +1.0 to −1.9 across seeds, so it may be noise.
7. **But naive slide normalisation is fragile.** On labelled-binary it *hurts* (87.3 → 84.0), with RDS falling 95.6 → 65.9. The slide statistics were computed from whichever nuclei were being scored, and the labelled-binary nuclei have a different class mix, so the "slide mean" moves and the correction is wrong. The set model does not suffer this (89.5 on labelled-binary): attention compares nuclei to each other instead of subtracting a fixed mean.
8. **The slide-normalisation gain is concentrated.** PLS 81 → 98, MIS 88 → 94, CIS 91 → 94; RDS and RVS get worse. Without FHS and PLS it is roughly level with the MLP.
9. **LEMON (fusion).** LEMON gives the MLP about +1.1 (5/6 slides, but level without the outlier slides). It does **not** stack with set context: pfus_shuf48 is 1.9 below part_shuffle48. RDS gets worse under fusion in the set models (66.8–71 vs ~75).

---

## 8. Why random same-slide groups help: the calibration explanation

Every slide is stained and scanned a little differently. A darker slide shifts *all* its nuclei's optical-density features in the same direction. The MLP z-scores with training statistics, so a reference slide that is darker than any training slide looks unfamiliar.

A set of nuclei **from the same slide** lets attention compute things like "this nucleus is darker and larger than the others in my set". That relative judgement is the same whether the whole slide is dark or light. In effect, the set model learns **slide-relative features**. Per-slide centring does the same thing by hand, which is why it recovers most of the gain.

This is the key distinction for the thesis: "set context from neighbouring nuclei helps" is **not** supported by these results; "features expressed relative to their own slide help" **is** supported on the annotated reference slides.

---

## 9. Mistakes and corrections along the way

- The labelled-binary test was first described as containing every detection. It doesn't (~3k nuclei per slide), so it can't be used as a leak check.
- Early slide-level differences (PLS, RDS) were attributed to context; the K = 0 control showed the same shifts came from architecture/optimiser.
- The first size-1 control was not batch-matched (64 vs 3,072 nuclei per step); fixed with `--batch_sets 3072/N`.
- A monitoring run held out 5 training slides and was briefly compared with full-data runs; a full-data rerun replaced it.
- The "near %" column was claimed added before it was; then 0.22 was misread as 0.22% instead of 22%.
- Run commands contained `--tta` without a value and `COMMON="..."` as a literal placeholder; both made runs fail instantly (no results affected).
- Fusion first crashed because labelled-binary nuclei have no LEMON embeddings; fusion runs use `--no_labelled`.
- `summarize.py` crashed with a division by zero for configs missing labelled-binary results; fixed.

---

## 10. Where this leaves the project

| claim | status |
|---|---|
| Spatial neighbourhood context beats the MLP | **Not supported.** CV-chosen size ties the MLP; all-detection context gives no gain. |
| Same-slide sets help | Supported on annotated reference (+3.3, 5/6 slides) |
| That help is mostly slide calibration | Supported: per-slide centring recovers +2.7 |
| Naive per-slide normalisation is deployable | **No.** It depends on which nuclei are scored; it breaks labelled-binary |
| LEMON improves the MLP | Small (+1.1), driven by the outlier slides |
| LEMON improves the set model | No |

Caveats: 6 reference slides, 3 seeds, Ki67 only. Region labels, not per-nucleus labels. Training-slide CV and reference slides disagree by ~5–10 points, so there is a distribution gap between them that none of these models closes.

### Open next steps

1. **Whole-slide normalisation statistics** (`--slide_norm_stats all`): compute each slide's mean/sd from *all* its detections, not the scored nuclei. Independent of annotations and class mix, and available at deployment. If an MLP with this matches part_shuffle48 on both test sets, it is the simplest deployable model.
2. If whole-slide normalisation fails, the same-slide set model (part_shuffle48) is the robust option, since it held up on labelled-binary.
3. LEMON for labelled-binary and context detections would need a new manifest and extraction run; only worth it if fusion is kept.
4. ER and PR: now being tested, see section 13.

---

## 11. Files

| file | purpose |
|---|---|
| `train.py` | MLP; `--input qupath/lemon/fusion`, `--slide_norm none/center/z` |
| `set_transformers/train_set.py` | per-nucleus K-neighbour sets (first attempt) |
| `set_transformers/train_partition.py` | notebook-style partition Set Transformer; CV, fusion, controls |
| `set_transformers/train_context.py` | all-detection context, loss on annotated nuclei only |
| `set_transformers/summarize.py` | reference-slide tables and paired deltas from `runs/logs` |
| `set_transformers/tune_summary.py` | CV tables from `runs/tune` |
| `set_transformers/run_compare.sh` | resumable runner: every config x stain x (5 CV folds + reference seeds) |
| `set_transformers/compare_all.py` | CV + reference tables per stain, paired deltas vs `mlp`, ALL STAINS table (`--per_slide`) |
| `slide_stats.py` | whole-slide per-feature mean/sd from the full detection GeoJSON (cached) |
| `nuc2vec/` | contrastive nucleus pretraining, see section 14 |

---

## 12. Update (2026-11-26): whole-slide normalisation

`train.py --slide_norm center|z --slide_norm_stats all` takes each slide's per-feature mean (and sd) from **every detection in the slide's QuPath export**, not from the scored nuclei. No labels used; the same statistics apply to both test sets. MLP, QuPath features, 3 seeds.

| config | annotated, all | annotated w/o FHS+PLS | labelled-binary, all | labelled w/o FHS+PLS |
|---|---|---|---|---|
| mlp | 83.06 | 87.33 | 87.26 | 92.37 |
| part_shuffle48 | 86.39 | 87.92 | 89.54 | 92.46 |
| mlp_sncenter (scored-nuclei stats) | 85.73 | 86.70 | 84.03 | 84.78 |
| **mlp_snallcenter (whole-slide mean)** | **87.29 ± 0.26** | **88.60** | **91.25 ± 0.37** | **94.77** |
| mlp_snallz (whole-slide mean and sd) | 85.01 | 86.54 | 81.47 | 81.46 |

- Whole-slide **centring** is the best model so far on both test sets: +4.2 over the MLP on the annotated reference (better on 4/6 slides), +0.9 over part_shuffle48, and +4.0 over the MLP on labelled-binary.
- The labelled-binary failure of scored-nuclei normalisation is fixed (RDS 65.9 → 96.0), which supports the explanation that the earlier statistics came from unrepresentative subsets.
- Dividing by the whole-slide sd hurts badly (RDS labelled-binary 43.9). The whole-slide spread is probably inflated by non-nucleus or atypical detections on some slides; the mean shift is the useful part.
- RVS is slightly worse with centring on both test sets (annotated 90.9 → 87.7, labelled 93.0 → 91.9).
- Caveat: `center` vs `z` was chosen after seeing Ki67 reference results, so this needs confirmation on data not used for that choice (ER and PR, or training-slide CV).

Current best candidate: **QuPath-feature MLP with whole-slide feature centring.** It needs no set model and no LEMON.

---

## 13. Update (2026-11-26): all models on CV and reference, per stain

Every model is scored two ways per stain: **5-fold slide-level CV on the training slides** (seed 42; ER 30 slides, Ki67 and PR 28) and the **6 annotated reference slides** (3 seeds, 42-44). `lab` is the labelled-binary reference set (not available for fusion runs). "better" counts slides where the config beat `mlp` (CV: held-out training slides; ref: the 6 reference slides). `_norm` = whole-slide centring (`--slide_norm center --slide_norm_stats all`); `fus_` = QuPath + LEMON PCA fusion. Runner: `run_compare.sh`; summary: `compare_all.py --per_slide`.

Reading guide: differences below about 0.5-1 point on the reference slides are within seed sd (0.1-1.2) and slide-to-slide variation with only 6 slides, so the "better" counts are more informative than the means. Ki67 `set1` has only 2 of 3 reference seeds: the redo was stopped, so that row is a 2-seed mean.

### ER

| config | architecture | centring | LEMON | CV | CV vs mlp | CV better | ref | ref vs mlp | ref better | lab | lab vs mlp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | MLP | no | no | 90.56 | +0.00 | - | 95.84 | +0.00 | - | 84.72 | +0.00 |
| mlp_norm | MLP | yes | no | 94.14 | +3.58 | 17/30 | 95.59 | -0.25 | 3/6 | 81.82 | -2.90 |
| set1 | set size 1 | no | no | 87.21 | -3.35 | 8/30 | 94.10 | -1.74 | 1/6 | 83.53 | -1.19 |
| set32 | set 32, spatial | no | no | 91.00 | +0.44 | 25/30 | 92.12 | -3.72 | 1/6 | 88.45 | +3.73 |
| shuf48 | 48 random same-slide | no | no | 92.12 | +1.56 | 19/30 | 94.64 | -1.20 | 1/6 | 83.21 | -1.51 |
| set1_norm | set size 1 | yes | no | 93.64 | +3.08 | 17/30 | 95.48 | -0.36 | 2/6 | 79.92 | -4.80 |
| set32_norm | set 32, spatial | yes | no | 95.20 | +4.64 | 25/30 | 94.47 | -1.37 | 2/6 | 80.91 | -3.81 |
| shuf48_norm | 48 random same-slide | yes | no | 93.73 | +3.17 | 20/30 | 96.54 | +0.70 | 5/6 | 83.56 | -1.16 |
| fus_mlp | MLP | no | yes | 90.92 | +0.36 | 14/30 | 95.56 | -0.28 | 3/6 | - | - |
| fus_mlp_norm | MLP | yes | yes | 94.18 | +3.62 | 17/30 | 95.42 | -0.42 | 2/6 | - | - |
| fus_set32 | set 32, spatial | no | yes | 90.61 | +0.05 | 25/30 | 91.84 | -4.00 | 0/6 | - | - |
| fus_set32_norm | set 32, spatial | yes | yes | 93.94 | +3.38 | 23/30 | 94.03 | -1.81 | 1/6 | - | - |

**ER: effect of each ingredient, change in reference accuracy (CV change in brackets)**

| architecture | vs MLP | + centring (vs same arch.) | + LEMON (vs same arch.) | + centring and LEMON (vs same arch. without either) |
|---|---|---|---|---|
| mlp | - | -0.25 (+3.58) | -0.28 (+0.36) | -0.42 (+3.62) |
| set1 | -1.74 (-3.35) | +1.38 (+6.43) | not run | not run |
| set32 | -3.72 (+0.44) | +2.35 (+4.20) | -0.28 (-0.39) | +1.91 (+2.94) |
| shuf48 | -1.20 (+1.56) | +1.90 (+1.61) | not run | not run |

**ER findings.**
- Architecture (no centring): nothing beats the MLP on the reference slides (`shuf48` -1.2, `set1` -1.7, `set32` -3.7). In CV `set32` is +0.4 and `shuf48` +1.6, but `set32` is better on only 1/6 reference slides.
- Centring: large CV gain for every architecture (MLP +3.6, set1 +6.4, set32 +4.2, shuf48 +1.6). On the reference slides it is flat for the MLP (-0.25), and it lifts `shuf48` from -1.2 to +0.7 vs the MLP (5/6 slides): `shuf48_norm` is the only model above the MLP.
- Fusion: no help. MLP -0.3 on reference (+0.4 CV); `set32` -0.3 and `set32_norm` -0.4. LEMON is not useful for ER here.
- Labelled-binary: centring hurts (`mlp_norm` -2.9; `set32_norm` is 7.5 below `set32`). `set32` alone has a large seed sd (4.4).
- Ceiling: the MLP is already at 95.8 on the reference slides.

### Ki67

| config | architecture | centring | LEMON | CV | CV vs mlp | CV better | ref | ref vs mlp | ref better | lab | lab vs mlp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | MLP | no | no | 84.65 | +0.00 | - | 83.06 | +0.00 | - | 87.26 | +0.00 |
| mlp_norm | MLP | yes | no | 85.07 | +0.42 | 16/28 | 87.29 | +4.23 | 4/6 | 91.25 | +3.99 |
| set1 | set size 1 | no | no | 85.14 | +0.49 | 10/28 | 81.92 | -1.14 | 1/6 | 86.92 | -0.34 |
| set32 | set 32, spatial | no | no | 90.35 | +5.70 | 25/28 | 83.03 | -0.03 | 2/6 | 87.19 | -0.07 |
| shuf48 | 48 random same-slide | no | no | 87.93 | +3.28 | 18/28 | 85.74 | +2.68 | 5/6 | 89.22 | +1.96 |
| set1_norm | set size 1 | yes | no | 83.58 | -1.07 | 18/28 | 87.01 | +3.95 | 4/6 | 91.13 | +3.87 |
| set32_norm | set 32, spatial | yes | no | 88.31 | +3.66 | 22/28 | 86.74 | +3.68 | 3/6 | 89.91 | +2.65 |
| shuf48_norm | 48 random same-slide | yes | no | 87.97 | +3.32 | 19/28 | 87.82 | +4.76 | 5/6 | 91.12 | +3.86 |
| fus_mlp | MLP | no | yes | 86.64 | +1.99 | 21/28 | 84.15 | +1.09 | 5/6 | - | - |
| fus_mlp_norm | MLP | yes | yes | 88.33 | +3.68 | 18/28 | 87.84 | +4.78 | 4/6 | - | - |
| fus_set32 | set 32, spatial | no | yes | 91.73 | +7.08 | 25/28 | 83.93 | +0.87 | 5/6 | - | - |
| fus_set32_norm | set 32, spatial | yes | yes | 91.00 | +6.35 | 22/28 | 87.28 | +4.22 | 5/6 | - | - |

**Ki67: effect of each ingredient, change in reference accuracy (CV change in brackets)**

| architecture | vs MLP | + centring (vs same arch.) | + LEMON (vs same arch.) | + centring and LEMON (vs same arch. without either) |
|---|---|---|---|---|
| mlp | - | +4.23 (+0.42) | +1.09 (+1.99) | +4.78 (+3.68) |
| set1 | -1.14 (+0.49) | +5.09 (-1.56) | not run | not run |
| set32 | -0.03 (+5.70) | +3.71 (-2.04) | +0.90 (+1.38) | +4.25 (+0.65) |
| shuf48 | +2.68 (+3.28) | +2.08 (+0.04) | not run | not run |

**Ki67 findings.**
- Architecture (no centring): `shuf48` is the only gain on the reference slides (+2.7, 5/6). `set32` ties the MLP (-0.03) although it is +5.7 in CV, and `set1` is below (-1.3). The CV gain of `set32` does not transfer.
- Centring: the largest effect in any stain. MLP +4.2 (4/6), set1 +5.1, set32 +3.7, shuf48 +2.1. After centring the architectures sit within 86.7-87.8, so most of the set-model gain was slide calibration (section 8).
- Fusion: +1.1 for the MLP and +0.9 for `set32` without centring; with centring it adds only about +0.5 (`fus_mlp_norm` +0.55 over `mlp_norm`, `fus_set32_norm` +0.54 over `set32_norm`), so LEMON largely overlaps with what centring provides.
- Labelled-binary: centring helps (+4.0 for the MLP), consistent with the reference.
- Provisional: `set1` reference has 2 of 3 seeds.

### PR

| config | architecture | centring | LEMON | CV | CV vs mlp | CV better | ref | ref vs mlp | ref better | lab | lab vs mlp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | MLP | no | no | 84.69 | +0.00 | - | 92.90 | +0.00 | - | 88.12 | +0.00 |
| mlp_norm | MLP | yes | no | 90.62 | +5.93 | 19/28 | 92.95 | +0.05 | 4/6 | 89.25 | +1.13 |
| set1 | set size 1 | no | no | 86.04 | +1.35 | 12/28 | 90.57 | -2.33 | 2/6 | 85.72 | -2.40 |
| set32 | set 32, spatial | no | no | 90.77 | +6.08 | 23/28 | 92.02 | -0.88 | 4/6 | 83.58 | -4.54 |
| shuf48 | 48 random same-slide | no | no | 85.62 | +0.93 | 16/28 | 93.55 | +0.65 | 3/6 | 87.65 | -0.47 |
| set1_norm | set size 1 | yes | no | 90.17 | +5.48 | 16/28 | 89.09 | -3.81 | 3/6 | 87.11 | -1.01 |
| set32_norm | set 32, spatial | yes | no | 93.44 | +8.75 | 22/28 | 90.46 | -2.44 | 4/6 | 86.07 | -2.05 |
| shuf48_norm | 48 random same-slide | yes | no | 89.23 | +4.54 | 17/28 | 93.27 | +0.37 | 4/6 | 88.37 | +0.25 |
| fus_mlp | MLP | no | yes | 87.07 | +2.38 | 18/28 | 93.43 | +0.53 | 3/6 | - | - |
| fus_mlp_norm | MLP | yes | yes | 91.63 | +6.94 | 19/28 | 93.96 | +1.06 | 5/6 | - | - |
| fus_set32 | set 32, spatial | no | yes | 92.92 | +8.23 | 22/28 | 94.39 | +1.49 | 5/6 | - | - |
| fus_set32_norm | set 32, spatial | yes | yes | 94.91 | +10.22 | 25/28 | 95.09 | +2.19 | 5/6 | - | - |

**PR: effect of each ingredient, change in reference accuracy (CV change in brackets)**

| architecture | vs MLP | + centring (vs same arch.) | + LEMON (vs same arch.) | + centring and LEMON (vs same arch. without either) |
|---|---|---|---|---|
| mlp | - | +0.05 (+5.93) | +0.53 (+2.38) | +1.06 (+6.94) |
| set1 | -2.33 (+1.35) | -1.48 (+4.13) | not run | not run |
| set32 | -0.88 (+6.08) | -1.56 (+2.67) | +2.37 (+2.15) | +3.07 (+4.14) |
| shuf48 | +0.65 (+0.93) | -0.28 (+3.61) | not run | not run |

**PR findings.**
- Architecture (no centring): `shuf48` +0.65 (3/6), `set32` -0.9, `set1` -2.3. Nothing is clearly better than the MLP.
- Centring: CV gain for the MLP (+5.9), set1 (+4.1), set32 (+2.7) and shuf48 (+3.6), but on the reference slides it is flat for the MLP (+0.05) and negative for `set1` (-1.5) and `set32` (-1.6). It does not help on the reference slides here.
- Fusion: the only stain where LEMON clearly helps. MLP +0.5 (+1.0 with centring), `set32` +2.4 (5/6 slides). `fus_set32_norm` is best at 95.09 (+2.2 vs MLP, 5/6) and also best in CV (94.91).
- Labelled-binary: everything is within about 2.5 points of the MLP except `set32` (-4.5).

### Across stains

- **Centring:** helps in CV for every architecture on ER and PR; on Ki67 it helps the MLP only slightly (+0.4), is neutral for `shuf48`, and hurts `set1` (-1.6) and `set32` (-2.0) in CV. On the reference slides it helps clearly only on Ki67. Treat it as stain-dependent and keep the non-centred model as the comparison.
- **Set architecture:** `set32` is the best CV architecture on Ki67 and PR but never clearly beats the MLP on the reference slides. `shuf48` is the only architecture above the MLP on the reference slides in Ki67 (+2.7), and in ER and PR after centring or fusion it is level with or slightly above.
- **Fusion:** useful on PR (strongest with `set32`), small on Ki67 and gone once centring is on, absent on ER.
- **Labelled-binary** partly reflects an existing classifier's own calls, so the annotated reference is primary.


---

## 14. Nuc2Vec-style contrastive pretraining (first round, 2026-11-27)

**Motivation.** Abhineet Singh suggested pretraining nucleus embeddings contrastively (Nuc2Vec, Feng et al., MIDL 2021) instead of relying on LEMON, then using them with the MLP or set model.

**What the paper does.** MoCo-v2-style contrastive learning on H&E nuclei with alignment + uniformity losses (Wang & Isola); ResNet34 -> 512 -> 128-d; 128x128 patches at 40x; augmentations including background replacement; evaluation is qualitative (nearest neighbours vs 27 handcrafted features, Ward + HDBSCAN cluster montages, cluster composition entropy, kNN assignment).

**Our adaptation (`LEMON/nuc2vec/`).**

| aspect | choice |
|---|---|
| data | IHC patches, 176 px stored at 0.25 um/px, model sees the central 128 px |
| sampling | detections within 100 um of annotated nuclei in **training** slides, up to 3,500 per slide, labels unused; reference slides are never used |
| balance | not class-balanced (random draw; labels unused). Revisit after clustering if Non Tumor clusters are tiny |
| encoder | ImageNet-init ResNet34 -> 512 -> 128-d, L2-normalised |
| loss | alignment + uniformity, uniformity within the mini-batch (no momentum queue) |
| augmentation | GPU: background paste from another patch of the same slide, rotation/flip/zoom, colour jitter, blur. Colour jitter (brightness, contrast, saturation, hue) was cut from the paper's 0.4/0.4/0.4/0.1 to 0.3/0.3/0.3/0.03 after the first augmentation grid showed hue shifts turning brown DAB olive/red (in IHC the brown vs blue contrast is the signal) |
| background replacement | `--bg_mode box` (default, paper-like): keep a square of side 1.15 x QuPath max diameter. `--bg_mode mask`: keep the nucleus polygon from the detection GeoJSON (`masks.py`), grown 2 px with a 1 px soft edge. Run both and compare |
| variants | pooled encoder over all stains (`n2v_*`) and per-stain encoders (`n2vs_*`) |
| diagnostics | top-1 retrieval of the positive pair in training logs; `probe.py` slide-identity linear probe |

**Pipeline.** `prepare_pretrain.py` (caches patches) -> `pretrain.py` (check `aug_check.png` first) -> `embed.py` (embeds every nucleus in the LEMON manifest) -> `cluster.py` (paper Figs 3-4: dendrogram, HDBSCAN montages, lowest-slide-entropy clusters, `clusters.csv`; labels used for evaluation only) and `neighbours.py` (paper Fig 2 plus same-label@k and same-slide@k) -> `run_compare.sh` with `n2v_*`, `n2vs_*`, `lem_*` configs (MLP, shuf48, fusion, fusion + whole-slide normalisation).

**Pretraining runs.**
- **Box** (`runs/pooled`, pooled over Ki67, ER, PR, 30 epochs, batch 256, about 1,930 nuclei/s): top-1 retrieval 0.686 after epoch 1, 0.93 by epoch 5, 0.972 at the end (chance 0.004); alignment 0.49 -> 0.10; uniformity about -3.93. Top-1 only shows instance discrimination inside a batch and can be solved with slide cues, so it says little about usefulness.
- **Mask** (`runs/pooled_mask`, `--bg_mode mask --mask_dilate 2`): same settings. Its augmentation grid showed the polygons lining up with the nuclei; very faint nuclei leave almost empty views.
- Embeddings: box encoder for all stains (`embeddings/nuc2vec_all_embeddings.npz`, 514,801 x 128) and Ki67 (`nuc2vec_Ki67_embeddings.npz`); mask encoder for Ki67 only (`nuc2vec_pooled_Ki67_mask.npz`, 200,768 x 128).

### 14.1 Label-free diagnostics (Ki67)

| test | QuPath (`all_derived`, z-scored) | LEMON | Nuc2Vec box | Nuc2Vec mask | reference |
|---|---|---|---|---|---|
| slide-identity linear probe (lower = less slide signal) | - | 0.258 | 0.166 | 0.162 | chance 0.036 |
| same-label@k, any slide | 0.889 | 0.777 | 0.763 | 0.746 | - |
| same-label@k, neighbours from other slides only | 0.827 | 0.752 | 0.737 | 0.725 | base rate 0.509 |
| Ward 11-branch label purity | - | - | 0.663 | 0.599 | majority baseline 0.558 |

- Nuc2Vec carries less slide identity than LEMON, but its neighbours agree on the label less often than QuPath's or LEMON's, including across slides. Box is ahead of mask on every test.
- Clustering (paper Figs 3-4): HDBSCAN leaves most nuclei unclustered (box: 3 clusters, 91.5% noise, purity 0.676, NMI 0.127; mask: 2 clusters, Ward NMI 0.033). On the pooled encoder the Ward branches split mainly by stain type (Ki67 vs ER/PR), not by Tumor vs Non Tumor.
- Nearest-neighbour montage (`figs/nn_Ki67.png`): QuPath neighbours match the query nucleus in darkness, size and isolation; LEMON and both Nuc2Vec encoders match the surrounding tissue (crowding, stroma) and less often the nucleus's stain colour or size. The mask encoder's neighbours for an isolated nucleus were mostly dense lymphocyte-like fields.
- Likely cause: the augmentations make the encoder invariant to exactly the cues QuPath uses. Random zoom down to 0.5 (up to 2x magnification) removes nucleus size, and brightness/contrast jitter of 30% weakens stain intensity. Background replacement only happens in training, so at embedding time the encoder still sees the real neighbourhood.

### 14.2 Downstream classification (box encoder, pooled, `n2v_*` configs)

Run with `run_compare.sh`, `EXTRA="--no_labelled"` (labelled-binary nuclei have no Nuc2Vec embedding, which made the first batch fail). Reference = 6 annotated slides, 3 seeds; CV = 5-fold on training slides. Format: **reference (CV)**.

| config | inputs | ER | Ki67 | PR |
|---|---|---|---|---|
| `lem_mlp` | LEMON only | 88.55 (89.77) | 82.25 (83.65) | 91.06 (86.71) |
| `n2v_mlp` | Nuc2Vec only | 88.58 (89.67) | 79.08 (81.15) | 87.35 (84.18) |
| `mlp` | QuPath | 95.84 (90.56) | 83.06 (84.65) | 92.90 (84.69) |
| `fus_mlp` | QuPath + LEMON | 95.56 (90.92) | 84.15 (86.64) | 93.43 (87.07) |
| `n2v_fus_mlp` | QuPath + Nuc2Vec | 94.96 (91.22) | 83.52 (85.81) | 93.10 (87.13) |
| `mlp_norm` | QuPath, centred | 95.59 (94.14) | 87.29 (85.07) | 92.95 (90.62) |
| `fus_mlp_norm` | QuPath + LEMON, centred | 95.42 (94.18) | 87.84 (88.33) | 93.96 (91.63) |
| `n2v_fus_mlp_norm` | QuPath + Nuc2Vec, centred | 95.43 (93.74) | **88.04** (87.57) | 93.45 (91.25) |
| `shuf48_norm` | QuPath, set model, centred | **96.54** (93.73) | 87.82 (87.97) | 93.27 (89.23) |
| `n2v_fus_shuf48_norm` | QuPath + Nuc2Vec, set model, centred | 95.70 (94.11) | **88.44** (90.11) | 93.02 (92.32) |
| `fus_set32_norm` | QuPath + LEMON, set 32, centred | 94.03 (93.94) | 87.28 (91.00) | **95.09** (94.91) |

Per-slide reference accuracy: see `compare_all.py --per_slide`. `n2v_shuf48` (Nuc2Vec only, set model) was not run: `train_partition.py --input` accepts only `qupath` and `fusion`.

**Findings.**
- **As a replacement for LEMON:** tied on ER (88.58 vs 88.55), worse on Ki67 (-3.2) and PR (-3.7). Both embedding-only models fail on ER CIS24-000938 (77.0 and 76.7), so that slide is a domain shift problem for image embeddings in general.
- **As an add-on to QuPath:** with centring, +0.75 on Ki67 and +0.5 on PR over `mlp_norm`, -0.2 on ER; without centring it lowers ER by 0.9. About the same as LEMON on ER and Ki67, 0.5 below on PR.
- **With the set model:** `n2v_fus_shuf48_norm` has the best Ki67 reference score of any config (88.44, 5/6 slides above `mlp`) and +2.1 CV over `shuf48_norm`, but only +0.6 on the reference, which is inside the noise. On ER (-0.8) and PR (-0.25) it is below `shuf48_norm`. There is no LEMON version of this config to compare with.
- **Overall:** this first Nuc2Vec encoder is about as useful as LEMON on ER and a bit weaker on Ki67 and PR, never beats the QuPath features on their own, and gives at most a small, unconfirmed gain in fusion. This matches the diagnostics: it encodes tissue context more than the nucleus.

### 14.3 Next round (pushing Nuc2Vec further)

New `pretrain.py` options: `--p_zoom`, `--zoom_min` (random zoom) and `--slides_per_batch k` (each batch from at most k slides, so negatives come mostly from the same slide and slide cues stop helping).

Planned, one change at a time against the box run, judged first by the cross-slide same-label@k (box 0.737; LEMON 0.752; QuPath 0.827) and then downstream:
1. **Keep nucleus size and stain:** `--p_zoom 0 --jitter 0.1 0.1 0.1 0.0`.
2. **Same-slide negatives:** add `--slides_per_batch 4`.
3. **Per-stain encoders** (`n2vs_*`), since the pooled encoder spends capacity separating Ki67 from ER/PR.
4. **Tighter field of view** if neighbours still match crowding (crop smaller than 128 px at embed and train time, or embed with the background replaced as in mask mode).
5. Optional: fuse QuPath + LEMON + Nuc2Vec; `n2v_shuf48` once `train_partition.py` accepts embedding-only input.

**What would count as a win.** Nuc2Vec beats LEMON with the same model on CV and reference, or adds to the QuPath features where LEMON does not.
