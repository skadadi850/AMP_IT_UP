# AMP Challenge 2027 — plan

## 0. Read this first: the timeline is the binding constraint

The competition proposal states the submission deadline as **October 1, 2026 AOE**.
Today is **August 25, 2026**. That is **about five weeks.**

Verify this on the Kaggle page before doing anything else — the PDF is a
proposal document and dates can slip. But plan for five weeks, because if it is
five weeks, then the ambitious version of this project (train a new pLM-based
diffusion architecture from scratch, mine LC-MS peptidomics, build a novel
predictor) does not fit, and attempting it produces nothing submittable.

The rest of this plan is written around that. It is organised as a **core path**
that reliably produces a strong, co-authorship-eligible submission in five
weeks, plus **stretch modules** that are genuinely novel and that you drop
without regret if time runs out. Register first, verify the pipeline second,
innovate third.

**Register now.** Registration requires an institutional email address — free
webmail is rejected. Use your university address, not a personal one.

---

## 1. What is actually scored

Most of the effort in this field goes into things this competition does not
measure. Read the rubric literally.

### Phase 1 — computational screening (all 50,000 sequences)

Scored with **`seqme`** ([arXiv:2511.04239](https://arxiv.org/abs/2511.04239),
`pip install seqme`), which is written by one of the organizers. Install it and
read the source. The metric families:

| Family | What it measures | `seqme` API |
|---|---|---|
| Surrogate activity | Aggregated AMP-classifier and MIC-predictor scores (AMPredictor, MBC-Attention, DeepAMP) | `HitRate`, `Threshold`, `ThirdPartyModel` |
| Sequence-level | Uniqueness, internal diversity, novelty vs known AMPs (normalised bit-scores), cluster coverage | `Uniqueness`, `Diversity`, `Novelty`, `NGramJaccardSimilarity`, `ClippedCoverage` |
| Distributional | Fréchet Biological Distance, MMD, precision/recall against two reference sets, in ESM2 **and ESM-C** embedding space | `FBD`, `MMD`, `Precision`, `Recall`, `KID`, `AuthPct`, `FKEA` |
| Property | Conformity of charge and amphiphilicity to known AMPs; synthesizability rate | `ConformityScore`, `Charge`, `HydrophobicMoment`, `Gravy` |

**The aggregation weights are withheld until Phase 1 closes.** So do not
optimise a weighted sum you invented. Optimise for *not being bad on any
family*, because an unknown weighting punishes a lopsided profile. A library
that is 95th percentile on four families beats one that is 99th on three and
40th on one, under almost any weighting.

Two specific traps in that table:

- **`AuthPct` (authenticity)** detects memorisation. A model that reproduces
  training data scores well on FBD and terribly here. Sampling temperature that
  looks "safe" is often just memorising.
- **`Precision`/`Recall` are a trade-off, and they are both scored.** Precision
  alone rewards a narrow, high-confidence library; recall alone rewards broad
  coverage. You need a library that covers the AMP manifold *and* stays on it.
  This is the real tension in the whole competition and it is where your
  "maximum diversity with maximum accuracy" instinct is correct — but it has to
  be operationalised as an explicit Pareto choice, not a vibe. See §4.

### Phase 2 — wet lab (this is the part people get wrong)

> "From each qualifying team's top-100 list, **25 peptides are drawn uniformly
> at random**... Team-level scores are the **arithmetic mean** of the relevant
> peptide metric across the team's 25 peptides."

Three consequences, and they are the highest-leverage insights available:

**(a) Ranking within your top-100 is worth nothing.** The draw is uniform. Your
expected score is the mean over all 100. There is no "put the best ones first"
strategy. The ranked list is documentation the organizers asked for; it is not
an input to scoring. Therefore: **your 100th-best candidate matters exactly as
much as your 1st.** Optimise the floor of the list.

**(b) MIC is censored at 64 µM, so failures are catastrophic and asymmetric.**
A peptide with no activity contributes the full ceiling to your mean. A
brilliant peptide at 0.5 µM cannot offset it. Under mean-of-censored-MIC, going
from 20% inactive to 5% inactive is worth far more than improving your best
candidates. **The objective is minimising expected failures, not maximising
predicted potency.** This is why OmegAMP's design philosophy — an
extremely-low-false-positive filter with only 43.5% TPR — is correct here, and
why it produced a 96% wet-lab hit rate. Low recall is fine. You have 50,000
candidates and need 100.

**(c) Correlated failure is the dominant risk.** With n=25 drawn from 100,
sampling variance is large. If all 100 candidates are variants of one motif and
that motif fails under the assay's exact conditions (LB broth,
4×10⁶ CFU/mL, 24 h, 37 °C), your whole score collapses at once. Diversity
inside the top-100 is **risk management**, not novelty theatre. `ranking.py`
implements this as a lower-confidence-bound objective with a per-cluster cap;
`expected_team_score()` simulates the draw so you can compare two candidate
lists on their 5th percentile, not just their mean.

**Five categories** are scored separately: broad-spectrum, Gram-positive,
Gram-negative, MDR, selectivity (SW = HC50/MIC50). You submit **one** top-100
for all five. Decide deliberately whether to build a broad-spectrum list or
tilt toward one category. Broad-spectrum cationic peptides tend to be hemolytic,
so the selectivity category is in direct tension with the potency categories —
which means it is also the least contested. A list built for selectivity is a
defensible differentiation strategy, and HC50 has a ceiling of 128 µM, so
"non-hemolytic" is achievable in a way that "sub-micromolar against 20 strains"
is not.

### The compliance cliff

Non-negotiable, from `verify_submission.py`:

- 50,000 sequences exactly; alphabet `ACDEFGHIKLMNPQRSTVWY`; length 8–50; all
  unique; no exact match to any of the 39,448 sequences in
  `data/reference/antibacterial.fasta`.
- **Top-100: no sequence may exceed 0.80 Levenshtein ratio to *any* reference.**
  This one fails teams. It is a scan against all 39k references, and a single
  violation invalidates the list. Note the real Phase 1 check uses **MMseqs2**
  identity against the **MarLys** database (~102k sequences) — broader than the
  released validator. Run both.
- `uv sync` + entry point, twice, byte-identical output.

`src/ampx/compliance.py` enforces all of this and is already wired into
`generate.py`. It passes today. Do not let it regress.

---

## 2. Track A — the predictor

You want this first, correctly, because it is what selects your top-100 and
therefore determines your Phase 2 score. It is also where the field is weakest
and where a CS contribution is most defensible.

### 2.1 The baseline to beat, and why

**OmegAMP** ([arXiv:2504.17247](https://arxiv.org/abs/2504.17247), code at
`szczurek-lab/OmegAMP`) is the reference point, and it is an organizer's paper.
Its classifier is deliberately unglamorous and it demolishes the deep-learning
baselines:

- **XGBoost**, not a neural net. 276 features: 156 global physicochemical
  descriptors (Biopython, modlAMP, `peptides`, `peptidy`), 20 amino-acid
  composition fractions, and an exponential moving average of the Eisenberg
  hydrophobicity scale along the sequence (positions 1–100).
- **Synthetic negatives are the whole trick.** 100k each of: random sequences,
  **shuffled** AMPs (preserves charge and hydrophobicity, destroys order), and
  **5-position mutants** of AMPs (preserves context, destroys function).
- **Weighted BCE** upweighting experimentally-validated data over synthetic.

Results on held-out AMPs vs challenging negatives: FPR **0.3%**, LR+ **138**,
Prec@100 **90.4%**. Every baseline (amPEPpy, AMPlify, AMPScanner, HydrAMP,
SenseXAMP, PyAMPA) sits at FPR 13–78% and LR+ 1–4. The wet-lab backtest gave
FPR **0.00** on four of six classifiers.

Read the ablation table (their Tab. 16) closely — it tells you where the value
is. Removing all synthetic negatives: LR+ 3.5. Adding random: 4.3. Adding
shuffled: 27.5. Adding mutated: **138.1**. **Mutated negatives are ~80% of the
gain.** If you implement one thing from this paper, implement mutated negatives.

The interpretability result is also worth internalising: mean charge at pH 7
appears in 30% of decision nodes. The model is mostly a sophisticated charge
detector. That is a hint about where headroom remains.

### 2.2 Where the headroom actually is

OmegAMP's classifier has **TPR 43.5%** and is a binary point estimate. Three
real gaps, in ascending order of ambition:

**(i) It is a filter, not a ranker.** You need to select 100 from 50,000 and be
scored on the *mean*. That needs a calibrated continuous prediction with
uncertainty, not a conservative binary gate. Ensemble it, calibrate it
(isotonic / Venn-Abers), and use the lower confidence bound.

**(ii) Nobody in this field does distribution-free risk control.** Your
candidates are out-of-distribution relative to any training set by
construction — that is the point of generating them. Standard calibration
gives you nothing under that shift. **Conformal prediction** does: you can
select a subset with a finite-sample guarantee on the false-discovery
proportion. Concretely, use **conformal selection** (Jin & Candès-style
conformal p-values for selecting candidates whose true value exceeds a
threshold, with FDR control via Benjamini–Hochberg on the conformal p-values).

This is, as far as I can tell from the literature, unexploited in AMP design,
and it maps *exactly* onto the competition's structure: you want a top-100
where the expected fraction of inactive peptides is provably bounded, because
the mean-of-25 rule punishes exactly that fraction. It is a clean CS
contribution, it is cheap to compute, and it is the thing I would build the
paper around.

**(iii) Multi-task MIC regression instead of binary classification.** DBAASP
has per-strain MIC values. The competition scores per-strain MIC, MIC50, MIC90,
and a safety window. Training a multi-task censored regressor (Tobit / interval
regression, since MIC values are right-censored at the assay ceiling) against
the 20-strain panel gets you far closer to the actual objective than a binary
AMP/non-AMP head. Add HC50 as a task for the selectivity category. Most teams
will submit a binary classifier's top-100 into a per-strain MIC scoring rule.

### 2.3 Build order

1. **Reproduce the OmegAMP classifier.** Their features, their synthetic
   negatives, their weighted loss. Do not skip this. It is a strong baseline,
   it takes ~30 min to train on CPU, and it gives you a floor.
2. **Add the hard negative sources they use for evaluation** — signal peptides
   and metabolic peptides from Peptipedia, and add/delete mutants (AD). Hold AD
   out of training as they do; it is the honest test of whether you have learned
   function rather than composition.
3. **Ensemble** — XGBoost + a small ESM-2 fine-tune + a descriptor MLP.
   Disagreement across heterogeneous inductive biases is a much better
   uncertainty signal than a deep ensemble of one architecture.
4. **Calibrate** on a held-out split. Then wrap in conformal selection.
5. **Multi-task MIC/HC50 heads** on DBAASP, with censoring handled properly.

### 2.4 Non-negotiable evaluation hygiene

The single biggest failure mode in AMP prediction papers is leakage from
sequence similarity. Random splits are meaningless here — AMP databases are
full of near-duplicate homologs, so a random split puts variants of the same
peptide on both sides and inflates every number.

**Cluster your data with MMseqs2 at 40% identity and split by cluster.** Report
performance on the clustered split only. If your numbers drop a lot versus a
random split, that gap is the leakage you would otherwise have shipped. Also
evaluate on the AD (add/delete) negatives, which is the test that separates
"learned charge" from "learned function".

---

## 3. Track B — the generator

### 3.1 Do not train a large model from scratch

In five weeks, with Phase 1 scored on distributional similarity and Phase 2 on
a 100-peptide mean, the marginal value of a novel architecture is low and the
variance is high. The two baselines are published with weights. Beat them on
the metrics that are scored.

**Core path (weeks 1–3): conditional diffusion on a biologically-informed
embedding.** Reimplement or adapt OmegAMP's generator. It is the strongest
published option and it is small — 35M parameters, trained for 72 h on a single
GTX 1080. You can afford it.

Its two components:

- **Embedding.** Each residue maps to a 5-vector of physicochemical scales:
  Wimley-White hydrophobicity, isoelectric point, Levitt secondary-structure
  propensity, Zhao-London transmembrane propensity, and the Juretić Average
  Amino Acid Selectivity Index. The scales are chosen so the map is
  **injective**, which makes decoding a nearest-neighbour lookup over 21
  vectors. The PAD token terminates decoding and thereby determines length.
  This beat both one-hot and ESM-2 embeddings in their ablation, and it beat
  ESM-2 at linearly decoding physicochemical properties. The full scale table
  is in their appendix Tab. 9 — transcribe it.
- **Conditioning.** `cond(s) = (1_AMP, length, charge, hydrophobicity)`, with
  elements randomly masked during training so the model learns every subset.
  All conditions are *deterministically computable* — deliberately, to avoid
  conditioning on noisy classifier labels. Sample with self-conditioning and the
  CADS sampler (condition-annealed, which recovers diversity that strong
  conditioning otherwise collapses).

**The two highest-leverage knobs, from their own results:**

1. **Property conditioning with expert ranges.** Charge 2–10, length 10–30,
   hydrophobicity −0.5 to 0.8. This took their classifier hit rate from 10.5%
   to 14.8% and their combined constraint satisfaction to 89.2% versus 39.1%
   for AMP-Diffusion. Same model, different sampling. Nearly free.
2. **Train on general peptides, not just AMPs.** Their ablation: adding 774k
   general Peptipedia peptides to 36k AMPs improved hit rate (8.1→10.5),
   fitness, diversity, *and* conditional controllability (length MAE 0.23→0.04).
   **Your instinct about wider training data is correct and there is a published
   ablation confirming it.** The mechanism matters though: the general peptides
   are not there to be imitated, they are there to teach the representation.
   The AMP flag in the conditioning vector is what keeps generation on-target.
   So: broad pretraining corpus, narrow conditioning. Not a broad corpus you
   sample from uniformly.

### 3.2 Stretch: masked discrete diffusion with multi-objective guidance

If Track A lands early, this is the defensible novelty. The AMP-Diffusion
authors say so themselves in their Cell Biomaterials discussion: they name
**masked discrete diffusion (MDLM)** and **discrete denoising posterior
prediction (DDPP)** as the right next architecture, and note their own model is
unconditional with no guidance mechanism. That is an explicit open problem
posted by one of the two baseline methods.

The natural build:

- **MDLM** backbone over the 20-residue alphabet (Sahoo et al., 2024) —
  discrete, so no embed/decode round-trip loss, and length is handled natively
  by masking.
- **Multi-objective guidance at sampling time.** Two good options:
  **PepTune**'s Monte Carlo Tree Guidance (ICML 2025, Chatterjee lab), or
  **multi-objective-guided discrete flow matching** / **AReUReDI** from the same
  group. These target a Pareto front over several property predictors rather
  than a scalarised objective.

The reason this is the right stretch goal and not a distraction: **the Pareto
framing is the correct formalisation of what you actually want.** "Maximum
diversity with maximum wet-lab accuracy" is not a single objective, and
`seqme` scores precision *and* recall, potency *and* novelty, separately, with
unknown weights. A method that produces a Pareto-diverse library is robust to
the withheld aggregation weights in a way that a scalarised method is not. That
is a genuine, publishable argument, and it is specific to this competition's
design.

**Do not do structural generation.** ESMFold / AlphaFold / ProteinMPNN are the
wrong tools here. Short linear cationic peptides are largely unstructured in
solution and only fold on membrane contact; predicted structures for 8–50-mers
are unreliable, and the competition explicitly forbids anything but linear
peptides with free termini. Structure prediction is a plausible *filter*
(helicity, amphipathic moment) but a poor generator. `seqme` ships an `ESMFold`
model if you want it as a descriptor — that is the right role for it.

---

## 4. Data curation

### 4.1 Core corpus (week 1)

| Purpose | Source | License | Notes |
|---|---|---|---|
| Positives + exclusion | MarLys AMP (MLAMP), ~102k | CC0 | **The reference set.** `data/reference/antibacterial.fasta` is a 39,448-sequence subset with charge/disulfide/activity metadata in the headers — parse it, don't just read sequences. |
| MIC labels | DBAASP v3 | CC BY 4.0 | Per-strain MIC. Filter to standard residues and free termini; standardise medium and CFU as OmegAMP does. |
| Positives | APD6, dbAMP 3.0 | verify / academic | |
| Diversity | AMPSphere | CC BY 4.0 | ~1M candidate AMPs from 60k+ metagenomes. This is your breadth. |
| General peptides + hard negatives | Peptipedia v2.0 | ODbl | The 774k general set for the OmegAMP ablation, plus signal and metabolic peptides as negatives. |

Log everything in `data/README.md` as you go. Full disclosure is a
co-authorship requirement and provenance misrepresentation is disqualifying.

### 4.2 On the LC-MS / sweat / frog / breast milk idea

This is scientifically the most interesting thing you proposed and I want to be
straight with you about it: **it is a great paper and a bad five-week plan.**

Why it is appealing: it is real, it is unexploited, and the organizers'
own senior author built a career on exactly this — the de la Fuente lab's
"molecular de-extinction" work mines extinct proteomes, their gut-metagenome
smORF work synthesised 78 peptides with 59–70% hit rates, and a 2025 ancient-
coprolite study got 36/40 (90%) active. Frog skin peptides are already in your
reference set (the `DADP` tag in those FASTA headers is the Database of Anuran
Defense Peptides). Milk peptidomics is a documented AMP source. `PepSAVI-MS`
exists as a methodology for exactly this.

Why it does not fit five weeks: raw LC-MS peptidomics requires spectral
identification, and the public repositories (PRIDE, MassIVE) hold *spectra*,
not curated peptide lists. Reprocessing them means database search, FDR
control, and deciding what counts as an endogenous peptide versus a digestion
artifact. That is a project, not a task. And crucially, sequences you mine are
**natural peptides** — the competition wants *de novo designs*, and anything
you mine that resembles a known AMP will be caught by either the exact-match
exclusion or the 80% identity ceiling.

**The version that does fit**, and is still novel:

- **AMPSphere already is the metagenomic mining result**, curated and CC BY 4.0.
  Roughly a million peptides from gut, marine, and soil metagenomes. Use it as
  your breadth corpus. You get the "wide band of data" benefit without the
  spectral pipeline.
- **Use exotic sources as *conditioning targets*, not training data.** Take the
  frog-skin subset (`DADP`) or a milk-derived set, compute their property
  vectors, and use OmegAMP-style **subset conditioning** to generate *de novo*
  peptides that inhabit that property region. Their Fig. 2a shows subset
  conditioning on a 750-sequence *A. baumannii* set doubled predicted activity
  over unconditional sampling. That is the mechanism: you steer toward the
  biochemistry of an underexplored source without copying its sequences, which
  is both more novel and compliant by construction.
- **Keep the peptidomics as the follow-up paper.** Write it into your method
  abstract as future work. It will read as vision, not as a gap.

### 4.3 Splitting

MMseqs2 cluster at 40% identity, split by cluster, never by sequence. See §2.4.

---

## 5. Novelty options, ranked

| # | Idea | Effort | Risk | Why it is defensible |
|---|---|---|---|---|
| 1 | **Conformal selection for the top-100** with FDR control on predicted inactivity | Low | Low | Directly matches the mean-of-25 rule; unexploited in AMP design; cheap; makes a clean paper section |
| 2 | **Risk-aware portfolio selection** — optimise the 5th percentile of the simulated draw, with cluster caps | Low | Low | Nobody else will model the sampling process at all; already scaffolded in `ranking.py` |
| 3 | **Multi-task censored MIC/HC50 regression** on the actual 20-strain panel | Medium | Low | Scoring rule is per-strain MIC and SW; a binary classifier is the wrong target |
| 4 | **Pareto-diverse library construction** robust to the withheld Phase 1 weights | Medium | Medium | The withheld-weights design makes robustness an explicit virtue; strong framing |
| 5 | **MDLM + multi-objective guidance** generator | High | Medium | Named as future work by the baseline's own authors |
| 6 | **Property-conditioned generation seeded from exotic-source property manifolds** (frog, milk, metagenome) | Medium | Medium | Your original idea, in a form that is compliant and testable |
| 7 | Raw LC-MS peptidomics mining | Very high | High | Excellent paper, wrong timescale — keep as future work |

Items 1–3 are cheap, land inside five weeks, are individually publishable as a
competition-paper contribution, and — importantly — improve your actual score.
Start there. Items 4–6 are the "really novel work" you asked for; take them in
order as time allows.

---

## 6. Week-by-week tasklist

### Week 0 — right now, before anything else

- [ ] **Confirm the deadline on the Kaggle page.** Everything below depends on it.
- [ ] **Register with your institutional email.** Free webmail is rejected.
- [ ] Push this scaffold to a public GitHub repo under MIT.
- [ ] Run `uv run generate` — confirm it passes. Run it twice, diff the outputs.
- [ ] Run `scripts/verify_submission.py <your-repo-url>` against the pushed repo.
      **You now have a valid, submittable entry.** Everything after this is
      improvement on a working baseline rather than a race to a first one.
- [ ] `pip install seqme`, run it on the two baseline libraries (AMP-Diffusion's
      50k library ships in its repo; HydrAMP's is in the starter kit). These are
      your published Phase 1 targets.

### Week 1 — data and predictor floor

- [ ] Build the dataset: MarLys, DBAASP (with MIC), Peptipedia general + signal
      + metabolic, AMPSphere. Log provenance in `data/README.md`.
- [ ] MMseqs2 cluster at 40%, produce cluster-disjoint splits.
- [ ] Generate synthetic negatives: random, shuffled, 5-position mutants
      (100k each). Hold out add/delete mutants for evaluation only.
- [ ] Reproduce the OmegAMP XGBoost classifier. Target: FPR <1%, Prec@100 >85%
      on the clustered split.
- [ ] Sanity-check against their reported numbers. If you are far off, the
      difference is almost certainly in the negatives or the split.

### Week 2 — generator

- [ ] Implement the 5-scale injective embedding (their Tab. 9) with
      encode/decode round-trip tests.
- [ ] 1D UNet + linear attention, masked property conditioning, self-conditioning.
- [ ] Train on AMPs + general peptides. Compare against AMP-only to confirm you
      reproduce their ablation — this validates your pipeline.
- [ ] Sample with CADS. Generate 50k under property conditioning with the expert
      ranges (charge 2–10, length 10–30, hydrophobicity −0.5 to 0.8).
- [ ] Score with `seqme`. Compare to the baseline libraries. **If you are not
      beating both baselines here, fix this before adding anything.**

### Week 3 — selection, the part that decides Phase 2

- [ ] Ensemble the predictor (XGBoost + ESM-2 head + descriptor MLP), calibrate.
- [ ] Implement conformal selection with FDR control. Validate coverage
      empirically on held-out data.
- [ ] Multi-task MIC/HC50 heads if time permits.
- [ ] Cluster the 50k library; run `select_top` with cluster caps.
- [ ] Run `expected_team_score()` on several candidate top-100 lists. Choose on
      the 5th percentile, not the mean.
- [ ] Verify the 80% ceiling against **both** Levenshtein (validator) and
      MMseqs2 vs MarLys (real Phase 1 check).

### Week 4 — hardening and writing

- [ ] Full 50k run, end to end, from a clean clone. Twice. Byte-identical.
- [ ] Time it. If the organizers' single-GPU run takes hours, shrink it.
      Cache nothing that a fresh clone will not have.
- [ ] Commit checkpoints via git-lfs. Confirm they are actually in the repo and
      that a fresh clone can run inference.
- [ ] Write `docs/METHOD.md`: abstract, training data disclosure, filters
      applied, selection procedure, **LLM-assistance disclosure** (required).
- [ ] Re-run the organizers' validator on the final public repo.

### Week 5 — buffer

Do not plan work here. Compliance failures, an lfs mistake, or a
non-deterministic CUDA kernel discovered late will each eat several days. The
72-hour remediation window after the automated check is not enough time to
retrain anything.

---

## 7. Reading list, in priority order

**Read completely, they define the target:**

1. **OmegAMP** — [arXiv:2504.17247](https://arxiv.org/abs/2504.17247), code
   `szczurek-lab/OmegAMP`. Organizer paper. 96% wet-lab hit rate. Read every
   appendix: C (datasets), D (classifier features and loss), Tab. 9 (the
   embedding scales), J.2 (the synthetic-negative ablation), K (the wet-lab
   protocol you are being scored under).
2. **seqme** — [arXiv:2511.04239](https://arxiv.org/abs/2511.04239), code
   `szczurek-lab/seqme`. This *is* the Phase 1 scoring function.
3. **HydrAMP** — Szymczak et al., Nat. Commun. 14:1453 (2023). Baseline #1.
4. **AMP-Diffusion** — Torres et al., *Cell Biomaterials* 1(9), 2025. Baseline
   #2, and you have the repo. Read the discussion section for their own stated
   next steps.

**Read for the generator:**

5. **MDLM** — Sahoo et al., "Simple and effective masked diffusion language
   models", NeurIPS 2024.
6. **PepTune** — Tang et al., ICML 2025. Masked discrete diffusion + Monte
   Carlo Tree Guidance for multi-objective peptide design.
7. **Multi-objective-guided discrete flow matching** — Chen, Zhang, Tang &
   Chatterjee, 2025 (and **AReUReDI**, arXiv:2510.00352).
8. **CADS** — Sadat et al., arXiv:2310.17347. The sampler that keeps
   conditioning from collapsing diversity.
9. **ProtFlow** — arXiv:2504.10983. Flow matching on compressed pLM embeddings,
   with an AMP section; a cheaper alternative to latent diffusion.

**Read for the predictor and selection:**

10. **APEX** — de la Fuente lab MIC-prediction ensemble
    (`gitlab.com/machine-biology-group-public/apex`). 40 models, 11 species.
    Not in the official metric list, but it is the assaying lab's own model, and
    the AMP-Diffusion repo's `note/` directory has a full benchmark against it
    with per-sequence CSVs. Use it as an independent check.
11. **Conformal selection** — Jin & Candès, "Selection by prediction with
    conformal p-values" (JMLR 2023). The formal tool for item 1 in §5.
12. **AMPredictor, MBC-Attention, DeepAMP** — the three named Phase 1
    surrogates. You do not need to beat them; you need to know what they reward,
    because they are scoring you.
13. **AI-driven AMP discovery: mining and generation** — Szymczak et al.,
    *Acc. Chem. Res.* 58(12):1831 (2025). Organizer-authored survey; the fastest
    way to see how they frame the field.

**Also useful:**

14. **PepCompass** — arXiv:2510.01988. Riemannian navigation of peptide
    embedding spaces, organizer-adjacent. Relevant if you want a principled
    diversity metric in latent space.
15. AMPSphere (Santos-Júnior et al.), the gut-smORF paper (Torres et al., *Cell*
    2024), and the ancient-coprolite AMP paper — for the §4.2 data story.

---

## 8. Things I would specifically avoid

- **Training a generator from scratch on 39k sequences.** Too little data,
  too little time. Condition and sample well instead.
- **Structural generation** (AlphaFold/ProteinMPNN/ESMFold as a generator) —
  see §3.2.
- **Optimising a guessed Phase 1 aggregate.** The weights are withheld
  deliberately. Balance beats spikes.
- **Random train/test splits.** They will make your predictor look excellent
  and select a bad top-100.
- **Ranking effort inside the top-100.** Uniform draw. Spend that effort on the
  floor.
- **Any terminal modification, non-canonical residue, or cyclisation**, however
  good the idea. Explicitly disqualifying.
- **Submitting near-duplicates of a collaborator's entry.** Post-submission
  pairwise overlap analysis across all teams is automated, and confirmed
  collusion disqualifies both.
