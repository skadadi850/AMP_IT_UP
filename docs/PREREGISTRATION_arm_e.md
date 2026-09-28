# Pre-registration — arm E (mutation control on generated seeds)

Written 2026-09-22, **before arm E has been run**. Nothing in this file was
informed by arm E data, because none exists. Committed ahead of the run so the
prediction cannot be adjusted to fit the outcome.

## Why this arm decides the claim

The refinement experiment's surviving result is arm B: inpainting the
generator's own best conditioned output beats spending the same budget on
fresh samples by +0.178 log10 MIC (82 of 89 starting peptides, winning at
every budget from 8 to 160). That comparison holds the operator fixed and
varies the seed source, so it establishes that **novelty headroom** matters —
a known AMP sits at the submission ceiling and 43.9% of its derivatives are
unsubmittable, against 6.4% for a generated peptide.

What it does not establish is that **inpainting** is why arm B wins. Arm B has
never been compared against mutation on its own seeds. Arm D tests the
operator only on known AMPs, where inpainting's advantage turns out to be
band-dependent and negative in the most novel band. So the operator claim and
the headroom claim are currently confounded in exactly the place the paper
would make its strongest statement.

Arm E resolves this. It applies random substitution at arm B's realised edit
distances, derivative-for-derivative, to arm B's identical seeds.

The contrast is also cleaner than B-vs-C. Arm B's seeds are the best of ~5,000
draws by the same regressor that scores their children, so they regress upward
by +0.223 log10 and B-versus-its-own-seeds is meaningless. Arm E starts from
those same seeds, so that max-statistic bias is common to both sides and
cancels in the paired difference — no resampling or budget-matching argument
is needed to defend it.

## What this prediction is for

The prediction below is not evidence and carries no authority. It is not a
hypothesis the experiment is testing, and confirming it is not a result. Its
only function is to make a post-hoc story impossible: whatever arm E returns,
the reading of that outcome was fixed before the number existed.

In particular, **arm E winning is not a failed prediction.** It is a real
result — mutation is the better operator in open space — and it should be
written up as one, with the same weight as the opposite outcome. Nothing here
licenses explaining it away, and any write-up that treats the predicted
outcome as the default and the other as an anomaly has misused this document.

## Prediction

**Arm B beats arm E, and by a larger margin than arm A beats arm D**
(A-vs-D pooled: +0.107 log10 for a_uniform, +0.138 for a_attribution).

Reasoning: arm B's seeds sit at ~0.55 global identity to the reference set,
with room to move in any direction. Both operators therefore have headroom,
and neither is pressed against the submission ceiling. In that regime the
model's advantage should come from knowing what a peptide looks like, rather
than from proximity to a memorised neighbour — which is the only thing that
could have been driving its advantage on known AMPs.

## What each outcome means

| outcome | reading |
|---|---|
| B beats E by more than A beats D | The operator claim survives on its own. Masked diffusion is a better local search operator where there is room to search, and arm A's weaker result was the novelty ceiling, not the operator. |
| B beats E by about as much as A beats D | The operator helps by a roughly constant amount regardless of seed source. Arm B's headline margin over arm C is then mostly headroom, and the honest claim stays "condition, then refine" rather than an operator claim. |
| **E beats B** | Mutation is the better operator in open space, and arm B's result was headroom rather than refinement. The pipeline claim survives — conditioning then refining still beats fresh sampling — but the refinement step should be mutation, not inpainting, and the diffusion-as-operator claim is dead. |

The banded version is what makes this interpretable. **If arm B beats arm E in
the `<0.70` band specifically** — the band where arm A lost to arm D by −0.122
— that reverses the manifold story rather than confirming it, and would be the
strongest result in the experiment. It is also the outcome I would find most
surprising, which is why it is written down here rather than discovered later.

## Conditions this prediction is made under

- Three sampling seeds, analysed with every best-of statistic computed within
  a sampling seed and averaged across them. Standard errors come from the
  across-sampling-seed spread, not the across-peptide-seed spread.
- Arm E mirrors arm B's realised edit distances, not its masked-position
  count. Matching on masked count would make E take ~1.6x more real edits than
  B and the contrast would measure edit distance rather than operator.
- Composition distance to the held-out fold reported alongside predicted MIC
  for every arm, tail and bulk separately. Arm D's best-100 drifts +0.644
  sigma from its own bulk while arm B's drifts +0.012, so an arm E result that
  arrives with a large tail drift is an oracle exploit and not a win, and will
  be read that way.
