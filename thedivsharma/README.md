# NSCE — Null-Space Constrained Edits for SEAL's catastrophic forgetting

This is a from-scratch rebuild (the earlier version of this folder was deleted;
see git history at commit `fdbf1f6` if you want the old one). Same goal, redone
mechanism, redone evaluation methodology.

## The problem

SEAL's own paper (arXiv 2506.10943, §5 Limitations) documents that its
continual self-editing setting — sequentially teaching a model new facts, each
one LoRA-merged permanently into the base weights — suffers catastrophic
forgetting: accuracy on earlier-taught facts degrades as more self-edits get
merged in. The paper doesn't fix this; it names two candidate directions and
leaves them as future work. One of those two is **null-space constrained
edits**, citing AlphaEdit (Fang et al., ICLR 2025) by name. NSCE is that idea,
implemented against this repo's own continual self-edit setting.

## The mechanism

Every self-edit's LoRA update is `dW = scaling * B @ A` (`A: [r, in_features]`,
`B: [out_features, r]`). Because `dW`'s row space is determined entirely by
`A`'s row space, constraining `A` to be orthogonal to a "protected" subspace
of input directions makes `dW` leave those directions untouched to first
order, regardless of what `B` does.

The question is what defines "protected." The redone version here follows
AlphaEdit specifically: for every target weight matrix (`self_attn.q_proj` and
`self_attn.v_proj` in each transformer layer), accumulate a real, empirical
covariance matrix `C = sum_t x_t x_t^T` over the actual input activations
`x_t` the model produces when reading previously-taught passages. The
protected subspace `U` is the top eigenvectors of `C`, kept up to an energy
threshold and hard-capped by a rank budget. A new self-edit's `A` is projected
orthogonal to `U` before merging (`lib/nsce.py`).

This is a deliberate change from the version that used to live in this folder,
which defined "protected" as *the row-space of previous self-edits' own `A`
matrices* — i.e. "whatever direction the optimizer happened to move in last
time," a proxy for what matters, not a measurement of it. It also had no rank
cap, so protected rank grew unboundedly (2,240 → 22,400 over 10 steps in that
version's own pilot), which — left unchecked — eventually protects the entire
input space and starves every future self-edit of room to write anything new.
The rank budget here (`lib/nsce.py::NullSpaceTracker`, default 64/module) is a
direct fix for that specific failure mode, at the honest cost of trading some
retention for guaranteed plasticity.

## What's being compared

Three merge conditions, run on **identical self-edits and identical article
sequences** per seed (only the merge step differs, matching every other part
of the pipeline exactly — see `run_continual.py`):

- **baseline** — plain SEAL merge, `W += scaling * B @ A`, no correction.
- **replay** — the simplest plausible competitor: interleave a sample of
  previously-seen articles' own self-edit text into each new self-edit's
  training data. Exists so NSCE has to beat something cheaper than itself.
- **nsce** — the null-space-constrained merge described above.

For every condition, at every step `k`, the model is evaluated on *every*
article taught so far (`0..k`), building the same lower-triangular
accuracy-matrix structure as the paper's own Figure 6.

## What's genuinely different from a paper-scale run (and why)

This runs on a 16GB M-series laptop, not a cluster, and that constraint shaped
real decisions, not just the model size:

- **Model**: `Qwen/Qwen2.5-0.5B` (base, not `-Instruct`) — see "what broke"
  below for why base-vs-instruct mattered here, not just size. Bfloat16, not
  fp32 (an fp32 1.5B model made the whole machine stop responding during the
  very first smoke test — see below).
- **Grading**: SQuAD F1/EM + substring-containment fallback (`lib/grading.py`),
  not GPT-4.1 semantic grading like the paper — no OpenAI/Azure key was
  available when this run was launched. The code path for Azure OpenAI grading
  exists and activates automatically if `AZURE_OPENAI_API_KEY` /
  `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT` are set before a run
  starts (grading mode is resolved once per run, not per call, so a run's
  results stay internally consistent).
- **Scale**: K=15 articles/sequence, 8 questions/article, 4 independent
  sequences (seeds). Bigger than the old prototype's K=10/7q/no-replicate
  pilot (which its own writeup called too noisy to declare a winner), but
  still far short of the paper's own multi-hour, multi-GPU continual-forgetting
  experiment.
- **Projection is post-hoc, not constrained-training.** `A` is trained freely
  then projected before merging, matching AlphaEdit's own closed-form
  correction. A more principled variant — reparameterizing `A` so gradient
  descent only ever explores the unprotected subspace during training, so no
  capacity is wasted on directions that get projected away afterward — was
  scoped out for time. It should make NSCE's numbers *better*, not worse, so
  this run likely understates NSCE's real ceiling.

## What actually broke while building this (kept here on purpose)

Worth keeping because it's the difference between a result you can trust and
one you can't:

1. **A fp32 1.5B model made the machine stop responding.** Not this process —
   the whole Mac, requiring the user to notice and report it. Fixed with
   bfloat16, gradient checkpointing, a hard `PYTORCH_MPS_HIGH_WATERMARK_RATIO`
   ceiling (so a runaway allocation crashes *this one subprocess* cleanly
   instead of taking the OS down with it), and eventually dropping to a 0.5B
   model once profiling showed the 1.5B model's driver-level memory (~9.3GB)
   left too little margin on a 16GB machine to be safe unattended.
2. **Batching training sequences together looked like the OOM cause; it
   wasn't the dominant one.** Switching to per-example (batch_size=1) steps,
   matching `TTT_server.py`'s own default, didn't fix it — the real cost was
   backward-pass activation memory through 24-28 frozen layers just to route
   gradients into the tiny LoRA branches, fixed with `gradient_checkpointing`.
3. **Variable-length per-example tensors fragmented the MPS allocator.** Even
   after fixing the above, memory still crept up within a single training
   call. Fixed-length padding (identical tensor shapes every iteration, so
   the allocator can actually reuse blocks) resolved it completely — this
   was the fix that finally got a full K=5 run to complete without an OOM.
4. **A subtler, more serious bug: `padding_side="left"` (needed for correct
   batched generation) was left as shared tokenizer state and also used
   during training.** Left-padding shifts real content to high position
   indices; Qwen's rotary embeddings are position-sensitive, so training on
   left-padded input corrupted the model — it produced fluent-looking
   gibberish after merging, not an error, which is the dangerous kind of bug.
   Fixed by setting `padding_side` explicitly per function (right for
   training/activation-capture, left only inside `batched_generate`, restored
   after).
5. **Even after that fix, training still collapsed into gibberish.** This
   traced to two compounding issues, not one: (a) no gradient clipping in the
   hand-rolled training loop (the real `TTT_server.py` trains via HF's
   `Trainer`, which clips to `max_grad_norm=1.0` by default — batch_size=1
   training on a 0.5B model is exactly the regime where one large per-example
   gradient can do real damage with nothing to average it out), and (b) the
   original hyperparameters (6 epochs, lr=3e-4, carried over from the old
   1.5B-model prototype) were simply too aggressive for a 0.5B model on 5-15
   short, topically-repetitive training sequences — a short empirical sweep
   (`/tmp/.../debug_train.py`, not kept in this folder) found epochs=2,
   lr=5e-5 as a stable, still-learning operating point.
6. **Instruct-vs-base model mismatch.** The first attempt used
   `Qwen2.5-0.5B-Instruct` with a bare completion-style training prompt but a
   proper chat-templated eval prompt — different distributions for the two,
   which the paper's own setup avoids entirely by using a *base* model with
   one consistent raw-completion prompt throughout (train and eval). Matching
   that (switching to `Qwen2.5-0.5B`, no chat template anywhere) was necessary
   but on its own did **not** fix the gibberish — items 4 and 5 above turned
   out to be the actual causes; this fix mattered for a different reason
   (keeping train/eval prompts consistent) and is kept for that reason.

None of this changes what NSCE *is* — it's all inner-loop plumbing that had to
be correct before the baseline-vs-nsce comparison could mean anything. A
training loop that occasionally produces gibberish doesn't invalidate a
finding, it makes the finding meaningless, because you can no longer tell
whether an accuracy drop is forgetting or just noise from unstable training.

## Results

Full suite: K=15 articles/sequence, 8 questions/article, 3 independent
sequences (seeds), all three merge conditions on identical self-edits per
seed. Qwen2.5-0.5B base model, F1/EM grading. Raw runs in
`results/runs/seed{1,2,3}_{baseline,replay,nsce}.json`, aggregate in
`results/comparison_summary.json` (regenerate with `analyze_results.py`).

**Retention ratio** (accuracy on an article at the end of the 15-step
sequence, as a fraction of its accuracy right when it was taught — 100% =
perfectly retained, 0% = completely forgotten):

| mode | retention ratio | retained accuracy | plasticity (just-learned) |
|---|---|---|---|
| baseline | 68.1% ± 14.2pp | 6.7% ± 0.5pp | 7.5% ± 1.3pp |
| replay | 65.4% ± 4.9pp | 6.9% ± 0.6pp | 9.7% ± 1.5pp |
| **nsce** | **77.3% ± 3.4pp** | **7.8% ± 0.7pp** | 9.2% ± 0.8pp |

nsce comes out ahead on both retention ratio and final retained accuracy,
while learning about as much initially as replay (both clearly more than
baseline's plasticity — plausibly because seeing more/varied token sequences
per step, from replay's mixed-in examples or nsce's slightly different
effective gradient direction, gives the optimizer more to work with; this
wasn't something the experiment was designed to isolate and shouldn't be
over-read).

**The more important, more honest finding is *why* the aggregate differs, not
just that it does.** Breaking the three seeds apart:

| seed | baseline ratio | nsce ratio |
|---|---|---|
| 1 | 85.7% | 77.1% |
| 2 | 78.6% | 71.4% |
| 3 | **40.0%** | **83.3%** |

nsce is not uniformly better — in seeds 1 and 2 it actually retains *less*
than baseline. The entire aggregate gap comes from seed 3, where baseline
suffered a severe forgetting event (40%) that nsce avoided (83%, in line with
its other two seeds). That is arguably the more faithful reading of what
"catastrophic" forgetting means in the first place — not steady average
decay, but the risk of a severe, seed-dependent collapse — and it reframes
the claim: this run's evidence is that **nsce reduces the incidence/severity
of catastrophic collapses**, more than it uniformly improves average
retention. baseline's own variance across seeds (±14.2pp SEM, driven by that
one collapse) is wide enough that the aggregate gap isn't a clean
mean-separates-from-mean result at n=3 seeds — it's a real, traceable,
seed-3-specific effect that a larger n would be needed to confirm as typical
rather than lucky.

**A concrete, traceable example** (seed 3, article "Boston", taught at step
5, re-evaluated after 10 further unrelated self-edits by step 15):

- **baseline**: 25% (2/8 questions) right after learning → **0%** by the end
  — complete forgetting.
- **nsce**: 38% (3/8 questions) right after learning → **38%** at the end —
  perfectly preserved through the same 10-step sequence of unrelated edits.

This is the single article most responsible for seed 3's gap, and it's
exactly the phenomenon SEAL's own Figure 6 documents: a fact holds up right
after being taught, then gets silently overwritten as later, unrelated
self-edits get merged into the same weights — except here, on this article,
nsce's projection actually prevented it.

**Mechanism diagnostics** (identical measurement in all three modes; only
nsce ever acts on it): mean collision fraction — how much of each new
self-edit's raw update already overlapped the protected subspace before any
correction — climbed to ~7% by mid-sequence and held roughly flat, consistent
across all three conditions (as it should be, since it's a property of the
self-edits and the model, not the merge choice). Protected rank hit its
configured cap (64/module × 48 modules = 3072) by roughly step 8-9 in every
run and *plateaued* rather than growing unboundedly — the direct fix for the
old prototype's failure mode (2,240 → 22,400 uncapped growth over 10 steps)
is doing what it was designed to do.

**What this run does and doesn't establish:** at n=3 seeds, this is evidence
nsce measurably changes the forgetting behavior SEAL's own paper documents as
unsolved — specifically by damping the worst-case collapses more than by
uniformly boosting average retention — not proof that it does so reliably.
The honest next step is more seeds (the baseline variance above all but
demands it before trusting the aggregate number over the per-seed story), not
a bigger model.

## Reproducing

```bash
# from thedivsharma/, using the repo's seal_env conda environment
/opt/anaconda3/envs/seal_env/bin/python run_suite.py     # generates caches + runs all (seed, mode) combos, resumable
/opt/anaconda3/envs/seal_env/bin/python analyze_results.py   # aggregate + report
```

## Honest scope of what this run can and can't claim

Even with everything above fixed, this is a 0.5B model with F1/EM grading —
absolute accuracy numbers will be modest and shouldn't be compared directly to
the paper's own 7B/GPT-4.1-graded numbers. What *is* comparable, apples-to-
apples, is the **relative** comparison between baseline / replay / nsce, since
all three run on identical self-edits, identical hardware, identical grading,
identical everything except the merge step. That relative comparison is the
actual deliverable here — whether NSCE measurably changes the forgetting curve
this repo's own paper documents, not whether a 0.5B laptop model matches
published SOTA numbers.

## A methodological gap worth flagging honestly

`lib/lora.py::LoRALinear.reset()` reinitializes `A` from `torch.randn` without
a fixed seed tied to the run. That means at step 1 of every sequence (before
any protected subspace exists, so baseline/replay/nsce are mechanically
identical there), the three conditions still start from *different* random
LoRA initializations — noticeable in the raw logs as step-1 accuracy
sometimes differing between modes on the same self-edit. This adds variance
to each condition's estimate but shouldn't bias the comparison (init
randomness is independent of which mode it lands in), so the aggregate
result above should still hold in expectation — it's just noisier than an
ideal shared-seed design would be. Fixing it (seed `A`'s init from
`hash(seed, step)` or similar, identical across modes) would tighten the
comparison for a follow-up run.

## Next steps if this is worth taking further

1. **Constrained-training variant** (Phase 1 in the original design
   conversation, scoped out here for time): reparameterize `A` during
   training instead of projecting post-hoc, so no capacity is spent on
   directions that get thrown away.
2. **Scale up** to match the paper's own Figure 6 setup (Qwen2.5-7B, more
   sequences) once the mechanism and hyperparameters are validated here —
   everything in this folder was built to make that a config change, not a
   rewrite (`run_continual.py --lora_r/--lora_alpha/--epochs/--lr`, and
   `generate_self_edits.py --model_name`).
3. **Semantic grading**: set `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT`
   / `AZURE_OPENAI_DEPLOYMENT` before a run to get GPT-4.1-style yes/no
   grading instead of F1/EM — directly comparable to the paper's own
   methodology.
