# NSCE results — TL;DR

SEAL trains a model to write its own training data ("self-edits") and merge
it permanently into its weights. The SEAL paper (arXiv 2506.10943) admits
this causes catastrophic forgetting when you do it repeatedly — new
self-edits overwrite old ones. In their limitations section they mention
"null-space constrained edits" as a possible fix, but never actually build
or test it. That's what this does.

**The idea:** before merging a new self-edit into the weights, figure out
which directions in the model's activations it actually uses to recall
previously-taught facts, and constrain the new update so it can't write into
those directions. Math is in the main README. I tested it against plain SEAL
merging and against a simple "replay old examples" baseline, using the same
self-edits for all three so it's a fair comparison, across 3 separate
15-article sequences.

## The result

| condition | retention (% of a fact remembered by the end vs. right after learning it) |
|---|---|
| plain SEAL merge | 68.1% (±14.2) |
| replay baseline | 65.4% (±4.9) |
| NSCE | **77.3% (±3.4)** |

NSCE comes out ahead , the whole gap comes from one
sequence where plain SEAL had a bad forgetting episode (down to 40%
retention) and NSCE didn't (83%).

Here's the clearest single example, from that sequence: an article about
Boston gets answered correctly 25% of the time right after the model is
taught it. Ten self-edits later (all on unrelated topics), plain SEAL's
version of the model has completely forgotten it — 0%. NSCE's version is
still at 38%, untouched.

So the honest takeaway isn't "NSCE makes the model remember more on
average," it's "NSCE stops the worst forgetting episodes from happening."
Which is actually closer to what "catastrophic" forgetting is supposed to
mean anyway , not a slow leak, but a sudden collapse.


## If you want more detail

- `../README.md` has the full writeup — the actual math, what broke while I
  was building this (a decent amount, since a training loop that quietly
  produces garbage output makes any forgetting comparison meaningless, so
  most of the real work was making sure that wasn't happening), and what I'd
  do next with more time.
- `comparison_summary.json` has the raw numbers behind the table above.
- `runs/` has the full per-step results for every individual run.
