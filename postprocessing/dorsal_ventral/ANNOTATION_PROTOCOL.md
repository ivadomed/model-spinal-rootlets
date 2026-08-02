# Minimal branch-reference protocol

Goal: obtain enough true dorsal/ventral labels to compare deterministic and
learned separators without painting full 3D masks.

## Sampling

- Start with five cases: the three legacy-reference cases, one strongly oblique
  or curved cord, and one high-bridge/failure case.
- Include every visible C2–T1 level and both sides.
- Keep at least one case untouched for evaluation if any model is fitted.

## Annotation unit

- One proximal attachment branch, not one voxel.
- Expert assigns `dorsal`, `ventral`, or `unclear` at the cord attachment.
- Record whether the attachment is directly visible and an optional note.
- Expected scale: roughly four branches × eight levels × five cases = 160
  branch decisions before exclusions.

Suggested table:

```text
subject,level,side,branch_id,class,attachment_visible,reviewer_confidence,notes
```

## Metrics

- Branch macro-F1 and balanced accuracy for visible, non-unclear attachments.
- Per-level and per-side error counts.
- Abstention coverage versus error rate.
- Missing-branch, merged-branch, split-branch, and side/level mismatch counts.
- Inter-rater agreement on a shared subset before resolving disagreements.

## Model gate

- Compare against AP-sign, all-dorsal, and deterministic branch-graph baselines.
- Train a small classifier or GNN only if deterministic errors are repeatable and
  enough labelled branches remain after the held-out case is reserved.
- Never use deterministic pseudo-labels as both training targets and validation.
