"""T3: a learned functional teacher, its controls, and the distillation that follows.

An earlier revision set ``distilled = qualified`` and performed no student update.
The flag was the whole of the distillation. What is here instead:

* :class:`FunctionalTeacher` is a **fitted** predictor of the measured outcome from
  the window's residues -- a one-hot site-by-residue design with a learned
  interaction block, trained on the eligible measured training rows by gradient
  descent, with its own early stop on a held-out slice of the *training* pool.
  It is not a neighbour average. A radius-2 neighbour teacher on a panel whose
  members are at least three edits from the whole old library has no neighbours at
  all and degenerates to one constant, which is the measured reason that teacher
  does not qualify here.
* :func:`qualify` scores the teacher against the two controls the plan names --
  shuffled outcomes and a position-only score -- on the **development** panel, with
  matched training exposure, and returns a decision with its margins.
* :func:`distil` performs actual optimizer steps on the student: the teacher's
  scores over a bank of eligible candidates become a soft ranking target, and the
  student's window log probabilities are fitted to it. The updated parameters are
  checkpointed and the before/after development ranking is measured.
* :func:`rescore` is the alternative use of the same teacher -- reranking the
  student's own candidates without changing its weights -- so "distillation helped"
  and "reranking helped" are separable rather than conflated.

Nothing here treats agreement with a teacher as independent biological validation,
and the structural teacher is a *callback* with a provenance record rather than an
unconditional unavailable flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .her2_runtime import require

TEACHER_FUNCTIONAL = "functional_selection"
TEACHER_NEIGHBOUR = "neighbour_mean"
TEACHER_STRUCTURAL = "structural_esmif1"
TEACHER_ABMAP = "abmap_representation"


def one_hot_windows(windows, *, residues="ACDEFGHIKLMNPQRSTVWY"):
    """``(N, L * 20)`` site-by-residue design matrix. Positions stay distinguished.

    A bag-of-residues encoding would make the teacher position-blind, which is
    exactly the property the position-only control is supposed to isolate.
    """
    rows = [str(window) for window in windows]
    require(rows, "No window to encode")
    length = len(rows[0])
    require(all(len(row) == length for row in rows),
            "A teacher design matrix needs windows of one length")
    index = {residue: position for position, residue in enumerate(residues)}
    out = np.zeros((len(rows), length * len(residues)), dtype=np.float32)
    for row, window in enumerate(rows):
        for position, residue in enumerate(window):
            column = index.get(residue)
            require(column is not None, f"{residue!r} is not canonical")
            out[row, position * len(residues) + column] = 1.0
    return out, length


class FunctionalTeacher(torch.nn.Module):
    """An additive site term plus a learned low-rank interaction block.

    The interaction block is what makes it more than the additive baseline the
    controls already cover: a purely additive teacher cannot rank two substitutions
    at one position differently depending on the background, and neither can a
    position-only score, so an additive teacher that beat the position-only control
    would be evidence about nothing.
    """

    def __init__(self, features, *, rank=16, width=64):
        super().__init__()
        self.linear = torch.nn.Linear(int(features), 1)
        self.project = torch.nn.Linear(int(features), int(rank), bias=False)
        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(int(rank), int(width)), torch.nn.GELU(),
            torch.nn.Linear(int(width), 1))
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, design):
        return (self.linear(design) + self.interaction(self.project(design))).squeeze(-1)

    def document(self):
        return {"kind": "additive site terms plus a low-rank learned interaction",
                "parameters": int(sum(p.numel() for p in self.parameters())),
                "why_not_additive_only": ("a purely additive teacher cannot rank two "
                                          "substitutions at one position differently by "
                                          "background, and neither can the position-only "
                                          "control, so beating that control would be evidence "
                                          "about nothing")}


@dataclass
class TeacherFit:
    """A fitted teacher plus the record of how it was fitted."""

    model: object
    length: int
    history: list = field(default_factory=list)
    holdout: dict = field(default_factory=dict)
    rows: int = 0
    seed: int = 0

    def score(self, windows):
        design, length = one_hot_windows(windows)
        require(length == self.length,
                f"The teacher was fitted on {self.length}-residue windows and was asked for "
                f"{length}")
        with torch.no_grad():
            return [float(value) for value in
                    self.model(torch.as_tensor(design, dtype=torch.float32))]

    def document(self):
        return {"rows": int(self.rows), "seed": int(self.seed),
                "final_loss": self.history[-1]["loss"] if self.history else None,
                "holdout": dict(self.holdout), "updates": len(self.history),
                "model": self.model.document()}


def fit_functional_teacher(windows, values, *, seed=20260927, updates=400, learning_rate=0.01,
                           holdout_fraction=0.2, shuffle_outcomes=False, rank=16):
    """Fit the teacher on measured TRAINING rows. Early stop on a training holdout.

    The holdout is carved out of the training pool, never out of development: the
    teacher's own model selection may not read the panel it is later qualified on,
    or the qualification measures a teacher that was tuned against its own test.
    """
    rows = [str(window) for window in windows]
    outcomes = np.asarray([float(value) for value in values], dtype=np.float64)
    require(len(rows) == outcomes.size and outcomes.size >= 8,
            f"A fitted teacher needs at least eight measured training rows, got {outcomes.size}")
    generator = np.random.default_rng(int(seed))
    if shuffle_outcomes:
        outcomes = outcomes[np.random.default_rng([int(seed), 771]).permutation(outcomes.size)]
    design, length = one_hot_windows(rows)
    order = generator.permutation(len(rows))
    cut = max(1, int(round(float(holdout_fraction) * len(rows))))
    holdout_rows, train_rows = order[:cut], order[cut:]
    require(train_rows.size >= 4, "Too few rows remain after the teacher holdout")
    features = torch.as_tensor(design, dtype=torch.float32)
    targets = torch.as_tensor(outcomes, dtype=torch.float32)
    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        model = FunctionalTeacher(features.shape[1], rank=int(rank))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    train_index = torch.as_tensor(train_rows, dtype=torch.long)
    holdout_index = torch.as_tensor(holdout_rows, dtype=torch.long)
    history, best, best_state, best_step = [], None, None, 0
    for step in range(1, int(updates) + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted = model(features.index_select(0, train_index))
        loss = torch.nn.functional.mse_loss(predicted,
                                            targets.index_select(0, train_index))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            held = float(torch.nn.functional.mse_loss(
                model(features.index_select(0, holdout_index)),
                targets.index_select(0, holdout_index)))
        history.append({"step": step, "loss": float(loss.detach()), "holdout_mse": held})
        if best is None or held < best:
            best, best_step = held, step
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return TeacherFit(model=model, length=length, history=history, rows=len(rows),
                      seed=int(seed),
                      holdout={"rows": int(holdout_rows.size), "best_mse": best,
                               "best_step": int(best_step),
                               "rule": ("carved out of the TRAINING pool. A teacher selected on "
                                        "the development panel is a teacher tuned against the "
                                        "set it is qualified on.")})


def position_only_scores(windows, *, wild_type, teacher):
    """The teacher's score with every position independently reverted to wild type.

    The control the plan names: a score that depends only on *which positions* are
    mutated and not on *what they are mutated to*, in the same units as the teacher.
    Implemented by scoring, for each window, the wild type with each of its own
    substitutions applied one at a time and summing the differences -- an additive
    projection of the teacher itself, so the comparison is against the same function
    with its context dependence removed rather than against a different model.
    """
    from .her2_data import CANONICAL
    baseline = teacher.score([wild_type])[0]
    effects = []
    for position, original in enumerate(wild_type):
        singles = [wild_type[:position] + residue + wild_type[position + 1:]
                   for residue in CANONICAL if residue != original]
        effects.append(float(np.mean(teacher.score(singles))) - baseline)
    out = []
    for window in windows:
        total = baseline
        for position, residue in enumerate(str(window)):
            if residue == wild_type[position]:
                continue
            total += effects[position]
        out.append(float(total))
    return out


def qualify(*, name, real, shuffled, position_only, outcomes, margin, spearman):
    """Does this teacher beat BOTH controls on development, by the declared margin?"""
    values = {"real": spearman(real, outcomes),
              "shuffled": spearman(shuffled, outcomes) if shuffled is not None else None,
              "position_only": spearman(position_only, outcomes)
              if position_only is not None else None}
    controls = [value for key, value in values.items()
                if key != "real" and value is not None]
    best_control = max(controls) if controls else None
    qualified = (len(controls) == 2 and values["real"] is not None
                 and all(np.isfinite(value) for value in values.values())
                 and best_control is not None
                 and values["real"] - best_control >= float(margin))
    return {"teacher": name, **values, "best_control": best_control,
            "margin": float(margin), "qualified": bool(qualified),
            "rule": ("real must beat BOTH the shuffled-outcome and the position-only control by "
                     "the declared margin on the development panel. A teacher that does not is "
                     "removed or redesigned, not distilled."),
            "not_validation": "agreement with a teacher is not independent biological validation"}


def distil(services, route, *, model, heads, teacher, bank, optimizer, updates, batch_size,
           temperature, rng, condition, weight=1.0):
    """Fit the student's window log probabilities to the teacher's soft ranking.

    Real optimizer steps on the student. The target is
    ``softmax(teacher_score / temperature)`` over a drawn batch of eligible bank
    candidates and the loss is the cross entropy of the student's normalised window
    log probabilities against it -- a ranking objective, because the teacher's units
    are not the student's and matching them pointwise would fit the teacher's scale.

    Returns the per-update history and the parameter movement, so "distilled" is a
    measured change rather than a flag.
    """
    require(bank, "Distillation needs a bank of eligible candidates")
    before = {name: value.detach().clone()
              for name, value in model.state_dict().items()}
    history = []
    model.train()
    for step in range(1, int(updates) + 1):
        picked = [bank[int(index)] for index in
                  rng.integers(0, len(bank), min(int(batch_size), len(bank)))]
        scores = np.asarray(teacher.score(picked), dtype=np.float64)
        target = torch.softmax(torch.as_tensor(scores / float(temperature),
                                               dtype=torch.float32), dim=0)
        student = services.window_scores(route, model, picked, condition=condition)
        student_log = torch.log_softmax(student, dim=0)
        loss = float(weight) * torch.sum(target.to(student_log.device) *
                                         (torch.log(target.to(student_log.device) + 1e-12)
                                          - student_log))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
            error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "loss": float(loss.detach()),
                        "gradient_norm": float(norm),
                        "teacher_spread": float(scores.max() - scores.min())})
    moved = 0.0
    for name, value in model.state_dict().items():
        if name in before and value.dtype.is_floating_point:
            moved = max(moved, float((value.detach() - before[name]).abs().max()))
    return {"updates": len(history), "history": history,
            "max_abs_parameter_change": moved,
            "objective": ("cross entropy of the student's normalised window log probabilities "
                          "against softmax(teacher / temperature) over a drawn bank batch. A "
                          "RANKING target: the teacher's units are not the student's and a "
                          "pointwise match would fit the teacher's scale."),
            "distilled": bool(moved > 0.0),
            "distilled_means": ("the student's parameters moved under the teacher's objective. "
                                "A flag set beside a qualification is not distillation.")}


def rescore(candidates, *, student_scores, teacher, weight):
    """Rerank the student's own candidates with the teacher. No weight changes.

    The separable alternative to distillation: same teacher, same candidates, no
    student update. Reporting one of the two and calling it "the teacher helped"
    leaves the reader unable to tell which mechanism produced the change.
    """
    teacher_scores = teacher.score(candidates)
    combined = [float(a) + float(weight) * float(b)
                for a, b in zip(student_scores, teacher_scores)]
    return {"candidates": len(candidates), "weight": float(weight),
            "student": [float(value) for value in student_scores],
            "teacher": [float(value) for value in teacher_scores],
            "combined": combined,
            "mechanism": "reranking only; the student's parameters are unchanged"}


def structural_teacher_provenance(checkpoint, *, allow_read=False):
    """Is a structural teacher obtainable here, and under what provenance?

    The ESM-IF1 checkpoint in the user profile's torch cache is real and readable,
    and the campaign is explicitly not permitted to download into that cache. So the
    answer is not an unconditional "unavailable": it is "present, readable only when
    explicitly qualified, and not distilled until its shuffled and position-only
    controls pass". This function returns that state rather than a flag.
    """
    from pathlib import Path
    target = None if checkpoint is None else Path(checkpoint)
    present = bool(target is not None and target.is_file())
    return {"teacher": TEACHER_STRUCTURAL,
            "checkpoint": None if target is None else str(target),
            "present": present, "read": bool(present and allow_read),
            "size_bytes": (target.stat().st_size if present else None),
            "provenance": ("an existing local checkpoint outside the artifact root. It is read "
                           "only when a run explicitly qualifies it, and nothing is downloaded "
                           "into that cache."),
            "gate": ("obtainable but NOT enrolled by default: a structural teacher enters only "
                     "after its shuffled and position-only controls pass on development, like "
                     "every other teacher here."),
            "reason_skipped": (None if present else
                               "no checkpoint path was supplied or the file is absent")}
