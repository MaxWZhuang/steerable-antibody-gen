"""Declared comparators for the HER2 core benchmark: prior, distance, neighbour, linear, CNN.

None of these are tuned. They exist because the headline question -- "does a
post-trained sequence policy rank held-out HER2 variants?" -- has several cheap
alternative answers that must be on the same plot: the class prior, distance to
the wild-type core, the label of the nearest training neighbour, and two
supervised classifiers fit on all three published classes.

The two classifiers are a **proxy**, and the word matters. They are independent
of the policy in parameters, architecture and objective; they are *not*
independent in data, because they are fit on the same library the policy's
positives came from, so they reward that library's idiosyncrasies. Their output
is never reported beside a measured assay value without the ``classifier_proxy``
label.

What the CNN is **not**, since the name invites the assumption: it is not the
generator, not a reward model, not the source of any preference pair, and not a
selection criterion for any checkpoint. p-IgGen is the generator. The CNN is a
separately trained auxiliary bin predictor that exists to answer "could a plain
supervised classifier rank these held-out variants?" -- a comparison, nothing
more. Its labels are assay *bins* (high/mid/low), not KD values.

The CNN is a PyTorch port of the upstream topology (Conv1D 400 x k5 -> ReLU ->
Dropout 0.2 -> MaxPool1D 2 stride 1 -> Flatten -> Dense 300 -> ReLU -> head) with
a **3-class** head. It is not a reproduction of the upstream binary Keras recipe,
and every departure is listed in :data:`CNN_DEPARTURES_FROM_UPSTREAM`.
"""
from __future__ import annotations

import numpy as np
import torch

from .her2_data import CANONICAL, CLASS_ORDER, CORE_LENGTH, WT_CORE, encode_cores, hamming_to
from .her2_runtime import require, save_json

CNN_DEPARTURES_FROM_UPSTREAM = (
    "framework: PyTorch, not Keras/TensorFlow",
    "head: 3-class softmax over (low, mid, high), not 1 sigmoid unit",
    "loss: 3-class cross-entropy, not binary cross-entropy",
    "batch size 256, not the upstream 16",
    "Adam learning rate 3e-4 (sqrt-scaled from 16/7.5e-5), not 7.5e-5",
    "no EarlyStopping callback; epoch-level validation selection replaces it",
    "no class weights, matching the upstream class_weight=None default",
)


def class_targets(classes):
    """Map the published class strings onto the fixed (low, mid, high) index order."""
    lookup = {name: index for index, name in enumerate(CLASS_ORDER)}
    values = [lookup.get(str(name)) for name in classes]
    require(all(v is not None for v in values), "Unknown class label")
    return np.asarray(values, dtype=np.int64)


def positive_prior(classes):
    return float((np.asarray(classes) == "high").mean())


def prior_scores(count, prior):
    """A constant scorer. AUROC is 0.5 by construction; AP equals the prevalence."""
    return np.full(int(count), float(prior), dtype=np.float64)


def negative_wt_distance(index):
    """Closer to the wild-type core scores higher. Constant within a d(WT) stratum."""
    return -hamming_to(np.asarray(index), encode_cores([WT_CORE])[0]).astype(np.float64)


# ---------------------------------------------------------------------------
# additive multinomial logistic ("linear") -- 603 parameters
# ---------------------------------------------------------------------------

class AdditiveLinear:
    """One weight per (position, residue, class) plus a per-class bias.

    Fit in float64 on CPU with deterministic LBFGS and strong Wolfe line search;
    there is no sklearn in this environment and none is needed.

    The explicit ridge does two things, and both should be said. It **regularizes**
    -- 1e-4 is small but it shrinks the weights and changes the fit, so this is not
    a "purely numerical" term. It also removes part of a genuine degeneracy: an
    additive softmax model is invariant to adding a constant to all residues at a
    position (and to all classes), so the unpenalized problem has a flat manifold
    of optima and the weights alone are arbitrary. The ridge pins the weight part
    of that manifold. It does **not** make the whole fit unique: the bias is
    deliberately unpenalized, and the softmax's overall per-class shift is
    unidentifiable without it, so no uniqueness is claimed. Predicted
    probabilities, which are what this model is used for, are unaffected by the
    shift that remains.
    """

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    @property
    def parameter_count(self):
        return int(self.weight.numel() + self.bias.numel())

    def logits(self, index):
        values = torch.as_tensor(np.asarray(index), dtype=torch.long)
        total = self.bias.expand(values.shape[0], -1).clone()
        for position in range(CORE_LENGTH):
            total = total + self.weight[position][values[:, position]]
        return total

    def probabilities(self, index):
        with torch.no_grad():
            return torch.softmax(self.logits(index), dim=-1).numpy()

    def positive_scores(self, index):
        return self.probabilities(index)[:, CLASS_ORDER.index("high")].astype(np.float64)


def fit_additive_linear(index, targets, *, ridge=1e-4, max_iterations=100):
    """Deterministic full-batch fit. Returns the model and its optimization record."""
    values = torch.as_tensor(np.asarray(index), dtype=torch.long)
    labels = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    require(values.shape[1] == CORE_LENGTH and labels.shape[0] == values.shape[0],
            "Mismatched additive-model inputs")
    weight = torch.zeros(CORE_LENGTH, len(CANONICAL), len(CLASS_ORDER), dtype=torch.float64,
                         requires_grad=True)
    bias = torch.zeros(len(CLASS_ORDER), dtype=torch.float64, requires_grad=True)
    model = AdditiveLinear(weight, bias)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=max_iterations,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                  tolerance_change=1e-12)
    trace = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        cross_entropy = torch.nn.functional.cross_entropy(model.logits(values), labels)
        loss = cross_entropy + ridge * (weight ** 2).sum()
        loss.backward()
        trace.append({"objective": float(loss.detach()),
                      "cross_entropy": float(cross_entropy.detach())})
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final = torch.nn.functional.cross_entropy(model.logits(values), labels)
    fitted = AdditiveLinear(weight.detach(), bias.detach())
    return fitted, {"parameters": fitted.parameter_count, "ridge": ridge,
                    "max_iterations": max_iterations, "objective_evaluations": len(trace),
                    "train_cross_entropy": float(final), "first_objective": trace[0]["objective"],
                    "final_objective": trace[-1]["objective"]}


# ---------------------------------------------------------------------------
# CNN proxy
# ---------------------------------------------------------------------------

class MasonCNN(torch.nn.Module):
    """Upstream-derived topology with a 3-class head. See CNN_DEPARTURES_FROM_UPSTREAM."""

    def __init__(self, *, channels=400, kernel=5, dense=300, dropout=0.2):
        super().__init__()
        self.conv = torch.nn.Conv1d(len(CANONICAL), channels, kernel_size=kernel,
                                    padding=kernel // 2)
        self.dropout = torch.nn.Dropout(dropout)
        self.pool = torch.nn.MaxPool1d(2, stride=1)
        self.dense = torch.nn.Linear(channels * (CORE_LENGTH - 1), dense)
        self.head = torch.nn.Linear(dense, len(CLASS_ORDER))

    def forward(self, one_hot):
        hidden = self.pool(self.dropout(torch.relu(self.conv(one_hot))))
        return self.head(torch.relu(self.dense(hidden.flatten(1))))


def one_hot(index, device="cpu"):
    """``(B, 20, 10)`` channels-first one-hot, the layout Conv1d expects."""
    values = torch.as_tensor(np.asarray(index), dtype=torch.long, device=device)
    encoded = torch.zeros(values.shape[0], len(CANONICAL), CORE_LENGTH, device=device)
    encoded.scatter_(1, values.unsqueeze(1), 1.0)
    return encoded


@torch.no_grad()
def cnn_probabilities(model, index, *, batch_size=1024, device="cpu"):
    model.eval()
    chunks = []
    for start in range(0, len(index), batch_size):
        logits = model(one_hot(index[start:start + batch_size], device=device))
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def cross_entropy_of(model, index, targets, *, batch_size=1024, device="cpu"):
    model.eval()
    total, rows = 0.0, 0
    labels = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    for start in range(0, len(index), batch_size):
        logits = model(one_hot(index[start:start + batch_size], device=device))
        chunk = labels[start:start + batch_size].to(device)
        total += float(torch.nn.functional.cross_entropy(logits, chunk, reduction="sum"))
        rows += len(chunk)
    return total / rows


def train_cnn(train_index, train_targets, val_index, val_targets, settings, *, seed, device,
              directory):
    """Ten fixed passes over every labelled training row; select by val 3-class CE.

    Selection reads validation only. No early stopping, no threshold tuning, and
    the epoch is chosen before anything looks at the test split or the assay.
    """
    torch.manual_seed(int(seed))
    model = MasonCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    history, best, best_state = [], None, None
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = np.random.default_rng([int(seed), epoch]).permutation(len(train_index))
        running, seen = 0.0, 0
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(one_hot(train_index[rows], device=device))
            labels = torch.as_tensor(train_targets[rows], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            require(bool(torch.isfinite(loss)), "Nonfinite classifier loss")
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(rows)
            seen += len(rows)
        entry = {"epoch": epoch, "train_cross_entropy": running / seen,
                 "val_cross_entropy": cross_entropy_of(model, val_index, val_targets,
                                                       device=device)}
        history.append(entry)
        print(f"cnn seed {seed} epoch {epoch}/{settings['epochs']} "
              f"val CE {entry['val_cross_entropy']:.6f}", flush=True)
        if best is None or entry["val_cross_entropy"] < best["val_cross_entropy"]:
            best = dict(entry)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_json(directory / "history.json", history)
    model.load_state_dict(best_state)
    return model.to(device), {"history": history, "selected_epoch": best["epoch"],
                              "selected_val_cross_entropy": best["val_cross_entropy"],
                              "departures_from_upstream": list(CNN_DEPARTURES_FROM_UPSTREAM),
                              "parameters": int(sum(p.numel() for p in model.parameters()))}


def ensemble_probabilities(models, index, *, batch_size=1024, device="cpu"):
    """Mean softmax across the classifier seeds, plus each seed separately.

    The ensemble is a fixed configuration declared before the fit; no seed is
    chosen by how good its generations look.
    """
    per_seed = [cnn_probabilities(model, index, batch_size=batch_size, device=device)
                for model in models]
    return np.mean(per_seed, axis=0), per_seed
