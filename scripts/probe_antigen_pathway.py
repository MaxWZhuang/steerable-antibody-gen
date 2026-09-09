"""Can the antigen stream determine what the MLM head generates?

A capacity probe, not a performance measurement. It answers one narrow question:
is antigen conditioning *expressible* by this architecture and reachable by this
optimizer -- the standard overfit-a-tiny-batch sanity check, specialised so that
it cannot pass vacuously.

The naive version of that check is useless here: the model can memorise the
antibody and ignore the antigen entirely, so driving loss to zero would prove
nothing. Every arm below therefore fixes what the label depends on, and the
arms differ in exactly one axis.

    A  antigen-determined   same antibody, two antigens  -> two targets
    B  antibody-determined  two antibodies, one antigen  -> two targets
    C  pathway control      arm A with the antigen HELD CONSTANT

Arm C is the load-bearing one. It is byte-identical to arm A except that the
second example's antigen is swapped back to the first's, so the input can no
longer distinguish the targets. Its loss must collapse to ln(2) = 0.6931, the
information-theoretic floor for two equiprobable targets. If arm C did NOT
collapse, arm A's success would be unreadable -- the probe would be passing on
something other than the antigen.

Arm B is the positive control: it shows the model can memorise at all, so an
arm-A failure would mean "cannot route the antigen" rather than "cannot learn".

What a result does and does not license
---------------------------------------
Passing is NECESSARY, NOT SUFFICIENT. It shows the pathway exists and carries
gradient on a two-example task. It says nothing about generalisation, about
whether a trained checkpoint actually uses the antigen, or about whether real
affinity data contains learnable ranking signal.

Failure under a fixed step budget shows DIFFICULTY UNDER THIS OPTIMIZATION
SETUP -- this architecture, optimizer, learning rate and budget -- and never
permanent inability. Report the budget with any negative claim.

Checkpoint mode
---------------
``--checkpoint`` rebuilds the model at the checkpoint's own architecture and
runs each arm twice: once from the trained weights, once from a random
initialisation of that same architecture, with identical examples, optimizer and
step budget. The random arm is the control that makes the trained arm readable.
A trained model that is slower than its own random init has settled into a
configuration that resists using the antigen; that is a statement about
optimisation from those weights, not about capacity.

    python scripts/probe_antigen_pathway.py
    python scripts/probe_antigen_pathway.py --checkpoint checkpoints/<dir>/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from smallAntibodyGen.antigen_tokenization import build_antigen_tokenizer
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention, MLMConfig
from smallAntibodyGen.tokenizer import AminoAcidTokenizer

IGNORE = -100
FLOOR = math.log(2)

# Two clearly distinct antigen sequences and two clearly distinct HCDR3 targets.
ANTIGEN_X = "MKTIIALSYIFCLVFADYKD"
ANTIGEN_Y = "GSHSMRYFYTSVSRPGRGEP"
PREFIX_P, PREFIX_Q = "CAR", "CGY"
SUFFIX, HCDR3_A, HCDR3_B = "WGQG", "AAAAAA", "WWWWWW"

#: (prefix, hcdr3, antigen) pairs per arm. Arm C differs from arm A in one field.
ARMS = {
    "A antigen-determined": [(PREFIX_P, HCDR3_A, ANTIGEN_X), (PREFIX_P, HCDR3_B, ANTIGEN_Y)],
    "B antibody-determined": [(PREFIX_P, HCDR3_A, ANTIGEN_X), (PREFIX_Q, HCDR3_B, ANTIGEN_X)],
    "C pathway control": [(PREFIX_P, HCDR3_A, ANTIGEN_X), (PREFIX_P, HCDR3_B, ANTIGEN_X)],
}
CHECKPOINT_ARMS = ("A antigen-determined", "C pathway control")


def build_example(tokenizer, antigen_tokenizer, prefix, hcdr3, antigen, max_length,
                  antigen_max_length):
    """Antibody ids with the HCDR3 span masked, its labels, and antigen ids."""
    ids = list(tokenizer.encode_sequence(prefix + hcdr3 + SUFFIX, locus="IGH",
                                         max_length=max_length))
    labels = [IGNORE] * len(ids)
    start = 2 + len(prefix)  # [CLS][IGH] precede the heavy sequence
    for offset in range(len(hcdr3)):
        labels[start + offset] = ids[start + offset]
        ids[start + offset] = tokenizer.mask_id
    return ids, labels, antigen_tokenizer.encode(antigen, max_length=antigen_max_length)


def pad(rows, value):
    width = max(len(r) for r in rows)
    return torch.tensor([r + [value] * (width - len(r)) for r in rows], dtype=torch.long)


def make_batch(tokenizer, antigen_tokenizer, spec, max_length, antigen_max_length, device):
    built = [build_example(tokenizer, antigen_tokenizer, p, h, a, max_length, antigen_max_length)
             for p, h, a in spec]
    antibody = pad([b[0] for b in built], tokenizer.pad_id).to(device)
    labels = pad([b[1] for b in built], IGNORE).to(device)
    antigen = pad([b[2] for b in built], tokenizer.pad_id).to(device)
    return (antibody, (antibody != tokenizer.pad_id).long(),
            antigen, (antigen != tokenizer.pad_id).long(), labels)


def optimise(model, batch, *, steps, lr):
    """Full-batch AdamW on the masked positions; returns the loss trace and final state."""
    antibody, antibody_mask, antigen, antigen_mask, labels = batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    trace = []
    marks = {0, steps // 8, steps // 4, steps // 2, steps - 1}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(antibody, antibody_mask, antigen, antigen_mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                               ignore_index=IGNORE)
        loss.backward()
        optimizer.step()
        if step in marks:
            trace.append((step + 1, float(loss.detach())))
    model.eval()
    with torch.no_grad():
        logits, _ = model(antibody, antibody_mask, antigen, antigen_mask)
        final = float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      labels.reshape(-1), ignore_index=IGNORE))
        supervised = labels != IGNORE
        accuracy = float(((logits.argmax(-1) == labels) & supervised).sum() / supervised.sum())
    return {"final_loss": final, "token_accuracy": accuracy, "trace": trace}


def report(rows, steps, lr):
    print("{:34}{:>12}{:>11}   {}".format("arm", "final loss", "token acc", "loss trace"))
    print("-" * 104)
    for name, r in rows:
        trace = "  ".join("{}:{:.3f}".format(s, v) for s, v in r["trace"])
        print("{:34}{:>12.4f}{:>11.0%}   {}".format(name, r["final_loss"],
                                                    r["token_accuracy"], trace))
    print("-" * 104)
    print("ln(2) = {:.4f} is the floor when the input cannot distinguish the targets.".format(FLOOR))
    print("budget: {} steps, AdamW lr={}. A negative result is bounded by this budget.".format(
        steps, lr))


def run_fresh(args):
    """Three arms on a small freshly initialised model of the same architecture class."""
    tokenizer = AminoAcidTokenizer()
    antigen_tokenizer = build_antigen_tokenizer("scratch", tokenizer, "")
    config = MLMConfig(vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_id,
                       max_length=args.max_length, d_model=128, n_heads=4, n_layers=2,
                       d_ff=256, dropout=0.0, antigen_max_length=args.max_length)
    device = torch.device("cpu")
    rows = []
    for name, spec in ARMS.items():
        torch.manual_seed(args.seed)  # identical init across arms
        model = AntibodyAntigenCrossAttention(config).to(device)
        batch = make_batch(tokenizer, antigen_tokenizer, spec, args.max_length,
                           args.max_length, device)
        rows.append((name, optimise(model, batch, steps=args.steps, lr=args.lr)))
        print("  ran {}".format(name), file=sys.stderr, flush=True)
    return rows


def run_checkpoint(args):
    """Arms A and C from trained weights and from random init of the SAME architecture."""
    from hcdr3_infill import config_from_checkpoint, load_dual_stream_model
    from mlm_train import build_model, build_tokenizer

    device = torch.device(args.device)
    trained, cfg = load_dual_stream_model(Path(args.checkpoint), data_path="unused",
                                          device=device)
    tokenizer = build_tokenizer()
    antigen_tokenizer = build_antigen_tokenizer(
        getattr(cfg, "antigen_encoder_type", "scratch"), tokenizer,
        getattr(cfg, "esm_model_name", ""))
    max_length = trained.config.max_length
    antigen_max_length = trained.config.effective_antigen_max_length
    print("checkpoint architecture: max_length={} antigen_max_length={} params={}".format(
        max_length, antigen_max_length, sum(p.numel() for p in trained.parameters())),
        file=sys.stderr, flush=True)

    saved = torch.load(Path(args.checkpoint), map_location=device)
    rows = []
    for name in CHECKPOINT_ARMS:
        spec = ARMS[name]
        batch = make_batch(tokenizer, antigen_tokenizer, spec, max_length,
                           antigen_max_length, device)
        for init in ("trained", "random"):
            if init == "trained":
                model, _ = load_dual_stream_model(Path(args.checkpoint), data_path="unused",
                                                  device=device)
            else:
                torch.manual_seed(args.seed)
                model = build_model(tokenizer, cfg, device)
            model.to(device)
            rows.append(("{} [{}]".format(name, init),
                         optimise(model, batch, steps=args.steps, lr=args.lr)))
            print("  ran {} [{}]".format(name, init), file=sys.stderr, flush=True)
    del saved
    return rows


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="probe this checkpoint's architecture, trained vs random init")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=48,
                        help="fresh-model mode only; checkpoint mode uses the saved value")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="write the report as JSON")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        parser.error("--checkpoint does not exist: {}".format(args.checkpoint))
    rows = run_checkpoint(args) if args.checkpoint else run_fresh(args)
    report(rows, args.steps, args.lr)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(
            {"floor": FLOOR, "steps": args.steps, "lr": args.lr, "seed": args.seed,
             "checkpoint": args.checkpoint,
             "arms": {name: r for name, r in rows}}, indent=2), encoding="utf-8")
        print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
