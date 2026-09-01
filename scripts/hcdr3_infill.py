#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallAntibodyGen.data.MLMCollator import OASSequenceDataset
from smallAntibodyGen.infill.hcdr3 import (
    AntigenCompatibilityScorer,
    EmpiricalHCDR3LengthPrior,
    FixedLengthHCDR3Infiller,
    LearnedLengthProposal,
    HCDR3Span,
)
from smallAntibodyGen.models.mlm import AntibodyAntigenCrossAttention
from smallAntibodyGen.tokenizer import AminoAcidTokenizer
from mlm_train import TrainConfig, _train_config_defaults, build_model, build_tokenizer, choose_device


# Repo-anchored (not CWD-relative) and pointed at the dir the stage-3 real-label
# config (configs/refine_antigen_real_label.yaml) actually writes (`_v5`), not the
# CLI-default `_refine` stage-name dir the chain never produces.
# Bumped v3 -> v4 with the pre-LN (norm_first) chain retrain, then v4 -> v5 with the
# 288/1024 context retrain (Open Decision 1) -- each time in the same commit as the
# config change: this default and the stage-3 output_dir must move together, or
# generation silently scores every candidate against the wrong generation's model.
DEFAULT_SCORE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "mlm_antigen_real_label_v5" / "best.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for antigen-conditioned HCDR3 generation.

    The CLI works with existing processed antibody-antigen JSONL files. A
    target record supplies the framework, optional light chain, antigen, and
    HCDR3 insertion boundaries. ``--length-mode fixed`` uses the record's known
    HCDR3 length. ``--length-mode empirical`` samples lengths from positive
    binder HCDR3 lengths in the training split and then asks the same fixed-
    length infiller to generate residues for each proposed length.

    ``--guidance-strength`` opts into ProteinGuide-style antigen-binder
    guidance. When it is ``0`` (default) the CLI uses the original single-pass
    ``infill`` sampler and output is unchanged. When it is ``> 0`` the CLI
    switches to the iterative ``guided_infill`` sampler, which steers each
    residue toward the binder class using the *generation model's own*
    compatibility head. This is distinct from ``--score-checkpoint``, which only
    attaches a post-hoc compatibility score for reporting and never influences
    sampling.

    Guidance additionally refuses to run when the head supplying the binder term
    was trained with ``compatibility_loss_weight: 0`` — the shipped stage-4
    setting — unless ``--allow-untrained-guidance-head`` is passed. See
    ``assert_guidance_head_was_trained`` for what that guard does and does not
    establish.
    """
    parser = argparse.ArgumentParser(description="Generate antigen-conditioned HCDR3 infill candidates.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--record-id", type=str)
    parser.add_argument("--num-records", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--length-mode", choices=("fixed", "empirical", "learned"), default="fixed"
    )
    parser.add_argument(
        "--learned-length-mode",
        choices=("sample", "top_k"),
        default="sample",
        help="How --length-mode learned turns the posterior into proposals: "
        "'sample' draws from it, 'top_k' takes the most probable feasible lengths.",
    )
    # ProteinGuide-style guidance (opt-in; 0.0 keeps the original infill path).
    parser.add_argument(
        "--guidance-strength",
        type=float,
        default=0.0,
        help="Binder-guidance factor gamma. 0 disables guidance (uses single-pass infill); "
        ">0 enables iterative guided_infill steering toward the binder class.",
    )
    parser.add_argument(
        "--guidance-order",
        choices=("confidence", "random", "left_to_right"),
        default="confidence",
        help="Unmasking order for guided decoding (only used when --guidance-strength > 0).",
    )
    parser.add_argument(
        "--guidance-target",
        type=int,
        default=1,
        choices=(0, 1),
        help="Compatibility-head class index to steer toward (1 = binder).",
    )
    parser.add_argument(
        "--guidance-checkpoint",
        type=str,
        default=None,
        help="Optional separately trained guidance classifier (a dual-stream "
        "checkpoint) whose compatibility head supplies the binder term instead "
        "of the generation model's own head. Requires --guidance-strength > 0.",
    )
    parser.add_argument(
        "--allow-untrained-guidance-head",
        action="store_true",
        help="Steer with a compatibility head that received compatibility_loss_weight "
        "== 0 in the run that produced its checkpoint. Off by default: such a head is "
        "frozen at whatever the PREVIOUS stage left while the representation feeding "
        "it kept training, so its binder term is not a validated classifier signal. "
        "Use only for a deliberate negative control, and say so in the writeup.",
    )
    parser.add_argument("--score-checkpoint", type=str, default=None)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def config_from_checkpoint(checkpoint: dict[str, Any], *, data_path: str, device: str | None = None) -> TrainConfig:
    """
    Reconstruct a TrainConfig from a saved checkpoint.

    Older checkpoints predate the HCDR3 infilling fields, so this helper starts
    from current dataclass defaults and overlays the checkpoint's saved
    ``train_config`` dictionary. ``data_path`` and ``device`` are supplied by
    the generation CLI because the checkpoint may have been trained on a
    different machine or with a different runtime device.
    """
    merged = _train_config_defaults()
    saved = checkpoint.get("train_config")
    if isinstance(saved, dict):
        # Only overlay keys that are still TrainConfig fields, so a checkpoint
        # saved under an older/newer schema (a renamed or removed field) cannot
        # crash reconstruction with an unexpected keyword argument.
        valid_fields = {f.name for f in fields(TrainConfig)}
        merged.update({k: v for k, v in saved.items() if k in valid_fields})
    merged["data_path"] = data_path
    if device is not None:
        merged["device"] = device
    cfg = TrainConfig(**merged)
    cfg.validate()
    return cfg


def load_dual_stream_model(
    checkpoint_path: Path,
    *,
    data_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, TrainConfig]:
    """
    Load an antibody-antigen checkpoint for infilling or scoring.

    The model architecture is reconstructed from the checkpoint's training
    config, then the checkpoint weights are loaded directly. The function
    expects a dual-stream checkpoint because HCDR3 infilling here is
    antigen-conditioned.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = config_from_checkpoint(checkpoint, data_path=data_path, device=str(device))
    tokenizer = build_tokenizer()
    model = build_model(tokenizer, cfg, device)
    if not isinstance(model, AntibodyAntigenCrossAttention):
        raise ValueError(
            f"checkpoint {checkpoint_path} reconstructs training_stage = {cfg.training_stage!r},\n which builds {type(model).__name__};\n HCDR3" 
            f"infilling needs a dual-stream AntibodyAntigenCrossAttention checkpoint. "
        )
    # strict=True so a renamed/resized/mismatched checkpoint fails loudly instead
    # of silently leaving submodules at random init and generating garbage.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def build_infiller(
    model: torch.nn.Module,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    device: torch.device,
    guidance_model: torch.nn.Module | None = None,
    guidance_cfg: TrainConfig | None = None,
) -> FixedLengthHCDR3Infiller:
    """
    Construct a FixedLengthHCDR3Infiller wired to the checkpoint's antigen stream.

    The antigen stream is tokenized independently of the model, so the infiller
    MUST be told which antigen encoder the checkpoint was trained with. If these
    fields are dropped, an ``antigen_encoder_type="esm"`` checkpoint is generated
    with the *scratch* amino-acid tokenizer: scratch token ids are then fed into
    an ESM encoder, silently corrupting the antigen representation at generation
    time with no error raised (the antigen 3-site consistency rule). Threading
    ``cfg`` here keeps generation byte-identical to training for both encoders.

    When ``guidance_model`` is provided, its OWN checkpoint config
    (``guidance_cfg``) is threaded the same way -- its antigen encoder settings
    and ``max_length`` must come from the guidance checkpoint, not the generation
    one, for exactly the reason above. Omitting it leaves the infiller in its
    default behavior (the generation model's own head supplies the binder term).
    """
    kwargs: dict[str, Any] = dict(
        max_length=cfg.max_length,
        device=device,
        antigen_encoder_type=cfg.antigen_encoder_type,
        esm_model_name=cfg.esm_model_name,
        antigen_max_length=cfg.antigen_max_length,
    )
    if guidance_model is not None:
        if guidance_cfg is None:
            raise ValueError("guidance_cfg is required when guidance_model is provided")
        kwargs.update(
            guidance_model=guidance_model,
            guidance_antigen_encoder_type=guidance_cfg.antigen_encoder_type,
            guidance_esm_model_name=guidance_cfg.esm_model_name,
            guidance_antigen_max_length=guidance_cfg.antigen_max_length,
            guidance_max_length=guidance_cfg.max_length,
        )
    return FixedLengthHCDR3Infiller(model, tokenizer, **kwargs)


def build_compatibility_scorer(
    model: torch.nn.Module,
    tokenizer: AminoAcidTokenizer,
    cfg: TrainConfig,
    device: torch.device,
) -> AntigenCompatibilityScorer:
    """
    Construct an AntigenCompatibilityScorer wired to its checkpoint's antigen stream.

    The scorer encodes the antigen with its *own* checkpoint's antigen tokenizer
    (which may differ from the generation model's), so it is passed the score
    checkpoint's config. Dropping the antigen-encoder fields here reintroduces
    the same scratch-tokens-into-ESM-encoder corruption described in
    ``build_infiller``, this time on the reported compatibility score.
    """
    return AntigenCompatibilityScorer(
        model,
        tokenizer,
        max_length=cfg.max_length,
        device=device,
        antigen_encoder_type=cfg.antigen_encoder_type,
        esm_model_name=cfg.esm_model_name,
        antigen_max_length=cfg.antigen_max_length,
    )


def select_records(dataset: OASSequenceDataset, *, record_id: str | None, num_records: int) -> list[Any]:
    """
    Select target records from one split for candidate generation.

    Records without valid HCDR3 spans are skipped because the current unknown-
    length infrastructure still needs numbered framework boundaries: it hides
    the length of the replacement loop, not the location of the loop.
    """
    selected: list[Any] = []
    for record in dataset.records:
        if record_id is not None and record.record_id != record_id:
            continue
        try:
            HCDR3Span.from_record(record)
        except ValueError:
            continue
        selected.append(record)
        if record_id is not None or len(selected) >= num_records:
            break
    if record_id is not None and not selected:
        raise ValueError(f"record_id not found with a valid HCDR3 span: {record_id}")
    return selected


def candidate_to_json(
    *,
    record: Any,
    true_span: HCDR3Span,
    length_mode: str,
    candidate: Any,
    guidance_strength: float = 0.0,
    guidance_order: str | None = None,
    guidance_checkpoint: str | None = None,
) -> dict[str, Any]:
    """
    Convert one generated candidate into a JSON-serializable output row.

    ``guidance_strength`` / ``guidance_order`` are recorded for provenance so a
    downstream consumer can tell guided candidates from unguided ones and knows
    which schedule produced them. ``guidance_order`` is reported only when
    guidance was actually active (``guidance_strength > 0``).
    ``guidance_checkpoint`` records WHICH classifier drove the binder term: a path
    string when an external guidance checkpoint was attached, ``None`` when the
    generation model's own head was used (or when guidance was off). Without it a
    merged sweep cannot tell self-judged rows from externally-judged ones.
    """
    return {
        "record_id": record.record_id,
        "target_key": record.target_key,
        "target_name": record.target_name,
        "split": record.split,
        "length_mode": length_mode,
        "true_hcdr3": true_span.original_hcdr3,
        "true_hcdr3_length": true_span.length,
        "proposed_hcdr3_length": candidate.length,
        "generated_hcdr3": candidate.generated_hcdr3,
        "generated_heavy_sequence": candidate.heavy_sequence,
        "log_probability": candidate.log_probability,
        "mean_log_probability": candidate.mean_log_probability,
        "compatibility_score": candidate.compatibility_score,
        "guidance_strength": guidance_strength,
        "guidance_order": guidance_order if guidance_strength > 0 else None,
        "guidance_checkpoint": guidance_checkpoint if guidance_strength > 0 else None,
    }


def assert_guidance_head_was_trained(
    binder_cfg: TrainConfig,
    *,
    checkpoint_path: Path,
    is_external: bool,
    allow_untrained: bool,
) -> None:
    """
    Refuse to steer with a compatibility head that was given zero loss weight.

    Guidance multiplies ``gamma`` by ``log p(binder | x[pos := a])`` read off one
    checkpoint's compatibility head. Whether that number means anything is an
    assumption this CLI can partly check: if the run that produced the checkpoint
    set ``compatibility_loss_weight: 0``, the head received NO gradient in that
    run while the encoder feeding it kept training on the MLM objective. The head
    is then frozen at whatever the previous stage left it, reading a
    representation that has since moved — so the binder term is a stale readout,
    not a validated classifier signal, and a gamma sweep against it cannot
    distinguish "steering does not help" from "the steerer is not a classifier".

    This is the shipped default for the chain's own stage 4
    (``configs/refine_antigen_hcdr3_infill.yaml`` sets
    ``compatibility_loss_weight: 0.0``, and ``mlm_train.py`` forces the same
    default for that stage), which is exactly the checkpoint the README's guided
    example passes to ``--checkpoint``. Refusing loudly follows the same
    fail-on-explicit rule as the ``--score-checkpoint`` guard below: the user
    explicitly asked for guidance, so a silently meaningless binder term is worse
    than an error.

    This checks the ONE thing that is mechanically knowable from the checkpoint.
    It does NOT establish that the head is a good classifier on the states
    guidance queries — guidance evaluates it on partially filled HCDR3 spans and
    on ~19 counterfactual residue substitutions per position, whereas stage 3
    trained it under ordinary ~10% corruption and stage 4 under ``full_span``
    masking. Training a head on that state distribution is what
    ``hcdr3_mask_mode: partial_span`` exists for; measuring whether gamma can move
    anything is what ``scripts/probe_steering_reachability.py`` is for. Neither is
    implied by passing this guard.

    Args:
        binder_cfg: Config of the checkpoint whose head supplies the binder term.
        checkpoint_path: That checkpoint's path, for the error message.
        is_external: Whether it came from ``--guidance-checkpoint``.
        allow_untrained: ``--allow-untrained-guidance-head`` escape hatch.

    Raises:
        SystemExit: If the head's loss weight was zero and no override was given.
    """
    if allow_untrained or binder_cfg.compatibility_loss_weight != 0:
        return
    which = (
        "--guidance-checkpoint" if is_external else "--checkpoint (its own head)"
    )
    raise SystemExit(
        f"refusing to guide with the compatibility head from {which}: "
        f"{checkpoint_path} was trained with compatibility_loss_weight = 0 "
        f"(training_stage = {binder_cfg.training_stage!r}), so that head got no "
        "gradient in the run that produced it while the representation feeding it "
        "kept training. Its binder term is a stale readout, not a validated "
        "classifier, and a gamma sweep against it cannot separate 'steering does "
        "not help' from 'the steerer is not a classifier'.\n"
        "Fix one of: pass --guidance-checkpoint pointing at a checkpoint whose "
        "compatibility head was actually trained (the stage-3 real-label run has "
        "compatibility_loss_weight: 1.0); or re-run the infill stage with a "
        "non-zero compatibility_loss_weight (ideally with "
        "hcdr3_mask_mode: partial_span, which trains the head on the partially "
        "filled states guidance queries); or pass "
        "--allow-untrained-guidance-head to proceed deliberately as a negative "
        "control."
    )


def assert_generated_any(rows: Sequence[dict[str, Any]], *, skipped_count: int) -> None:
    """
    Guard against a silent no-op generation run.

    Per-length ``ValueError``s (e.g. a proposed length overflowing ``max_length``)
    are skipped so one bad length does not abort the whole run. But if EVERY length
    was skipped and no candidate was produced, writing an empty JSONL and exiting 0
    makes a systematic data fault — wrong ``--data-path``, records missing antigen
    sequences, no valid HCDR3 spans — look like a successful run. Fail loudly
    instead so the no-op is not mistaken for success.
    """
    if not rows:
        raise SystemExit(
            f"no HCDR3 candidates were generated ({skipped_count} length(s) skipped). "
            "Check that --data-path points at antigen-annotated records with valid HCDR3 "
            "spans and antigen sequences, and that proposed lengths fit within max_length."
        )


def write_jsonl(rows: Sequence[dict[str, Any]], output_path: str | None) -> None:
    """
    Emit generated candidates as JSONL.

    When ``output_path`` is omitted, rows are printed to stdout so the command
    can be used in shell pipelines. When a path is provided, parent directories
    are created and rows are written there.
    """
    if output_path is None:
        for row in rows:
            print(json.dumps(row))
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.num_records <= 0:
        parser.error("--num-records must be > 0")
    if args.num_samples <= 0:
        parser.error("--num-samples must be > 0")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.top_k is not None and args.top_k < 0:
        parser.error("--top-k must be >= 0 (0 or omitted disables top-k filtering)")
    if args.guidance_strength < 0:
        parser.error("--guidance-strength must be >= 0 (0 disables guidance)")
    if args.guidance_checkpoint and args.guidance_strength == 0:
        # A guidance checkpoint at gamma == 0 never runs: gamma == 0 takes the
        # unguided branch and no classifier is consulted. Accepting and ignoring
        # it would poison a sweep -- rows would be labeled with a guidance
        # checkpoint that had no effect on a single sampled residue.
        parser.error(
            "--guidance-checkpoint requires --guidance-strength > 0; at gamma=0 no "
            "classifier is consulted, so a guidance checkpoint would be a silent no-op."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = choose_device(args.device)
    tokenizer = build_tokenizer()
    model, cfg = load_dual_stream_model(Path(args.checkpoint), data_path=args.data_path, device=device)

    guidance_model = None
    guidance_cfg = None
    if args.guidance_checkpoint:
        guidance_path = Path(args.guidance_checkpoint)
        if not guidance_path.exists():
            parser.error(f"--guidance-checkpoint path does not exist: {guidance_path}")
        # Loaded through the same strict dual-stream loader as the generation
        # model; its own config is threaded into the infiller (antigen tokenizer
        # + max_length) by build_infiller.
        guidance_model, guidance_cfg = load_dual_stream_model(
            guidance_path, data_path=args.data_path, device=device
        )

    if args.guidance_strength > 0:
        # Checked on whichever checkpoint actually supplies the binder term, so
        # routing a stale head in through --guidance-checkpoint is not a way
        # around the guard. Only reachable at gamma > 0: at gamma == 0 no
        # classifier is consulted, so the head's provenance is irrelevant.
        assert_guidance_head_was_trained(
            guidance_cfg if guidance_cfg is not None else cfg,
            checkpoint_path=Path(
                args.guidance_checkpoint if guidance_cfg is not None else args.checkpoint
            ),
            is_external=guidance_cfg is not None,
            allow_untrained=args.allow_untrained_guidance_head,
        )

    infiller = build_infiller(
        model,
        tokenizer,
        cfg,
        device,
        guidance_model=guidance_model,
        guidance_cfg=guidance_cfg,
    )

    scorer = None
    score_checkpoint_arg = args.score_checkpoint or None  # treat "" as not provided
    score_checkpoint = Path(score_checkpoint_arg) if score_checkpoint_arg else DEFAULT_SCORE_CHECKPOINT
    if not args.no_score:
        # An explicitly-provided --score-checkpoint that is missing is a user
        # error: fail loudly rather than silently emitting null scores. Only the
        # implicit DEFAULT_SCORE_CHECKPOINT is allowed to be absent (skip scoring).
        if score_checkpoint_arg is not None and not score_checkpoint.exists():
            parser.error(f"--score-checkpoint path does not exist: {score_checkpoint}")
        if score_checkpoint.exists():
            score_model, score_cfg = load_dual_stream_model(score_checkpoint, data_path=args.data_path, device=device)
            scorer = build_compatibility_scorer(score_model, tokenizer, score_cfg, device)
        else:
            # The implicit default is absent: skip scoring, but say so rather than
            # silently emitting null compatibility_score on every candidate.
            print(
                f"[info] default score checkpoint not found ({score_checkpoint}); "
                "skipping compatibility scoring (compatibility_score will be null).",
                file=sys.stderr,
            )

    target_dataset = OASSequenceDataset(args.data_path, split=args.split)
    target_records = select_records(target_dataset, record_id=args.record_id, num_records=args.num_records)

    length_prior = None
    if args.length_mode == "empirical":
        prior_dataset = OASSequenceDataset(args.data_path, split="train")
        length_prior = EmpiricalHCDR3LengthPrior.fit(prior_dataset.records, positive_only=True)
    elif args.length_mode == "learned":
        # The conditional posterior lives in the generation checkpoint, so its
        # length_head_max is the authority; passing the checkpoint's own value
        # means the two can never disagree.
        if getattr(model, "length_head", None) is None:
            parser.error(
                "--length-mode learned requires a checkpoint trained with "
                "length_loss_weight > 0 (no length_head in this checkpoint)"
            )
        length_prior = LearnedLengthProposal(
            infiller,
            length_head_max=cfg.length_head_max,
            mode=args.learned_length_mode,
        )

    # RE-SEED immediately before generation, after every model has been built.
    # `build_model` random-initializes a whole network, which advances the GLOBAL
    # torch RNG -- so attaching --score-checkpoint or --guidance-checkpoint shifted
    # the sampling stream and changed the generated sequences at a fixed --seed.
    # That directly contradicted this CLI's documented promise that
    # --score-checkpoint "never influences sampling", and made the paired
    # comparison "these exact candidates, now with a judge score" impossible.
    # Re-seeding here makes the draws a function of --seed alone.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    rows: list[dict[str, Any]] = []
    skipped_count = 0
    for record in target_records:
        true_span = HCDR3Span.from_record(record)
        if args.length_mode == "fixed":
            lengths = [true_span.length] * args.num_samples
        else:
            assert length_prior is not None
            # Guarded like the per-length body below. The learned proposal runs a
            # model forward, so it can raise for the same record-level reasons a
            # per-length generate can (a missing antigen sequence being the common
            # one). Unguarded, ONE such record aborted the whole run and wrote no
            # output, while --length-mode fixed/empirical skipped it with a warning
            # and still produced results for every usable record.
            try:
                lengths = length_prior.propose_lengths(
                    record, num_lengths=args.num_samples, rng=rng
                )
            except ValueError as exc:
                skipped_count += 1
                print(
                    f"[warn] skipping length proposal for {record.record_id}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not lengths:
                # A learned proposal can legitimately return nothing when no
                # length fits max_length for this scaffold. Count it as a skip so
                # assert_generated_any can distinguish "nothing was feasible" from
                # "the run silently produced no rows".
                skipped_count += 1
                print(
                    f"[warn] no feasible proposed length for {record.record_id}",
                    file=sys.stderr,
                )
                continue

        for proposed_length in lengths:
            try:
                if args.guidance_strength > 0:
                    # Opt-in guided path: iterative, binder-steered decoding. The
                    # binder term comes from the generation model's own
                    # compatibility head unless --guidance-checkpoint attached an
                    # external classifier.
                    candidates = infiller.guided_infill(
                        record,
                        length=proposed_length,
                        num_samples=1,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        guidance_strength=args.guidance_strength,
                        guidance_target=args.guidance_target,
                        order=args.guidance_order,
                        scorer=scorer,
                        rng=rng,
                    )
                else:
                    # Default path: unchanged single-pass independent sampling.
                    candidates = infiller.infill(
                        record,
                        length=proposed_length,
                        num_samples=1,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        scorer=scorer,
                    )
            except ValueError as exc:
                # e.g. a proposed length that overflows max_length; skip this
                # length rather than aborting the whole generation run. A run that
                # skips *every* length is caught below so it is not a silent no-op.
                skipped_count += 1
                print(f"[warn] skipping length {proposed_length} for {record.record_id}: {exc}", file=sys.stderr)
                continue
            for candidate in candidates:
                rows.append(
                    candidate_to_json(
                        record=record,
                        true_span=true_span,
                        length_mode=args.length_mode,
                        candidate=candidate,
                        guidance_strength=args.guidance_strength,
                        guidance_order=args.guidance_order,
                        guidance_checkpoint=args.guidance_checkpoint or None,
                    )
                )

    assert_generated_any(rows, skipped_count=skipped_count)
    write_jsonl(rows, args.output_path)


if __name__ == "__main__":
    main()
