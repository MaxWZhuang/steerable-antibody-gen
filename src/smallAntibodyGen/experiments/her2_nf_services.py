"""The concrete runtime: data, models, banks, pilots, production, audits and the smoke path.

Everything in this module touches the GPU or the dataset. It is separated from
:mod:`her2_nf_campaign` so the orchestration -- stage order, gates, statuses,
the supervisor -- is testable against a stub while this layer is exercised by
the native opt-in tests and by the miniature end-to-end smoke run.

Three conventions are enforced here rather than left to each call site:

* **configured artifact roots only.** Every historical input is resolved through
  ``context.artifact_root``. Bounded evidence copies that live beside a review
  loop are inputs to a reader and to tests; production never reads them, because
  a number attributed to bytes nobody verified is not evidence.
* **C selects, E is scored last.** The challenge's monitoring, checkpoint
  selection and comparator regularization all read ``C``. ``E`` is scored only
  after the challenge choices and endpoints are frozen, and the loaders carry a
  ``ForbiddenRows`` guard that refuses an ``E`` row inside a challenge training
  population.
* **measure, then forecast.** :meth:`FlightServices.profile` runs real work and
  records seconds per unit. No method here writes a cost it did not measure.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from . import her2_data as data_lib
from . import her2_nf_banks as nf_banks
from . import her2_nf_calibration as calibration_lib
from . import her2_nf_contract as contract
from .her2_nf_cost import BudgetExhausted, WorkBudget
from . import her2_nf_coupling as coupling
from . import her2_nf_metrics as metrics
from . import her2_nf_mixture as mixture_lib
from . import her2_nf_monitor as monitor_lib
from . import her2_nf_objectives as nf_objectives
from . import her2_nf_proximity as proximity
from . import her2_nf_reuse as nf_reuse
from . import her2_nf_spec as spec
from . import her2_nf_storage as storage
from . import her2_nf_trajectory as trajectory_lib
from . import her2_preferences as preferences
from . import her2_replay as replay_lib
from . import her2_replay_streams as streams_lib
from . import her2_support_paths as paths
from .her2_runtime import require


class FlightServices:
    """Concrete runtime for one flight context."""

    def __init__(self, context, *, device="cuda"):
        self.context = context
        self.config = context.config
        self.device = device
        self._cache = {}
        self.streams = nf_banks.StreamRegistry()
        stream_path = self.context.path("stream_registry.json")
        if stream_path.is_file():
            saved = paths.read_json(stream_path)
            for name, lineage in saved["streams"].items():
                restored = self.streams.stream(**lineage["descriptor"])
                require(restored == lineage and name in self.streams.entries,
                        f"stream provenance changed for {name}")
        self.reuse = {}

    def stream_seed(self, **descriptor):
        """One spawned, domain-separated, collision-checked stream."""
        lineage = self.streams.stream(**descriptor)
        paths.write_json(self.context.path("stream_registry.json"), self.streams.document())
        return lineage

    def release_transient(self):
        """Keep input tables, but release parent-specific banks between model jobs."""
        transient = ("replay:", "development:", "dev_bank:", "reference:", "pair_reference:", "loaded:")
        for key in list(self._cache):
            if key.startswith(transient):
                del self._cache[key]
        storage.collect_unused()

    # -- inputs ----------------------------------------------------------
    def raw_root(self):
        return self.context.artifact_root("raw")

    def historical_root(self):
        return self.context.artifact_root("historical")

    def split(self, name):
        require(name in ("train", "val"),
                f"{name!r} is not readable here. The reserved test LABELS are never opened; only "
                "her2_data.test_sequences (usecols=['seq']) is available, for membership "
                "diagnostics alone.")
        key = f"split:{name}"
        if key not in self._cache:
            self._cache[key] = data_lib.load_split(self.raw_root(), name)
        return self._cache[key]

    def index(self, name):
        key = f"index:{name}"
        if key not in self._cache:
            self._cache[key] = data_lib.encode_cores(self.split(name).seq)
        return self._cache[key]

    def scaffold(self):
        if "scaffold" not in self._cache:
            self._cache["scaffold"] = data_lib.load_scaffold(self.raw_root())
        return self._cache["scaffold"]

    def vocab(self):
        from . import her2_policy
        if "vocab" not in self._cache:
            # The RAW ROOT, not raw_root/"piggen": every inherited loader appends
            # PIGGEN_DIR itself, so passing the joined path asks for
            # .../piggen/piggen/tokenizer.json and fails on the real tree.
            self._cache["vocab"] = her2_policy.load_vocab(self.raw_root())
        return self._cache["vocab"]

    # -- models ----------------------------------------------------------
    def policy(self, checkpoint=None, *, architecture_only=False):
        """A ``CorePolicy`` on the pinned architecture, optionally at a checkpoint.

        ``checkpoint`` goes through :func:`her2_nf_reuse.load_checkpoint`, which
        accepts this flight's own ``her2-core-policy/1`` endpoints AND the
        completed campaign's ``her2-parent-replay-endpoint/1`` ones, strict-loads
        either, and re-derives the recorded state digest. Strictness is not
        relaxed and no parameter is coerced.
        """
        from . import her2_nf_reuse as reuse
        from . import her2_policy
        root = self.raw_root()
        model = (her2_policy.architecture_model(root, device=self.device) if architecture_only or checkpoint is not None
                 else her2_policy.load_pinned_model(root, device=self.device))
        if checkpoint is not None:
            self._cache[f"loaded:{checkpoint}"] = reuse.load_checkpoint(
                Path(checkpoint), model, device=self.device)
        return her2_policy.CorePolicy.from_prefix(model, self.scaffold().prefix, self.vocab(),
                                                  device=self.device)

    def parent_checkpoint(self, seed):
        entry = dict(self.config["block_a"]["parents"])[str(seed)]
        return self.context.repository_root / entry["logical"], entry

    def optimizer_and_scheduler(self, policy, optimization=None):
        import torch
        optimization = dict(optimization or self.config["optimization"])
        optimizer = torch.optim.AdamW(policy.model.parameters(),
                                      lr=float(optimization["learning_rate"]),
                                      betas=tuple(optimization["betas"]),
                                      weight_decay=float(optimization["weight_decay"]))
        warmup = int(optimization["warmup_updates"])
        # ``step + 1`` is the inherited convention: LambdaLR evaluates its lambda
        # at last_epoch=0 during construction, so without it the FIRST optimizer
        # step would run at learning rate zero.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0))
        return optimizer, scheduler

    # -- M0/M1: the probability contract ---------------------------------
    def probability_contract(self, *, seed=None, rows=256):
        seed = int(seed if seed is not None else self.config["block_a"]["parent_seeds"][0])
        checkpoint, entry = self.parent_checkpoint(seed)
        policy = self.policy(checkpoint)
        index = self.index("val")[:int(rows)]
        lineage = self.stream_seed(domain="contract_probe", role="probe", parent=f"seed{seed}")
        block = contract.verify_probability_contract(
            policy, index, draws=int(self.config["probability"]["parity_draws"]),
            seed=lineage["seed"], batch_size=int(self.config["inference"]["score_batch_size"]))
        block["prompt"] = contract.verify_prompt_shape(self.scaffold())
        block["parent"] = {"seed": seed, "logical": entry["logical"],
                           "expected_state_sha256": entry.get("state_sha256"),
                           "observed_state_sha256": _state_digest(policy)}
        if entry.get("state_sha256"):
            require(block["parent"]["observed_state_sha256"] == entry["state_sha256"],
                    "the loaded parent does not reproduce the state digest the config records for "
                    "it; every later margin would be computed against a different reference")
        return block

    # -- M2: historical stream and kernel parity -------------------------
    def verify_historical_streams(self):
        """LOAD each saved task stream, then re-derive it and compare the digests.

        Two different statements, and only one of them was being made. Loading
        the saved arrays is the reuse: those are the rows the completed campaign
        trained on, including the saved REPLAY ORDER, which was not being read at
        all. Re-deriving them and comparing digests is a check on the derivation
        rule, which is worth having and is not a substitute.
        """
        from . import her2_nf_reuse as reuse
        block_a = self.config["block_a"]
        frame = self.split("train")
        population = preferences.build_population(frame, "train")
        batch_rows = int(block_a["chosen_per_update"])
        results, matches = {}, []
        for seed in block_a["parent_seeds"]:
            if self._historical_stream_document(seed) is None:
                results[str(seed)] = {"available": False, "loaded": False,
                                      "reason": "the historical stream record is not readable "
                                                "from the configured root"}
                matches.append(False)
                continue
            saved = reuse.load_task_stream(self.historical_root(), seed, batch_rows=batch_rows)
            identity = dict(saved.record.get("identity") or {})
            pairing = preferences.PreferencePairing(
                population, seed=int(identity["pairing_seed"]))
            derived = streams_lib.resolve_task_stream(
                pairing, seed=int(seed), exposures=int(identity["exposures"]),
                batch_rows=int(identity["batch_rows"]),
                pairing_seed=int(identity["pairing_seed"])).document()
            block = reuse.compare_stream(saved, derived)
            matches.append(block["all_match"])
            results[str(seed)] = {
                "available": True, "loaded": True, "matches": block["all_match"],
                "comparison": block["comparison"],
                "loaded_exposures": saved.exposures,
                "replay_order_sha256": block["replay_order_sha256"],
                "replay_order_recorded":
                    (saved.record.get("replay_order") or {}).get("order_sha256"),
                "replay_order_reused": True}
        return {"schema_version": contract.NF_SCHEMA, "record_kind": "historical_stream_check",
                "per_seed": results, "all_match": bool(matches and all(matches)),
                "consequence": ("a mismatch voids the reuse of the historical trajectories: they "
                                "become external references and the affected paths are retrained. "
                                "Their early checkpoints remain MISSING and are not reconstructed."),
                "replay_order_note": ("the saved replay order is LOADED from the same archive and "
                                      "is the one the training loop consumes. It is not "
                                      "re-derived and assumed equal.")}

    def _historical_stream_document(self, seed):
        logical = self.config["historical"]["stream_record"].format(seed=int(seed))
        target = self.historical_root() / logical
        if not target.is_file():
            return None
        return paths.read_json(target)

    #: The two historical kernels this flight reuses. BOTH are checked: agreeing
    #: on the no-preservation arm says nothing about the replay term, its frozen
    #: teacher cache or its saved replay order, and the FKL arm is half of the
    #: reused Block-A cells.
    PARITY_ARMS = (
        {"name": "ipo_lambda0", "trajectory": "trajectories/ipo_lambda0_seed{seed}",
         "preservation": "none", "lambda": 0.0},
        {"name": "ipo_lambda10", "trajectory": "trajectories/ipo_lambda10_seed{seed}",
         "preservation": "fkl", "lambda": 10.0},
    )

    def verify_update_kernel_parity(self, *, updates=None, atol=None, rtol=None):
        """Run BOTH new kernels on the historical state, cache, stream and bank.

        The comparison is against the historical per-update task losses AND the
        saved update-25 monitor score vectors, at a declared tolerance, on
        identical state and data. There are no historical update-25 *weights* to
        hash against, so no hash comparison is attempted and none is
        approximated.
        """
        settings = dict(self.config["reuse_parity"])
        updates = int(updates if updates is not None else settings["updates"])
        atol = float(atol if atol is not None else settings["task_loss_atol"])
        rtol = float(rtol if rtol is not None else settings["task_loss_rtol"])
        seed = int(settings["seed"])
        arms, all_passed, available = {}, [], []
        for arm in self.PARITY_ARMS:
            trajectory = arm["trajectory"].format(seed=seed)
            journal = self.historical_root() / trajectory / "updates.jsonl"
            if not journal.is_file():
                arms[arm["name"]] = {
                    "available": False, "passed": False,
                    "reason": (f"{trajectory}/updates.jsonl is not readable from the configured "
                               "root; reuse of this arm cannot be established and it is retrained")}
                available.append(False)
                all_passed.append(False)
                continue
            recorded = _read_journal_head(journal, updates)
            outcome = self._continuation(
                seed=seed, updates=updates, coefficients={"tau": float(settings["tau"])},
                preservation=arm["preservation"], preservation_lambda=float(arm["lambda"]))
            rows, worst = [], 0.0
            for observed, expected in zip(outcome["task_losses"], recorded):
                difference = abs(float(observed) - float(expected["task_loss"]))
                worst = max(worst, difference)
                rows.append({"update": int(expected["update"]),
                             "historical": expected["task_loss"],
                             "recomputed": float(observed), "absolute_difference": difference,
                             "within_tolerance": difference <= atol + rtol * abs(
                                 float(expected["task_loss"]))})
            monitor = self._monitor_vector_parity(trajectory, outcome["policy"], update=updates,
                                                  atol=atol, rtol=rtol)
            passed = (bool(rows) and all(row["within_tolerance"] for row in rows)
                      and bool(monitor["passed"]))
            arms[arm["name"]] = {
                "available": True, "passed": passed, "updates": len(rows),
                "max_absolute_difference": worst, "rows": rows,
                "monitor_vector": monitor,
                "sources": {"stream": outcome["stream_source"],
                            "reference_cache": outcome["reference_source"],
                            "replay_bank": outcome["replay_source"]},
                "preservation": arm["preservation"], "lambda": arm["lambda"]}
            available.append(True)
            all_passed.append(passed)
        return {"schema_version": contract.NF_SCHEMA, "record_kind": "update_kernel_parity",
                "available": all(available), "passed": bool(available) and all(all_passed),
                "arms": arms, "atol": atol, "rtol": rtol, "updates": updates,
                "coverage": ("both reused kernels: IPO without preservation and IPO + forward-KL "
                             "at lambda 10. Per-update task scalars and the saved update-25 "
                             "monitor score vector are both compared."),
                "basis": ("identical parent state, the SAVED frozen reference cache, the SAVED "
                          "row order and the SAVED replay bank with its frozen teacher cache. "
                          "No checkpoint hash is compared approximately and no missing update-25 "
                          "weights are required."),
                "measurement_note": ("native GPU gradients differ between kernels by float32 "
                                     "reduction order -- a measured ~2e-5 maximum on this "
                                     "architecture. That is a numeric observation, not evidence "
                                     "of bitwise identity, and the exact identity is tested "
                                     "separately in CPU float64.")}

    def _monitor_vector_parity(self, trajectory, policy, *, update, atol, rtol):
        """Compare the saved update-25 monitor score vector with this kernel's."""
        from . import her2_nf_reuse as reuse
        saved = reuse.load_monitor_scores(self.historical_root(), trajectory, update=update)
        if saved is None:
            # The declared gate compares the SAVED vector. An absent one is an
            # uncompared requirement, never a passed check: the arm then becomes
            # an external reference and is retrained, which is exactly what the
            # reuse rule prescribes when parity cannot be established.
            return {"available": False, "passed": False,
                    "reason": (f"no saved monitor vector for {trajectory} at update "
                               f"{int(update)}; the required score-vector comparison "
                               "did not run")}
        arrays = saved["arrays"]
        name = next((key for key in ("chosen_scores", "scores", "chosen")
                     if key in arrays), None)
        if name is None:
            return {"available": True, "passed": False,
                    "reason": f"unrecognized monitor arrays {sorted(arrays)}"}
        recorded = np.asarray(arrays[name], dtype=np.float64)
        pairs = self.validation_pairs()
        rows = min(int(recorded.size), int(pairs["chosen_index"].shape[0]))
        observed = np.asarray(preferences.score_sequences(
            policy, pairs["chosen_index"][:rows],
            batch_size=int(self.config["inference"]["score_batch_size"]), progress_every=0),
            dtype=np.float64)
        difference = np.abs(observed - recorded[:rows])
        allowance = atol + rtol * np.abs(recorded[:rows])
        return {"available": True, "passed": bool((difference <= allowance).all()),
                "rows": int(rows), "array": name, "path": saved["path"],
                "max_absolute_difference": float(difference.max()),
                "basis": ("the saved monitor score vector at this update, compared row for row "
                          "against the same population under the new kernel")}

    def _continuation(self, *, seed, updates, coefficients, preservation="none",
                      preservation_lambda=0.0):
        """A short continuation from the historical parent on the SAVED stream and cache.

        Identical initial state, identical frozen reference cache, identical row
        order and -- for the replay arm -- the identical saved replay bank,
        frozen teacher cache and saved replay order. Re-deriving any of those
        would make this a comparison against a reconstruction rather than
        against what the completed campaign actually ran.
        """
        checkpoint, entry = self.parent_checkpoint(seed)
        policy = self.policy(checkpoint)
        optimizer, scheduler = self.optimizer_and_scheduler(policy)
        frame = self.split("train")
        population = preferences.build_population(frame, "train")
        batch_rows = int(self.config["block_a"]["chosen_per_update"])
        micro = int(self.config["block_a"]["microbatch_rows"])
        pairing_seed = int(self.config["block_a"]["pairing"]["seed_base"]) + int(seed)
        stream, order, stream_source = self._streams_for(
            block="A", regime="original_split", seed=seed, population=population,
            pairing_seed=pairing_seed, batch_rows=batch_rows, horizon=int(updates),
            replay_rows=int(self.config["banks"]["replay_rows"]))
        reference = self._reference_cache(policy, population, seed=seed,
                                          parent_file_sha256=paths.sha256_file(checkpoint))
        preservation_batch = None
        replay_bank = None
        if preservation != "none":
            parent_policy = self.policy(checkpoint)
            replay_bank = self._replay_bank(parent_policy, seed, "original_split",
                                            teacher_rows=int(batch_rows * updates))
            preservation_batch = self._preservation_batch(policy, replay_bank, preservation)
        losses = []
        policy.model.train()
        for update in range(1, int(updates) + 1):
            chosen_rows, rejected_rows = stream.batch(update)
            totals = {"task": batch_rows}
            factors = {"task": 1.0}
            if preservation != "none":
                totals[preservation] = batch_rows
                factors[preservation] = float(preservation_lambda)
            accumulator = replay_lib.MicrobatchAccumulator(
                totals, coefficients=factors, label=f"parity update {update}")
            replay_rows = (streams_lib.replay_batch_rows(order, update, batch_rows=batch_rows)
                           if preservation != "none" else None)
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, batch_rows, micro):
                stop = min(start + micro, batch_rows)
                counts, means = {}, {}
                mean, _ = self._task_microbatch(
                    policy, population, reference, task="ipo", coefficients=coefficients,
                    chosen_rows=chosen_rows[start:stop], rejected_rows=rejected_rows[start:stop])
                counts["task"], means["task"] = stop - start, mean
                if preservation != "none":
                    keep, _ = preservation_batch(replay_rows[start:stop])
                    counts[preservation], means[preservation] = stop - start, keep
                accumulator.add(counts, means, backward=lambda tensor: tensor.backward())
            components = accumulator.finish()
            replay_lib.clip_and_step(policy.model, optimizer, scheduler,
                                     gradient_clip=float(self.config["optimization"]
                                                         ["gradient_clip"]))
            losses.append(components["component_means"]["task"])
        return {"task_losses": losses, "state_sha256": _state_digest(policy),
                "policy": policy, "stream_source": stream_source,
                "reference_source": reference.get("source"),
                "replay_source": (replay_bank or {}).get("source")}

    def _reference_cache(self, policy, population, *, key=None, seed=None,
                         parent_file_sha256=None):
        """Frozen parent scores for the chosen and rejected populations.

        For a Block-A parent the completed campaign's **saved** cache is loaded
        and verified against this flight's population, prompt and parent. That is
        the point: an arm that reuses a historical trajectory has to consume the
        same frozen reference the old one did, and rescoring 234k rows to a
        number that happens to agree is a different artifact with the same value.
        Only when no saved cache exists is one computed, and the record says
        which happened.
        """
        cache_key = f"reference:{key}" if key is not None else None
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]
        block = None
        if seed is not None:
            block = self._saved_reference_cache(population, seed,
                                                parent_file_sha256=parent_file_sha256)
        if block is None:
            batch = int(self.config["inference"]["score_batch_size"])
            chosen = preferences.score_sequences(policy, population.chosen_index,
                                                 batch_size=batch, progress_every=0)
            rejected = preferences.score_sequences(policy, population.rejected_index,
                                                   batch_size=batch, progress_every=0)
            block = {"chosen": np.asarray(chosen, dtype=np.float64),
                     "rejected": np.asarray(rejected, dtype=np.float64),
                     "source": "computed",
                     "identity": preferences.reference_identity(
                         checkpoint_sha256=_state_digest(policy), config_sha256="nf",
                         scaffold_prefix=self.scaffold().prefix,
                         index=np.concatenate([population.chosen_index,
                                               population.rejected_index]))}
        if cache_key is not None:
            self._cache[cache_key] = block
        return block

    def _saved_reference_cache(self, population, seed, *, parent_file_sha256=None):
        from . import her2_nf_reuse as reuse
        try:
            block = reuse.load_reference_cache(
                self.historical_root(), seed, population=population,
                scaffold_prefix=self.scaffold().prefix,
                parent_checkpoint_sha256=parent_file_sha256)
        except ValueError as error:
            # A cache that EXISTS and disagrees is a hard failure -- that is the
            # wrong-parent/wrong-population case. A cache that is simply not
            # readable from here is a reuse that did not happen, and the record
            # says so rather than claiming one.
            target = (self.historical_root() / "banks" / f"seed{int(seed)}"
                      / "ipo_reference_cache.npy")
            if target.is_file():
                raise
            self.reuse.setdefault("reference_cache", {})[str(seed)] = {
                "reused": False, "reason": f"{type(error).__name__}: {error}"}
            return None
        self.reuse.setdefault("reference_cache", {})[str(seed)] = {
            "reused": True, "rows": block["rows"], "path": block["path"],
            "identity": block["identity"]}
        return block

    def validation_pairs(self):
        """The fixed validation pair set the Block-A gate is defined on.

        Built from the VALIDATION population, not the training one. Pointing the
        gate at training pairs would silently redefine D on a population the gate
        was never calibrated against -- and it would still return a plausible
        number at every check.
        """
        if "validation_pairs" not in self._cache:
            population = preferences.build_population(self.split("val"), "val")
            pairing = preferences.PreferencePairing(
                population, seed=int(self.config["block_a"]["pairing"]["validation_seed"]))
            self._cache["validation_pairs"] = pairing.fixed_validation_pairs(
                count=self.config["block_a"]["pairing"].get("validation_pairs"))
        return self._cache["validation_pairs"]

    def original_strata(self):
        """Validation rows' distance to the ORIGINAL training set. Computed once."""
        if "original_strata" not in self._cache:
            self._cache["original_strata"] = proximity.original_proximity_strata(
                self.index("val"), self.index("train"))
        return self._cache["original_strata"]

    def _task_microbatch(self, policy, population, reference, *, task, coefficients, chosen_rows,
                         rejected_rows):
        import torch
        device = policy.device
        chosen = policy.sequence_log_probs(population.chosen_index[chosen_rows])
        spec_entry = nf_objectives.TASK_OBJECTIVES[task]
        rejected = (policy.sequence_log_probs(population.rejected_index[rejected_rows])
                    if spec_entry.uses_rejected else None)
        reference_chosen = reference_rejected = None
        if spec_entry.uses_reference:
            reference_chosen = torch.as_tensor(reference["chosen"][chosen_rows],
                                               device=device).to(chosen.dtype)
            reference_rejected = torch.as_tensor(reference["rejected"][rejected_rows],
                                                 device=device).to(chosen.dtype)
        return nf_objectives.task_term(task, policy_chosen=chosen, policy_rejected=rejected,
                                       reference_chosen=reference_chosen,
                                       reference_rejected=reference_rejected,
                                       coefficients=coefficients)

    # -- M3: geometry ----------------------------------------------------
    def build_geometry(self):
        """Try the declared panel sizes largest-first at r=2, then the named r=1 fallback."""
        train_frame, val_frame = self.split("train"), self.split("val")
        train_index, val_index = self.index("train"), self.index("val")
        audit = contract.audit_cores({"train": train_frame, "val": val_frame},
                                     label="challenge construction")
        contract.require_no_label_conflicts(audit)
        attempts = []
        # The ORIGINAL proximity labels are computed once, at the primary radius,
        # and reused for both the primary construction and the radius-1 fallback.
        # Recomputing them at radius 1 collapses the distance-2 rows into ">=3"
        # and loses the bin entirely -- the labels describe distance to the
        # ORIGINAL training set and must not move with the purge radius.
        strata = proximity.original_proximity_strata(val_index, train_index,
                                                     radius=proximity.PRIMARY_RADIUS)
        for radius in (proximity.PRIMARY_RADIUS, proximity.FALLBACK_RADIUS):
            for panel_size in proximity.PANEL_SIZES:
                evaluation = proximity.build_evaluation_panel(val_frame, strata["labels"],
                                                              panel_size=panel_size)
                calibration = proximity.build_calibration_panel(
                    val_frame, val_index, evaluation["rows"], radius=radius,
                    cap=int(self.config["split"]["calibration_rows"]))
                populations = proximity.build_populations(
                    train_frame, train_index,
                    evaluation_index=val_index[evaluation["rows"]],
                    calibration_index=val_index[calibration["rows"]], radius=radius)
                panel_strata = {name: int((strata["labels"][evaluation["rows"]] == name).sum())
                                for name in proximity.ORIGINAL_STRATA}
                feasibility = proximity.feasibility_report(
                    train_frame=train_frame, train_index=train_index, populations=populations,
                    evaluation_rows=evaluation["rows"], calibration_rows=calibration["rows"],
                    val_frame=val_frame, val_index=val_index, radius=radius,
                    strata_counts=panel_strata)
                attempts.append({"radius": radius, "panel_size": panel_size,
                                 "passed": feasibility["passed"], "checks": feasibility["checks"]})
                if feasibility["passed"]:
                    certificate = proximity.neighbour_certificate(
                        train_frame=train_frame, train_index=train_index, val_frame=val_frame,
                        val_index=val_index, populations=populations,
                        evaluation_rows=evaluation["rows"],
                        calibration_rows=calibration["rows"], radius=radius,
                        source_digests=self._source_digests())
                    manifest = proximity.split_manifest(
                        train_frame=train_frame, val_frame=val_frame, train_index=train_index,
                        val_index=val_index, evaluation_rows=evaluation["rows"],
                        calibration_rows=calibration["rows"], populations=populations,
                        strata=strata, radius=radius, panel_size=panel_size,
                        source_digests=self._source_digests())
                    self._persist_populations(populations, evaluation, calibration, radius)
                    paths.write_json(self.context.path("split_manifest.json"), _jsonable(manifest))
                    paths.write_json(self.context.path("neighbor_certificate.json"), _jsonable(certificate))
                    return {"radius": radius, "panel_size": panel_size, "attempts": attempts,
                            "feasibility": feasibility, "certificate": certificate,
                            "manifest": manifest, "core_audit": audit,
                            "named_challenge": ("radius 2" if radius == proximity.PRIMARY_RADIUS
                                                else "radius 1 -- a DIFFERENT challenge, named "
                                                     "as the predeclared fallback")}
        return {"radius": proximity.PRIMARY_RADIUS, "panel_size": None, "attempts": attempts,
                "feasibility": {"passed": False, "radius": proximity.PRIMARY_RADIUS,
                                "checks": attempts[-1]["checks"] if attempts else {},
                                "counts": {}, "criteria": proximity.FEASIBILITY},
                "certificate": {"zero_violations": False},
                "manifest": {"record_kind": "split_manifest", "status": "not built"},
                "core_audit": audit}

    def _persist_populations(self, populations, evaluation, calibration, radius):
        paths.write_arrays(self.context.path("split", f"populations_r{radius}.npz"), {
            "t0_rows": np.asarray(populations["t0_rows"]),
            "purge_rows": np.asarray(populations["purge_rows"]),
            "match_rows": np.asarray(populations["match_rows"]),
            "evaluation_rows": np.asarray(evaluation["rows"]),
            "calibration_rows": np.asarray(calibration["rows"])})

    def _source_digests(self):
        digests = {}
        for name in ("train", "val"):
            target = data_lib.split_path(self.raw_root(), name)
            digests[name] = paths.sha256_file(target)
        return digests

    def load_populations(self, *, radius=None):
        manifest = paths.read_json(self.context.path("split_manifest.json"))
        radius = int(radius if radius is not None else manifest["radius"])
        arrays = paths.read_arrays(self.context.path("split", f"populations_r{radius}.npz"))
        return {"manifest": manifest, **{key: np.asarray(value)
                                         for key, value in arrays.items()}}

    def forbidden_evaluation_rows(self):
        """The ``E`` guard the challenge loaders carry."""
        block = self.load_populations()
        val_frame = self.split("val")
        cores = [val_frame.seq.iloc[int(row)] for row in block["evaluation_rows"]]
        return contract.ForbiddenRows.from_cores(
            cores, label="evaluation_panel_E",
            reason=("E never enters challenge training or challenge selection. It is scored only "
                    "after the challenge choices and endpoints are frozen."))

    # -- M3: the cheap mixture rescore -----------------------------------
    def mixture_rescore(self):
        """Score parent and policy on the old validation split and persist the whole alpha grid."""
        val_frame = self.split("val")
        val_index = self.index("val")
        positive = (val_frame["class"] == "high").to_numpy()
        curves = []
        for entry in self.config["mixture"]["pairs"]:
            parent_path, _ = self.parent_checkpoint(entry["seed"])
            policy_path = self.historical_root() / entry["policy_logical"]
            if not policy_path.is_file():
                curves.append({"label": entry["label"], "available": False,
                               "reason": "the named policy checkpoint is not readable from the "
                                         "configured root; no mixture is reported for it"})
                continue
            batch = int(self.config["inference"]["score_batch_size"])
            parent_scores = np.asarray(self.policy(parent_path).score(
                val_index, batch_size=batch)["sum_log_probability"], dtype=np.float64)
            policy_scores = np.asarray(self.policy(policy_path).score(
                val_index, batch_size=batch)["sum_log_probability"], dtype=np.float64)
            # The row-indexed vectors, persisted: every mixture number in the
            # report is recomputable from these without rescoring anything, and
            # a paired contrast needs the same rows on both sides.
            target = self.context.path("row_scores", f"mixture_{_slug(entry['label'])}.npz")
            paths.write_arrays(target, {
                "parent_sum_log_probability": parent_scores,
                "policy_sum_log_probability": policy_scores,
                "mixture_sum_log_probability_primary": mixture_lib.mixture_log_probability(
                    parent_scores, policy_scores, mixture_lib.PRIMARY_ALPHA),
                "positive": positive.astype(np.int8)})
            curve = mixture_lib.alpha_curve(
                parent_scores, policy_scores, labels=positive,
                yield_budgets=metrics.YIELD_BUDGETS,
                yield_function=metrics.expected_distinct_yield)
            parent_yield = metrics.expected_distinct_yield(parent_scores[positive], 10_000)
            policy_yield = metrics.expected_distinct_yield(policy_scores[positive], 10_000)
            curve.update(
                label=entry["label"], available=True, seed=int(entry["seed"]),
                policy_logical=entry["policy_logical"],
                parent_yield_10k=parent_yield, policy_yield_10k=policy_yield,
                concavity_bound_at_primary=metrics.mixture_yield_bound(
                    parent_yield, policy_yield, mixture_lib.PRIMARY_ALPHA),
                panel=("distinct high identities of the ORIGINAL validation split. Absolute "
                       "counts are not comparable with the challenge panel's."))
            curves.append(curve)
        return {"schema_version": contract.NF_SCHEMA, "record_kind": "mixture_rescore",
                "curves": curves, "grid": list(mixture_lib.ALPHA_GRID),
                "cost_note": ("generation runs ONE component per draw; exact scoring runs both "
                              "models. Two checkpoints are stored."),
                "status": ("scores recomputed by this flight from the named checkpoints. No "
                           "pasted table is reported as a measurement here.")}

    # -- M4: profiling ---------------------------------------------------
    def profile(self):
        """Measure every declared cost category on this box. Nothing is estimated."""
        from . import her2_nf_campaign as campaign
        settings = dict(self.config["profile"])
        import torch
        sync = torch.cuda.synchronize if self.device == "cuda" else lambda: None
        profile = campaign.Profile()
        stage_started = time.perf_counter()
        seed = int(self.config["block_a"]["parent_seeds"][0])
        checkpoint, _ = self.parent_checkpoint(seed)
        policy = self.policy(checkpoint)
        optimizer, scheduler = self.optimizer_and_scheduler(policy)
        frame = self.split("train")
        population = preferences.build_population(frame, "train")
        reference = self._reference_cache(policy, population, key=f"A:original_split:{seed}",
                                          seed=seed,
                                          parent_file_sha256=paths.sha256_file(checkpoint))
        batch_rows = int(self.config["block_a"]["chosen_per_update"])
        micro = int(self.config["block_a"]["microbatch_rows"])
        pairing = preferences.PreferencePairing(
            population, seed=int(self.config["block_a"]["pairing"]["seed_base"]) + seed)
        updates = int(settings["optimizer_updates"])
        stream = streams_lib.resolve_task_stream(
            pairing, seed=seed, exposures=batch_rows * updates, batch_rows=batch_rows,
            pairing_seed=int(self.config["block_a"]["pairing"]["seed_base"]) + seed)
        bank = self._replay_bank(policy, seed, "profile",
                                 rows=int(settings["replay_rows"]),
                                 teacher_rows=int(settings["replay_rows"]))
        order = streams_lib.replay_order(bank_rows=bank["index"].shape[0],
                                         exposures=batch_rows * updates,
                                         seed=int(self.config["seeds"]["replay_order"]),
                                         parent_seed=seed)
        # One measurement per loss family, because their per-update costs differ:
        # a replay arm pays for an extra forward pass and a cache read that a
        # no-preservation control does not. The forecast uses the SLOWEST family,
        # which is the conservative direction for a schedule.
        per_family = {}
        for family, task, preservation, coefficients in (
                ("ipo_none", "ipo", "none", {"tau": 0.1}),
                ("ipo_fkl", "ipo", "fkl", {"tau": 0.1, "lambda": 10.0}),
                ("ipo_tail", "ipo", "tail", {"tau": 0.1, "lambda": 0.1}),
                ("dpo_none", "dpo", "none", {"beta": 0.5}),
                ("dpo_fkl", "dpo", "fkl", {"beta": 0.5, "lambda": 10.0}),
                ("dpo_tail", "dpo", "tail", {"beta": 0.5, "lambda": 0.1})):
            del optimizer, scheduler, policy
            storage.collect_unused()
            policy = self.policy(checkpoint)
            optimizer, scheduler = self.optimizer_and_scheduler(policy)
            policy.model.train()
            preservation_batch = self._preservation_batch(policy, bank, preservation)
            sync()
            started = time.perf_counter()
            for update in range(1, updates + 1):
                chosen_rows, rejected_rows = stream.batch(update)
                optimizer.zero_grad(set_to_none=True)
                totals = {"task": batch_rows}
                factors = {"task": 1.0}
                if preservation != "none":
                    totals[preservation] = batch_rows
                    factors[preservation] = float(coefficients["lambda"])
                accumulator = replay_lib.MicrobatchAccumulator(
                    totals, coefficients=factors, label=f"profile {family} update {update}")
                replay_rows = streams_lib.replay_batch_rows(order, update, batch_rows=batch_rows)
                for start in range(0, batch_rows, micro):
                    stop = min(start + micro, batch_rows)
                    counts, means = {}, {}
                    mean, _ = self._task_microbatch(
                        policy, population, reference, task=task, coefficients=coefficients,
                        chosen_rows=chosen_rows[start:stop],
                        rejected_rows=rejected_rows[start:stop])
                    counts["task"], means["task"] = stop - start, mean
                    if preservation != "none":
                        keep, _ = preservation_batch(replay_rows[start:stop])
                        counts[preservation], means[preservation] = stop - start, keep
                    accumulator.add(counts, means, backward=lambda tensor: tensor.backward())
                accumulator.finish()
                replay_lib.clip_and_step(policy.model, optimizer, scheduler,
                                         gradient_clip=float(self.config["optimization"]
                                                             ["gradient_clip"]))
            sync()
            per_family[family] = (time.perf_counter() - started) / updates
        profile.record("optimizer_update", seconds=max(per_family.values()) * updates,
                       units=updates,
                       detail={"seconds_per_update_by_family": per_family,
                               "recorded": "the slowest family, which is the conservative "
                                           "direction for a schedule",
                               "batch_rows": batch_rows, "microbatch": micro})
        del preservation_batch

        policy.model.eval()
        pairs = self.validation_pairs()
        started = time.perf_counter()
        preferences.score_sequences(policy, pairs["chosen_index"], progress_every=0)
        preferences.score_sequences(policy, pairs["rejected_index"], progress_every=0)
        full_gate_seconds = time.perf_counter() - started
        populations = self.load_populations()
        calibration_index = self.index("val")[populations["calibration_rows"]]
        high_c = (self.split("val")["class"].to_numpy()[populations["calibration_rows"]] == "high")
        started = time.perf_counter()
        policy.score(calibration_index[high_c])
        b_gate_seconds = time.perf_counter() - started
        profile.record("full_gate", seconds=full_gate_seconds, units=1,
                       detail={"pairs": int(pairs["pairs"]), "sides": 2,
                               "block_b_high_c_rows": int(high_c.sum()),
                               "block_b_seconds": b_gate_seconds,
                               "population": "the inherited fixed validation pair set"})

        started = time.perf_counter()
        take = min(int(monitor_lib.SENTINEL_PAIRS), int(pairs["pairs"]))
        # BOTH pair halves plus the parent draws: that is what a sentinel
        # actually costs. Timing the chosen half alone understated it by about
        # a third and would have understated the whole monitoring budget.
        preferences.score_sequences(policy, pairs["chosen_index"][:take], progress_every=0)
        preferences.score_sequences(policy, pairs["rejected_index"][:take], progress_every=0)
        preferences.score_sequences(policy, bank["index"][:monitor_lib.SENTINEL_PARENT_DRAWS],
                                    progress_every=0)
        profile.record("sentinel", seconds=time.perf_counter() - started, units=1,
                       detail={"pairs": take, "sides": 2,
                               "parent_draws": monitor_lib.SENTINEL_PARENT_DRAWS,
                               "measured": "both pair halves plus the parent draws"})

        draws = int(settings["generation_draws"])
        lineage = self.stream_seed(domain="profile", role="generation", parent=f"seed{seed}")
        started = time.perf_counter()
        index, _ = policy.sample(draws, seed=lineage["seed"],
                                 batch_size=int(self.config["inference"]["sample_batch_size"]))
        sample_seconds = time.perf_counter() - started
        conditional_started = time.perf_counter()
        conditionals = mixture_lib.position_log_conditionals(
            policy, index, batch_size=int(self.config["inference"]["conditional_batch_size"]))
        profile.record("generation_draw", seconds=time.perf_counter() - started, units=draws,
                       detail={"includes": ("sampling plus the conditional pass that RETAINS the "
                                            "native 20-way log vectors, which is what a coupling "
                                            "bank costs"),
                               "sample_seconds": sample_seconds,
                               "conditional_seconds": time.perf_counter() - conditional_started,
                               "retained_shape": list(np.asarray(conditionals).shape)})

        scoring_rows = min(int(settings["scoring_rows"]), int(self.index("val").shape[0]))
        probe = self.index("val")[:scoring_rows]
        started = time.perf_counter()
        policy.score(probe, batch_size=int(self.config["inference"]["score_batch_size"]))
        measured = time.perf_counter() - started
        profile.record("score_50k_pass", seconds=measured, units=scoring_rows,
                       detail={"measured_rows": scoring_rows,
                               "unit": "one scored row",
                               "note": ("a measured per-row cost. The 50k pass is a FORECAST "
                                        "quantity built from it, and the forecast multiplies "
                                        "rows; no extrapolated total is recorded here as a "
                                        "measurement with units=1.")})

        steps = int(settings["stage1_probe_steps"])
        stage1 = dict(self.config["block_b"]["stage1"])
        stage1_batch = int(stage1["batch_size"])
        stage1_micro = int(stage1.get("microbatch_rows") or stage1_batch)
        high_rows = np.flatnonzero((frame["class"] == "high").to_numpy())
        sample = self.index("train")[high_rows[:steps * stage1_batch]]
        del optimizer, scheduler, policy
        storage.collect_unused()
        policy = self.policy()
        opt = stage1["optimization"]
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=float(opt["learning_rate"]),
                                      betas=tuple(opt["betas"]), weight_decay=float(opt["weight_decay"]))
        policy.model.train()
        sync()
        started = time.perf_counter()
        for start in range(0, sample.shape[0], stage1_batch):
            chunk = sample[start:start + stage1_batch]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, chunk.shape[0], stage1_micro):
                piece = chunk[offset:offset + stage1_micro]
                (policy.loss(piece) * (piece.shape[0] / float(chunk.shape[0]))).backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), float(opt["gradient_clip"]))
            optimizer.step()
        sync()
        elapsed = time.perf_counter() - started
        epochs = int(stage1["epochs"])
        # The ACTUAL stage-1 population, counted from the data, not a planning
        # guess: the high rows of the training split.
        rows_per_epoch = int((frame.iloc[populations["purge_rows"]]["class"] == "high").sum())
        seconds_per_step = elapsed / max(1, steps)
        steps_per_fit = epochs * int(np.ceil(rows_per_epoch / stage1_batch))
        profile.record("stage1_fit", seconds=elapsed, units=steps,
                       detail={"measured_steps": steps, "seconds_per_step": seconds_per_step,
                               "steps_per_fit": steps_per_fit,
                               "rows_per_epoch": rows_per_epoch,
                               "effective_batch": stage1_batch,
                               "microbatch_rows": stage1_micro,
                               "unit": "one measured optimizer step; full fit is forecast separately"})

        # Every comparator family, separately. A small main-effects fit cannot
        # stand in for a 16,435-column interaction fit or for a CNN epoch, and
        # recording one cost for all three understates the comparator budget by
        # the two expensive members.
        comparator_rows = int(settings["comparator_rows"])
        started = time.perf_counter()
        additive_probe = self._comparator_probe(rows=comparator_rows, pairwise=False)
        additive_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pairwise_probe = self._comparator_probe(rows=comparator_rows, pairwise=True)
        pairwise_seconds = time.perf_counter() - started
        cnn_probe = self._cnn_probe(rows=comparator_rows)
        total_comparator = additive_seconds + pairwise_seconds + cnn_probe["seconds"]
        profile.record("comparator_fit", seconds=total_comparator, units=1,
                       detail={"measured_rows": comparator_rows,
                               "additive_seconds": additive_seconds,
                               "pairwise_seconds": pairwise_seconds,
                               "cnn_seconds": cnn_probe["seconds"],
                               "cnn_epochs_measured": cnn_probe["epochs"],
                               "training_rows": additive_probe["train_rows"],
                               "probes": {"additive": additive_probe, "pairwise": pairwise_probe,
                                          "cnn": {k: v for k, v in cnn_probe.items()
                                                  if k != "model"}},
                               "unit": ("one comparator SET at the probe row count: additive, "
                                        "pairwise and one CNN epoch. The forecast multiplies "
                                        "this by the declared number of sets and by the row "
                                        "ratio, and both factors are stated rather than folded "
                                        "into the measurement.")})

        started = time.perf_counter()
        target = self.context.path("profile_io_probe.npz")
        paths.write_arrays(target, {"probe": index, "log_conditionals": conditionals})
        paths.read_arrays(target)
        io_seconds = time.perf_counter() - started
        target.unlink(missing_ok=True)
        profile.record("storage_io", seconds=io_seconds, units=int(index.shape[0]),
                       detail={"operation": "atomic npz write plus read-back"})

        stats_started = time.perf_counter()
        coupling.pairwise_mutual_information(index)
        coupling.permutation_floor(index, seed=18, draws=int(self.config["coupling"]["permutation_draws"]))
        coupling.bootstrap_total_correlation(conditionals, seed=19,
                                             draws=int(self.config["coupling"]["bootstrap_draws"]))
        _bank_bootstrap(index, seed=20, draws=int(self.config["coupling"]["bootstrap_draws"]))
        stats_seconds = time.perf_counter() - stats_started
        save_started = time.perf_counter()
        store = trajectory_lib.TrajectoryStateStore(self.context.path("profile", "state_probe"),
                                                     identity={"role": "profile"})
        store.save(policy=policy, optimizer=optimizer,
                   scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
                   state=trajectory_lib.TrajectoryState(), stream_digests={})
        checkpoint_seconds = time.perf_counter() - save_started
        profile.record("profiling_compute", seconds=time.perf_counter() - stage_started, units=1,
                       detail={"note": "the profiling stage's own cost, charged separately",
                               "dependence_probe_rows": draws, "dependence_probe_seconds": stats_seconds,
                               "resume_checkpoint_seconds": checkpoint_seconds})
        del policy, optimizer
        self.release_transient()
        return profile

    def _stratified_slice(self, rows):
        """A class-stratified slice of the training split.

        Stratified rather than the first N rows: a head slice of a sorted file
        can be single-class, and the fitter would then refuse -- turning the
        measurement into a no-op that still gets recorded as a cost.
        """
        frame = self.split("train")
        generator = np.random.default_rng(int(self.config["seeds"]["replay_order"]))
        picked = []
        per_class = max(1, int(rows) // 3)
        for name in ("low", "mid", "high"):
            candidates = np.flatnonzero((frame["class"] == name).to_numpy())
            take = min(per_class, candidates.size)
            if take:
                picked.append(generator.choice(candidates, size=take, replace=False))
        if len(picked) < 3:
            return None
        selected = np.sort(np.concatenate(picked))
        order = generator.permutation(selected.size)
        half = selected.size // 2
        return {"index": self.index("train")[selected],
                "classes": np.asarray(frame["class"])[selected],
                "train_rows": order[:half], "development_rows": order[half:]}

    def _comparator_probe(self, *, rows, pairwise=False):
        """One bounded statistical comparator fit, timed by the caller."""
        block = self._stratified_slice(rows)
        if block is None:
            return {"skipped": "the training population does not carry all three classes"}
        index, classes = block["index"], block["classes"]
        train_rows, development_rows = block["train_rows"], block["development_rows"]
        fit = coupling.fit_interaction_classifier(
            index[train_rows], classes[train_rows],
            development_index=index[development_rows],
            development_classes=classes[development_rows], pairwise=bool(pairwise),
            regularization_grid=(0.1,),
            max_iterations=int(self.config["coupling"]["classifier"]["max_iterations"]))
        return {"rows": int(index.shape[0]), "train_rows": int(train_rows.size), "pairwise": bool(pairwise),
                "features": fit["features"], "fit": fit["selected_C"],
                "development_average_precision": fit["development_average_precision"]}

    def _cnn_probe(self, *, rows, epochs=1):
        """One measured CNN epoch at the probe row count."""
        from . import her2_baselines as baselines
        block = self._stratified_slice(rows)
        if block is None:
            return {"seconds": 0.0, "epochs": 0,
                    "skipped": "the training population does not carry all three classes"}
        index, classes = block["index"], block["classes"]
        train_rows, development_rows = block["train_rows"], block["development_rows"]
        targets = baselines.class_targets(classes)
        settings = dict(self.config["comparators"]["cnn"], epochs=int(epochs))
        directory = self.context.path("profile", "cnn_probe")
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        _, record = baselines.train_cnn(index[train_rows], targets[train_rows],
                                        index[development_rows], targets[development_rows],
                                        settings, seed=0, device=self.device,
                                        directory=directory)
        return {"seconds": time.perf_counter() - started, "epochs": int(epochs),
                "rows": int(train_rows.size), "parameters": record["parameters"],
                "note": ("one measured epoch. The forecast multiplies by the declared epoch "
                         "count and by the row ratio to the real population.")}

    # -- M4: calibration -------------------------------------------------
    def calibrate(self):
        """Run the bounded ladders with a real pilot runner and return the outcome.

        The ledger is restored from disk if this stage already ran: it carries
        which settings were measured, how many updates they completed and what
        they cost, so a resumed calibration does not spend the three-hour
        allocation a second time.
        """
        target = self.context.path("calibration_ledger.json")
        settings = dict(self.config["calibration"])
        persist = lambda document: paths.write_json(target, _jsonable(document))  # noqa: E731
        if target.is_file():
            ledger = calibration_lib.CalibrationLedger.restore(paths.read_json(target),
                                                               persist=persist)
        else:
            ledger = calibration_lib.CalibrationLedger(
                gpu_hour_cap=float(settings["gpu_hour_cap"]),
                max_bracket_expansions=int(settings.get("max_bracket_expansions", 2)),
                max_fallbacks_per_family=int(settings.get("max_fallbacks_per_family", 1)),
                persist=persist)
        seed = int(settings["parent_seed"])
        import torch
        baseline_path = self.context.path("calibration", "parent_baseline.json")
        identity = {"seed": seed, "source": self._snapshot_digest(),
                    "checkpoint": str(self.parent_checkpoint(seed)[0])}
        budget = WorkBudget(
            self.context.path("calibration", "budgets", "parent_baseline.json"),
            identity=identity,
            limit_seconds=ledger.remaining_seconds() + ledger.overhead_charged_seconds,
            synchronize=torch.cuda.synchronize if self.device == "cuda" else None)
        if baseline_path.is_file():
            baseline = paths.read_json(baseline_path)
            require(baseline["identity"] == identity, "Calibration baseline identity changed")
        else:
            try:
                with budget.unit("parent_development_baseline", reserve_seconds=600.0):
                    baseline = {"identity": identity, "metrics": self._parent_development_metrics(seed)}
                paths.write_json(baseline_path, _jsonable(baseline))
            finally:
                ledger.record_overhead(budget.data)
                self.release_transient()
        ledger.record_overhead(budget.data)
        parent_ap = baseline["metrics"]["macro_average_precision"]
        return calibration_lib.run_calibration(
            lambda entry, *, resume_from=None, budget_seconds=None: self._run_pilot(
                entry, seed=seed, resume_from=resume_from, budget_seconds=budget_seconds),
            parent_macro_ap=parent_ap, ledger=ledger,
            criteria=dict(settings.get("criteria") or {}),
            simpo=bool(settings.get("include_simpo")))

    def _parent_development_metrics(self, seed):
        checkpoint, _ = self.parent_checkpoint(seed)
        policy = self.policy(checkpoint)
        return self._development_metrics(policy, seed=seed, parent_policy=policy)

    def _development_metrics(self, policy, *, seed, parent_policy):
        """Macro-AP on the development population and inclusive tails on the 10k bank.

        Ranking uses the mean log probability -- an order-preserving rescale at
        fixed length -- and the yield uses the SUM, because the yield is a
        function of the probability and the mean is not one.
        """
        val_frame, val_index = self.split("val"), self.index("val")
        scored = policy.score(val_index,
                              batch_size=int(self.config["inference"]["score_batch_size"]))
        ranking_score = scored["mean_log_probability"]
        density = scored["sum_log_probability"]
        positive = (val_frame["class"] == "high").to_numpy()
        strata = self.original_strata()["labels"]
        ranking = metrics.stratified_rank_block(ranking_score, positive, strata,
                                                categories=proximity.ORIGINAL_STRATA)
        bank = self._development_bank(seed, parent_policy)
        drop = metrics.drop_block(bank["parent_scores"],
                                  policy.score(bank["index"])["sum_log_probability"])
        return {"macro_average_precision": ranking["macro_average_precision"],
                "pooled_average_precision": ranking["pooled"]["average_precision"],
                "tenfold_rate": drop["tails"]["events"]["tenfold"]["rate"],
                "hundredfold_rate": drop["tails"]["events"]["hundredfold"]["rate"],
                "yield_10k": metrics.expected_distinct_yield(density[positive], 10_000),
                "yield_1m": metrics.expected_distinct_yield(density[positive], 1_000_000),
                "yield_panel": ("distinct high identities of the original validation split; not "
                                "comparable with the challenge panel's absolute counts"),
                "ranking": ranking, "drop": drop}

    def _development_bank(self, seed, parent_policy, *, regime="original_split", rows=None):
        """The INDEPENDENT 10k development-preservation bank for one parent.

        Independent of the replay bank, which is the whole point of the role
        table: the first 10k rows of the training bank are rows the arm trained
        on, and preservation measured there is a memorization statistic. For a
        Block-A parent this is the completed campaign's saved 10k monitor bank,
        which is exactly that independent draw; otherwise it is drawn once per
        parent and persisted so a resume reads the same rows.
        """
        from . import her2_nf_reuse as reuse
        count = int(rows if rows is not None else self.config["banks"]["monitor_rows"])
        digest = _state_digest(parent_policy)
        key = f"dev_bank:{regime}:{seed}:{count}:{digest}"
        if key in self._cache:
            return self._cache[key]
        parent_id = f"parent::{regime}::seed{seed}"
        directory = self.historical_root() / "banks" / f"seed{int(seed)}" / "monitor"
        if (regime == "original_split" and count == int(self.config["banks"]["monitor_rows"])
                and (directory / "bank.complete.json").is_file()):
            saved = reuse.load_replay_bank(self.historical_root(), seed, role="monitor",
                                           parent_state_sha256=digest, expected_rows=count)
            self.reuse.setdefault("development_bank", {})[str(seed)] = {
                "reused": True, "rows": saved["rows"], "directory": saved["directory"]}
            block = {"index": saved["index"], "parent_scores": saved["parent_scores"],
                     "teacher_probabilities_array": saved["teacher_probabilities_array"],
                     "teacher_log_probabilities_array": saved["teacher_log_probabilities_array"],
                     "document": dict(saved["record"], source="saved",
                                      role="development_preservation", parent_id=parent_id,
                                      parent_state_sha256=digest)}
            self._cache[key] = block
            return block
        target = self.context.path("banks", f"development_{regime}_seed{seed}_{count}.npz")
        lineage = self.stream_seed(domain="development_preservation",
                                   role="development_preservation", regime=regime,
                                   parent=f"seed{seed}", model=digest[:16])
        existing = nf_banks.load_bank(target, role="development_preservation",
                                      parent_id=parent_id, parent_state_sha256=digest,
                                      draw_seed=lineage["seed"])
        if existing is not None:
            block = {"index": existing["index"], "parent_scores": existing["parent_scores"],
                     "document": existing["document"]}
        else:
            bank, document = nf_banks.draw_bank(
                parent_policy, role="development_preservation", parent_id=parent_id,
                parent_state_sha256=digest, seed_lineage_record=lineage, rows=count,
                batch_size=int(self.config["inference"]["sample_batch_size"]))
            nf_banks.save_bank(target, bank, document)
            block = {"index": bank.index, "parent_scores": bank.sum_log_probability,
                     "document": document}
        self._cache[key] = block
        return block

    def _development_teacher(self, seed, parent_policy, block):
        """The development bank's frozen teacher cache, saved or built once."""
        import torch
        if block.get("teacher_probabilities") is not None:
            return block
        if block.get("teacher_probabilities_array") is not None:
            probabilities = np.asarray(block.pop("teacher_probabilities_array"))
            log_probabilities = np.asarray(block.pop("teacher_log_probabilities_array"))
        else:
            probabilities, log_probabilities, document = nf_banks.teacher_cache(
                parent_policy, block["index"], label=f"development bank seed {seed}",
                batch_size=int(self.config["inference"]["teacher_batch_size"]))
            block["teacher_document"] = document
        block["teacher_probabilities"] = torch.from_numpy(probabilities)
        block["teacher_log_probabilities"] = torch.from_numpy(log_probabilities)
        return block

    #: Every pilot at one parent resolves its stream at this horizon, so a
    #: 500-update pilot and its 1,000-update continuation are prefixes of ONE
    #: stream and the continuation resumes in place. Restarting from the parent
    #: instead spent 10,500 updates against a nominal 7,500.
    CALIBRATION_HORIZON = calibration_lib.EXTENDED_UPDATES

    def _pilot_directory(self, entry_id):
        """One directory per SETTING, shared by a pilot and its continuation."""
        base = calibration_lib.extension_of(entry_id) or entry_id
        return self.context.path("calibration", str(base).replace(":", "_"))

    def _run_pilot(self, entry, *, seed, resume_from=None, budget_seconds=None):
        """One calibration pilot, or the continuation of one, plus development metrics.

        The pilots run on a **separate** spawned pairing/optimization stream, so
        a calibrated coefficient is not chosen on the exact row order the
        production arm at this parent will later consume. It is still tuning
        inside one parent family and is not an independent-parent replication.

        ``resume_from`` names the 500-update entry this call continues. The
        directory, the stream identity and the trajectory are the SAME, so the
        resumable loop picks up at update 500 and runs 500 more.

        A trajectory that stops at the gate or runs out of budget is a scientific
        outcome and is recorded as one. Anything else -- a missing bank, a shape
        mismatch, a wrong-parent refusal -- is machinery, and it raises
        :class:`~her2_nf_calibration.MachineryFailure` so the stage aborts with
        its diagnostics instead of recording a ladder as infeasible because a
        path was wrong.
        """
        directory = self._pilot_directory(entry.entry_id)
        stream_seed = self.stream_seed(domain="calibration_stream", role="calibration",
                                       regime="original_split", parent=f"seed{seed}")["seed"]
        started = time.perf_counter()
        allocated = int(entry.allocated_updates)
        import torch
        budget = WorkBudget(
            self.context.path("calibration", "budgets", _slug(entry.entry_id) + ".json"),
            identity={"entry_id": entry.entry_id, "seed": seed,
                      "coefficients": entry.coefficients, "source": self._snapshot_digest()},
            limit_seconds=(float(budget_seconds) if budget_seconds is not None else 10800.0),
            synchronize=torch.cuda.synchronize if self.device == "cuda" else None)
        archived = None
        if resume_from is not None:
            require(directory.is_dir(),
                    f"{entry.entry_id} continues {resume_from} but {directory} does not exist; a "
                    "continuation never silently becomes a fresh run from the parent")
            require((directory / trajectory_lib.RESUME_STATE).is_file(),
                    f"{entry.entry_id} continues {resume_from} but {directory} carries no resume "
                    "state; continuing would restart from the parent and spend the pilot twice")
            # The 500-update record stays, under its own name.
            archived = trajectory_lib.archive_terminal_status(directory)
        outcome, block, reason = None, {}, None
        try:
            outcome = self.run_one_trajectory(
                trajectory=self._pilot_directory(entry.entry_id).name, directory=directory,
                block="calibration", regime="original_split", seed=seed, task=entry.task,
                preservation=entry.preservation, coefficients=entry.coefficients,
                updates=allocated, endpoints=(allocated,), checkpoints=(), parent_seed=seed,
                pairing_seed=stream_seed, max_horizon=int(self.CALIBRATION_HORIZON), budget=budget)
            if outcome["status"] == trajectory_lib.STATUS_COMPLETED:
                with budget.unit("development_scoring", reserve_seconds=180.0):
                    block = self._development_metrics(outcome["policy"], seed=seed,
                                                      parent_policy=outcome["parent_policy"])
        except BudgetExhausted as error:
            reason = str(error)
        except KeyboardInterrupt:
            raise
        except Exception as error:                              # noqa: BLE001 - classified below
            elapsed = time.perf_counter() - started
            raise calibration_lib.MachineryFailure(
                f"{entry.entry_id}: the pilot did not produce a scientific observation "
                f"({type(error).__name__}: {error}). It ran {elapsed:.1f}s in {directory}. A "
                "machinery failure is not a failed pilot: recording it as one would let a whole "
                "ladder be declared infeasible because a path was wrong.") from error
        elapsed = time.perf_counter() - started
        document = outcome["document"] if outcome is not None else {}
        completed = int(document.get("updates", trajectory_lib.durable_progress(directory)["updates"]))
        status = outcome["status"] if outcome is not None else trajectory_lib.STATUS_INCOMPLETE
        if reason is not None or not block:
            status = trajectory_lib.STATUS_INCOMPLETE if status != trajectory_lib.STATUS_STOPPED else status
        if status == trajectory_lib.STATUS_COMPLETED and completed < allocated:
            status = trajectory_lib.STATUS_INCOMPLETE
        over_budget = budget.charged_seconds > budget.data["limit_seconds"]
        if over_budget or budget.data["overruns"]:
            status = trajectory_lib.STATUS_INCOMPLETE
        result = {"status": "completed" if status == trajectory_lib.STATUS_COMPLETED else status,
                "completed_updates": completed,
                "metrics": {key: block.get(key) for key in
                            ("macro_average_precision", "pooled_average_precision",
                             "tenfold_rate", "hundredfold_rate", "yield_10k", "yield_1m")},
                "cost": {"gpu_seconds": budget.data["measured_seconds"],
                         "uncertainty_debit_seconds": budget.data["uncertainty_debit_seconds"],
                         "gpu_work_units": budget.data, "invocation_wall_seconds": elapsed,
                         "basis": "synchronized wall time of GPU work, including setup/dispatch; "
                                  "interrupted-unit reservations are separate uncertainty debits",
                         "ledger": document.get("cost"),
                         "budget_seconds": budget_seconds, "over_budget": bool(over_budget),
                         "resumed_from": resume_from, "archived_status": archived,
                         "updates_actually_run": completed - (
                             int((archived or {}).get("updates") or 0) if resume_from else 0)},
                "reason": (reason or document.get("stop_reason")
                           or ("the measured GPU-hour allocation was exhausted during this pilot"
                               if over_budget else None))}
        del outcome
        self.release_transient()
        return result

    # -- production ------------------------------------------------------
    def run_production(self, *, max_trajectories=None, heartbeat=None):
        from . import her2_nf_campaign as campaign
        preflight = campaign.read_stage_record(self.context.run_root, "preflight") or {}
        # ``require_frozen=False`` so a family the calibration REFUSED does not
        # take the whole queue down with it: its arms stay
        # ``coefficients_pending`` and are skipped, every other declared arm
        # runs, and the pending rows remain visible in every table. Production
        # still never starts one, which is the guarantee that matters.
        queue = campaign.build_queue(self.config,
                                     frozen=campaign.frozen_coefficients(self.context),
                                     reuse_verified=bool(preflight.get("reuse_verified")),
                                     require_frozen=False)
        started = 0
        counts = {"completed": 0, "stopped_by_gate": 0, "incomplete": 0, "failed": 0}
        for row in queue:
            if row.status in ("deferred_optional", "reuse_verified", "coefficients_pending",
                              "no_qualified_configuration"):
                continue
            if max_trajectories is not None and started >= int(max_trajectories):
                break
            directory = self.context.path("trajectories", row.trajectory)
            existing = trajectory_lib.read_terminal_status(directory)
            if existing is not None and existing["status"] != trajectory_lib.STATUS_FAILED:
                counts[existing["status"]] = counts.get(existing["status"], 0) + 1
                continue
            started += 1
            outcome = self.run_one_trajectory(
                trajectory=row.trajectory, directory=directory, block=row.block,
                regime=row.regime, seed=row.seed, task=row.task,
                preservation=row.preservation, coefficients=row.coefficients,
                updates=max(row.endpoints), endpoints=row.endpoints,
                checkpoints=row.checkpoints, parent_seed=row.seed, heartbeat=heartbeat)
            # Written after the arm, when a Block-B challenge parent exists: the
            # declared checkpoint set starts at update 0 and that is the parent,
            # recorded as a pointer with its digest rather than an 88 MB copy.
            self.save_initial_checkpoint(directory, row.seed, block=row.block, regime=row.regime)
            counts[outcome["status"]] = counts.get(outcome["status"], 0) + 1
            del outcome
            self.release_transient()
        paths.write_json(self.context.path("stream_registry.json"), self.streams.document())
        paths.write_json(self.context.path("historical_reuse.json"),
                         _jsonable(nf_reuse.reuse_record(**self.reuse)))
        state = campaign.campaign_state(self.context.run_root, queue)
        remaining = [entry["trajectory"] for entry in state["rows"]
                     if entry["observed_status"] in campaign.NON_TERMINAL_STATUSES]
        pending = [entry["trajectory"] for entry in state["rows"]
                   if entry["observed_status"] == "coefficients_pending"]
        return {"started": started, "remaining": remaining,
                "unrunnable": pending,
                "unrunnable_reason": ("these arms cite a family the bounded calibration refused. "
                                      "That is a recorded scientific outcome; they are never "
                                      "started at an invented coefficient and they stay in every "
                                      "table with that status."),
                "all_terminal": not remaining,
                "any_terminal": bool(state["counts"].get("completed")
                                     or state["counts"].get("stopped_by_gate")
                                     or state["counts"].get("incomplete")),
                **{key: counts.get(key, 0) for key in
                   ("completed", "stopped_by_gate", "incomplete", "failed")}}

    def run_one_trajectory(self, *, trajectory, directory, block, regime, seed, task,
                           preservation, coefficients, updates, endpoints, checkpoints,
                           parent_seed, bank_rows=None, monitor_rows=None, pairing_seed=None,
                           namespace=None, max_horizon=None, heartbeat=None, budget=None):
        """Wire one arm's callbacks and hand them to the resumable loop.

        ``bank_rows``/``monitor_rows``/``namespace`` exist for the miniature
        smoke path, which has to exercise this exact function rather than a
        parallel one; ``pairing_seed`` exists for the calibration pilots, which
        run on their own spawned stream. The production queue leaves them at
        their configured values, so every arm at one parent consumes the
        identical ordered stream.

        ``max_horizon`` fixes the stream length independently of this call's
        ``updates``. A 500-update pilot and its 1,000-update continuation
        therefore resolve the SAME stream, one being a prefix of the other, and
        the continuation resumes instead of restarting. Resolving the stream at
        ``updates`` made the two different streams and made the resume refuse.
        """
        import random
        import torch
        start_seed = int(pairing_seed if pairing_seed is not None else seed) % (2 ** 32)
        random.seed(start_seed)
        np.random.seed(start_seed)
        torch.manual_seed(start_seed)
        directory = Path(directory)
        with (budget.unit("preparation", reserve_seconds=300.0)
              if budget is not None else nullcontext()):
            population, parent_policy, parent_checkpoint, parent_entry = self._population_for(
                block, regime, parent_seed)
            policy = self.policy(parent_checkpoint)
            self.save_initial_checkpoint(directory, parent_seed, block=block, regime=regime)
            optimizer, scheduler = self.optimizer_and_scheduler(
                policy, optimization=self.config.get("optimization"))
            settings = self.config["block_a" if block != "B" else "block_b"]
            batch_rows = int(settings["chosen_per_update"])
            micro = int(settings["microbatch_rows"])
            horizon = int(max_horizon if max_horizon is not None else updates)
            require(horizon >= int(updates),
                    f"the declared stream horizon {horizon} is shorter than the {updates} updates "
                    "this call asks for")
            resolved_pairing_seed = int(
                pairing_seed if pairing_seed is not None
                else int(self.config["block_a"]["pairing"]["seed_base"]) + int(seed))
            stream, order, stream_source = self._streams_for(
                block=block, regime=regime, seed=seed, population=population,
                pairing_seed=resolved_pairing_seed, batch_rows=batch_rows, horizon=horizon,
                replay_rows=int(bank_rows if bank_rows is not None
                                else self.config["banks"]["replay_rows"]))
            reference = self._reference_cache(
                parent_policy, population, key=f"{block}:{regime}:{parent_seed}",
                seed=(parent_seed if block != "B" else None),
                parent_file_sha256=(parent_entry or {}).get("file_sha256"))
            monitor_rows = int(monitor_rows if monitor_rows is not None
                               else self.config["banks"]["monitor_rows"])
            bank_rows = int(bank_rows if bank_rows is not None
                            else self.config["banks"]["replay_rows"])
            teacher_rows = bank_rows if preservation == "fkl" else 0
            replay_bank = self._replay_bank(parent_policy, parent_seed, regime,
                                            teacher_rows=teacher_rows or None, rows=bank_rows)
            development = self._development_bank(parent_seed, parent_policy, regime=regime,
                                                 rows=monitor_rows)
            self._development_teacher(parent_seed, parent_policy, development)
            # Cadence is block-specific: Block A keeps the inherited pair gate every
            # 100 updates with sentinels between; Block B's gate is all high C rows,
            # which is cheap enough to run at every sentinel, so it runs at 25.
            plan = monitor_lib.MonitorPlan(
                endpoints=endpoints, checkpoints=checkpoints,
                full_interval=int(self.config["monitor"][
                    "block_b_full_interval" if block == "B" else "full_interval"]),
                sentinel_interval=int(self.config["monitor"]["sentinel_interval"]))
            gate, sentinel = self._gates(block, parent_policy, development)
            identity = {"trajectory": trajectory, "block": block, "regime": regime,
                        "task": task, "preservation": preservation,
                        "coefficients": dict(sorted(coefficients.items())), "seed": int(seed),
                        "parent_state_sha256": _state_digest(parent_policy),
                        "namespace": namespace,
                        "source_snapshot_sha256": self._snapshot_digest()}
            digests = {"chosen": stream.document()["chosen_rows_sha256"],
                       "rejected": stream.document()["rejected_rows_sha256"],
                       "replay_order": paths.array_digest(order),
                       "horizon": str(horizon)}
            row = {"trajectory": trajectory, "block": block, "regime": regime, "task": task,
                   "seed": int(seed),
                   "arm_id": nf_objectives.arm_identifier(task, preservation, coefficients)}
            lam = float(coefficients.get("lambda", 0.0))
            policy.model.train()
            optimizer.zero_grad(set_to_none=True)
            random.seed(start_seed)
            np.random.seed(start_seed)
            torch.manual_seed(start_seed)
        document = trajectory_lib.run_trajectory(
            row=row, directory=directory, policy=policy, optimizer=optimizer,
            scheduler=scheduler, stream=stream, replay_order=order, plan=plan,
            endpoints=endpoints, batch_rows=batch_rows, microbatch_rows=micro,
            task_batch=lambda chosen, rejected: self._task_microbatch(
                policy, population, reference, task=task, coefficients=coefficients,
                chosen_rows=chosen, rejected_rows=rejected),
            preservation_batch=self._preservation_batch(policy, replay_bank, preservation),
            full_check=lambda **kwargs: self._full_check(gate, policy, development, **kwargs),
            sentinel_check=lambda **kwargs: sentinel.evaluate(policy, **kwargs),
            restore_sentinel=sentinel.restore_previous_D,
            on_endpoint=lambda **kwargs: self._endpoint(policy, directory, identity, **kwargs),
            on_checkpoint=lambda **kwargs: self._checkpoint(policy, directory, identity, **kwargs),
            checkpoints=checkpoints, budget=budget,
            gradient_clip=float(self.config["optimization"]["gradient_clip"]),
            identity=identity, uses_rejected=nf_objectives.TASK_OBJECTIVES[task].uses_rejected,
            preservation_family=preservation, preservation_lambda=lam, heartbeat=heartbeat,
            max_updates=int(updates), stream_digests=digests,
            gradient_diagnostic=(self._gradient_diagnostic(policy, population, reference,
                                                           replay_bank, task, coefficients,
                                                           preservation)
                                 if preservation != "none" else None))
        document["stream_source"] = stream_source
        document["bank_sources"] = {"replay": replay_bank.get("source", "drawn"),
                                    "development": development["document"].get("source", "drawn"),
                                    "reference_cache": reference.get("source", "computed")}
        return {"status": document["status"], "document": document, "policy": policy,
                "parent_policy": parent_policy, "population": population}

    def _streams_for(self, *, block, regime, seed, population, pairing_seed, batch_rows,
                     horizon, replay_rows):
        """The ordered task stream and the replay order for one arm.

        For a Block-A parent both are the completed campaign's **saved** arrays,
        read from ``streams/seed{seed}/task_stream.npz``: the chosen and rejected
        row order the old arms trained on, and the replay order they read the
        bank in. Re-deriving them and observing that the digests agree is a check
        on the derivation rule, not a reuse of the arrays, and the saved replay
        order was not being read at all.
        """
        from . import her2_nf_reuse as reuse
        exposures = int(batch_rows) * int(horizon)
        record = self.historical_root() / "streams" / f"seed{int(seed)}" / "task_stream.complete.json"
        inherited_pairing = int(self.config["block_a"]["pairing"]["seed_base"]) + int(seed)
        reuse_production = (block == "A" and regime == "original_split"
                            and int(pairing_seed) == inherited_pairing
                            and int(replay_rows) == 100_000)
        if reuse_production and record.is_file():
            saved = reuse.load_task_stream(self.historical_root(), seed, batch_rows=batch_rows)
            if saved.exposures >= exposures:
                self.reuse.setdefault("task_stream", {})[str(seed)] = {
                    "reused": True, "exposures": saved.exposures,
                    "replay_order_sha256": paths.array_digest(saved.replay_order)}
                return saved, saved.replay_order[:exposures], "saved"
        pairing = preferences.PreferencePairing(population, seed=int(pairing_seed))
        stream = streams_lib.resolve_task_stream(
            pairing, seed=int(seed), exposures=exposures, batch_rows=int(batch_rows),
            pairing_seed=int(pairing_seed))
        order = streams_lib.replay_order(bank_rows=int(replay_rows), exposures=exposures,
                                         seed=(int(pairing_seed) if block == "calibration"
                                               else int(self.config["seeds"]["replay_order"])),
                                         parent_seed=int(seed))
        return stream, order, "derived"

    def _population_for(self, block, regime, parent_seed):
        if block != "B":
            checkpoint, entry = self.parent_checkpoint(parent_seed)
            population = preferences.build_population(self.split("train"), "train")
            entry = dict(entry)
            entry.setdefault("file_sha256", paths.sha256_file(checkpoint)
                             if Path(checkpoint).is_file() else None)
            return population, self.policy(checkpoint), checkpoint, entry
        block_data = self.load_populations()
        rows = block_data["purge_rows" if regime == "purge" else "match_rows"]
        frame = self.split("train").iloc[np.asarray(rows)].reset_index(drop=True)
        forbidden = self.forbidden_evaluation_rows()
        forbidden.check(frame.seq, where=f"Block B {regime} training population")
        population = preferences.build_population(frame, f"train_{regime}")
        checkpoint = self._challenge_parent(regime, parent_seed, frame)
        return population, self.policy(checkpoint), checkpoint, {}

    def _challenge_parent(self, regime, seed, frame):
        directory = self.context.path("parents", f"{regime}_seed{seed}")
        record = directory / "selection.json"
        if record.is_file():
            selected = paths.read_json(record)
            expected_population = paths.array_digest(data_lib.encode_cores(frame.seq))
            require(selected.get("population_sha256") == expected_population
                    and selected.get("source_snapshot_sha256") == self._snapshot_digest(),
                    "Challenge parent selection belongs to another population or source")
            target = Path(selected["selected"]["path"])
            require(nf_reuse.checkpoint_metadata(target)["state_sha256"]
                    == selected["selected"]["state_sha256"], "Selected parent checkpoint changed")
            return target
        block = self.load_populations()
        val_frame = self.split("val")
        calibration_rows = np.asarray(block["calibration_rows"])
        selection_index = self.index("val")[calibration_rows]
        selection_labels = (val_frame["class"].to_numpy()[calibration_rows] == "high")
        high = frame[frame["class"] == "high"]
        outcome = self.fit_stage_one(
            index=data_lib.encode_cores(high.seq), seed=int(seed), directory=directory,
            plan_config=self.config["block_b"]["stage1"],
            optimization=self.config["block_b"]["stage1"]["optimization"],
            selection_index=selection_index, selection_labels=selection_labels)
        paths.write_json(record, {"schema_version": contract.NF_SCHEMA,
                                  "record_kind": "challenge_parent_selection",
                                  "population_sha256": paths.array_digest(data_lib.encode_cores(frame.seq)),
                                  "source_snapshot_sha256": self._snapshot_digest(),
                                  "regime": regime, "seed": int(seed), **_jsonable(outcome)})
        return Path(outcome["selected"]["path"])

    def fit_stage_one(self, *, index, seed, directory, plan_config, optimization,
                      selection_index, selection_labels, after_update=None):
        from . import her2_nf_stage1
        return her2_nf_stage1.fit(
            self.policy(), index=index, seed=seed, directory=directory,
            plan_config=plan_config, optimization=optimization,
            selection_index=selection_index, selection_labels=selection_labels,
            source_sha256=self._snapshot_digest(), after_update=after_update)

    def _replay_bank(self, parent_policy, seed, regime, *, teacher_rows=None, rows=None):
        """The parent's replay bank, with the teacher cache built only when it is used.

        For a Block-A parent this is the completed campaign's **saved** 100k
        replay bank, including its frozen teacher cache, loaded through
        ``read_shard`` and refused if it was drawn by a different parent. An arm
        that trains on freshly drawn replay contexts is not the arm the reused
        historical trajectory ran.

        A no-preservation control does no replay forward pass, reads no cache and
        moves no replay stream -- that is what makes its update identical to one
        produced by code with no replay term at all. It still needs the bank's
        draws and parent scores for the recorded preservation diagnostics, so the
        bank and the 160 MB teacher cache are built apart.
        """
        count = int(rows if rows is not None else self.config["banks"]["replay_rows"])
        key = f"replay:{regime}:{seed}:{count}:{_state_digest(parent_policy)}"
        if key not in self._cache:
            self._cache[key] = self._resolve_replay_bank(parent_policy, seed, regime, rows=count)
        block = self._cache[key]
        # Rebuilt when a later arm needs MORE rows than the first one did: a
        # cache covering the monitor prefix would index out of range under a
        # replay order that reaches the whole bank.
        if teacher_rows is not None and int(block["teacher_rows"]) < int(teacher_rows):
            import torch
            if block.get("saved_teacher") is not None:
                probabilities, log_probabilities = block.pop("saved_teacher")
                cache_document = {"source": "saved", "rows": int(probabilities.shape[0]),
                                  "basis": "the completed campaign's frozen teacher cache"}
                take = int(probabilities.shape[0])
            else:
                take = int(teacher_rows)
                subset = block["index"][:take]
                probabilities, log_probabilities, cache_document = nf_banks.teacher_cache(
                    parent_policy, subset, label=f"replay bank {regime} seed {seed}",
                    batch_size=int(self.config["inference"]["teacher_batch_size"]))
            # CPU tensors, moved per microbatch: the full (100k, 10, 20) pair is
            # 160 MB and does not belong resident on a 4 GiB device beside the
            # model and its optimizer state.
            block["teacher_probabilities"] = torch.from_numpy(np.asarray(probabilities))
            block["teacher_log_probabilities"] = torch.from_numpy(np.asarray(log_probabilities))
            block["teacher_document"] = cache_document
            block["teacher_rows"] = take
        return block

    def _resolve_replay_bank(self, parent_policy, seed, regime, *, rows):
        """Saved first, persisted second, freshly drawn only as a last resort."""
        from . import her2_nf_reuse as reuse
        digest = _state_digest(parent_policy)
        parent_id = f"parent::{regime}::seed{seed}"
        if regime == "original_split" and rows == int(self.config["banks"]["replay_rows"]):
            directory = (self.historical_root() / "banks" / f"seed{int(seed)}" / "replay")
            if (directory / "bank.complete.json").is_file():
                saved = reuse.load_replay_bank(self.historical_root(), seed, role="replay",
                                               parent_state_sha256=digest, expected_rows=rows)
                self.reuse.setdefault("replay_bank", {})[str(seed)] = {
                    "reused": True, "rows": saved["rows"], "directory": saved["directory"]}
                return {"index": saved["index"], "parent_scores": saved["parent_scores"],
                        "document": dict(saved["record"], source="saved", role="replay",
                                         parent_id=parent_id, parent_state_sha256=digest),
                        "saved_teacher": (saved["teacher_probabilities_array"],
                                          saved["teacher_log_probabilities_array"]),
                        "teacher_probabilities": None, "teacher_log_probabilities": None,
                        "teacher_rows": 0, "source": "saved"}
        target = self.context.path("banks", f"replay_{regime}_seed{seed}_{rows}.npz")
        lineage = self.stream_seed(domain="replay", role="replay", regime=regime,
                                   parent=f"seed{seed}", model=digest[:16])
        existing = nf_banks.load_bank(target, role="replay", parent_id=parent_id,
                                      parent_state_sha256=digest, draw_seed=lineage["seed"])
        if existing is not None:
            return {"index": existing["index"], "parent_scores": existing["parent_scores"],
                    "document": existing["document"], "teacher_probabilities": None,
                    "teacher_log_probabilities": None, "teacher_rows": 0, "source": "persisted"}
        bank, document = nf_banks.draw_bank(
            parent_policy, role="replay", parent_id=parent_id, parent_state_sha256=digest,
            seed_lineage_record=lineage, rows=int(rows),
            batch_size=int(self.config["inference"]["sample_batch_size"]))
        nf_banks.save_bank(target, bank, document)
        return {"index": bank.index, "parent_scores": bank.sum_log_probability,
                "document": document, "teacher_probabilities": None,
                "teacher_log_probabilities": None, "teacher_rows": 0, "source": "drawn"}

    def _preservation_batch(self, policy, replay_bank, preservation):
        if preservation == "none":
            return None
        import torch
        if preservation == "fkl":
            probabilities = replay_bank["teacher_probabilities"]
            log_probabilities = replay_bank["teacher_log_probabilities"]

            def fkl(rows):
                student = replay_lib.student_log_probabilities(policy, replay_bank["index"][rows])
                selection = torch.as_tensor(np.asarray(rows), dtype=torch.long)
                return replay_lib.replay_term(
                    probabilities[selection].to(student.device),
                    log_probabilities[selection].to(student.device),
                    student, label="replay")
            return fkl

        def tail(rows):
            parent = torch.as_tensor(replay_bank["parent_scores"][rows], device=policy.device)
            policy_scores = policy.sequence_log_probs(replay_bank["index"][rows])
            return nf_objectives.tail_term(parent.to(policy_scores.dtype), policy_scores)
        return tail

    def _gradient_diagnostic(self, policy, population, reference, replay_bank, task,
                             coefficients, preservation):
        preservation_batch = self._preservation_batch(policy, replay_bank, preservation)

        def diagnostic(*, update, chosen_rows, rejected_rows, preservation_rows):
            return trajectory_lib.gradient_term_diagnostics(
                policy.model,
                lambda: self._task_microbatch(policy, population, reference, task=task,
                                              coefficients=coefficients,
                                              chosen_rows=chosen_rows,
                                              rejected_rows=rejected_rows)[0],
                lambda: preservation_batch(preservation_rows)[0])
        return diagnostic

    def _gates(self, block, parent_policy, development):
        """The authoritative gate and the cheap sentinel, on their declared populations.

        Both sentinel halves come from populations the arm does NOT train on: the
        fixed validation pairs (chosen AND rejected) and the independent
        development bank. Reading the first rows of the training replay bank
        instead measured preservation on rows the arm had already seen.
        """
        draws = min(int(monitor_lib.SENTINEL_PARENT_DRAWS),
                    int(development["index"].shape[0]))
        if block != "B":
            pairs = self.validation_pairs()
            identity = {"population": "fixed_validation_pairs", "pairs": int(pairs["pairs"]),
                        "scaffold_prefix": self.scaffold().prefix}
            # Cached per parent state: the reference is 51,444 frozen scores and
            # every arm at one parent reads the identical one. Rebuilding it per
            # trajectory is a minute of GPU time that measures nothing new.
            key = f"pair_reference:{_state_digest(parent_policy)}"
            if key not in self._cache:
                self._cache[key] = _pair_reference(parent_policy, pairs, identity)
            reference = self._cache[key]
            take = min(int(monitor_lib.SENTINEL_PAIRS), int(pairs["pairs"]))
            gate = monitor_lib.FullPairGate(reference)
            sentinel = monitor_lib.SentinelGate(
                chosen_index=pairs["chosen_index"][:take],
                parent_chosen=np.asarray(reference.chosen)[:take],
                rejected_index=pairs["rejected_index"][:take],
                parent_rejected=np.asarray(reference.rejected)[:take],
                parent_index=development["index"][:draws],
                parent_log_probability=development["parent_scores"][:draws])
            return {"kind": "pairs", "gate": gate, "pairs": pairs}, sentinel
        block_data = self.load_populations()
        val_frame = self.split("val")
        rows = np.asarray(block_data["calibration_rows"])
        high = rows[(val_frame["class"].to_numpy()[rows] == "high")]
        high_index = self.index("val")[high]
        parent_scores = preferences.score_sequences(parent_policy, high_index, progress_every=0)
        gate = monitor_lib.HighRowGate(row_index=high_index,
                                       parent_log_probability=parent_scores,
                                       label="all high C rows")
        take = min(int(monitor_lib.SENTINEL_PAIRS), int(high_index.shape[0]))
        sentinel = monitor_lib.SentinelGate(
            chosen_index=high_index[:take],
            parent_chosen=np.asarray(parent_scores)[:take],
            parent_index=development["index"][:draws],
            parent_log_probability=development["parent_scores"][:draws])
        return {"kind": "high_rows", "gate": gate}, sentinel

    #: When the in-fit preservation diagnostic runs. The GATE -- the only check
    #: that can stop a trajectory -- runs at every full check regardless. The
    #: 10k-row preservation block is the expensive half and runs at the declared
    #: modest cadence: the first update, every endpoint, every checkpoint, the
    #: final state, and whenever a sentinel asked for a look.
    PRESERVATION_REASONS = ("first_update", "exposure_endpoint", "checkpoint", "final_state",
                            "sentinel_request")

    def _full_check(self, gate, policy, development, *, update, reason, exposures):
        """The authoritative gate, plus preservation on the INDEPENDENT bank at cadence."""
        if gate["kind"] == "pairs":
            verdict = gate["gate"].evaluate(policy, gate["pairs"], update=update, reason=reason)
        else:
            verdict = gate["gate"].evaluate(policy, update=update, reason=reason)
        if reason in self.PRESERVATION_REASONS:
            verdict["preservation"] = monitor_lib.preservation_block(
                policy, development["index"],
                parent_log_probability=development["parent_scores"],
                teacher_probabilities=development.get("teacher_probabilities"),
                teacher_log_probabilities=development.get("teacher_log_probabilities"),
                conditional_batch=int(self.config["inference"]["conditional_batch_size"]),
                label="development preservation bank")
        else:
            verdict["preservation"] = {
                "available": False, "reason": "not scheduled at this check",
                "cadence": list(self.PRESERVATION_REASONS)}
        verdict["preservation_population"] = ("the independent development bank, which the arm "
                                              "does not train on")
        verdict["preservation_cadence"] = ("the gate runs at every full check; the 10k-row "
                                           "preservation diagnostic runs at the first update, "
                                           "every endpoint and checkpoint, the final state and "
                                           "any sentinel request. That is a declared cadence, "
                                           "not a silent sampling of it.")
        return verdict

    def _endpoint(self, policy, directory, identity, *, update, exposures, gate, preservation,
                  ledger):
        from . import her2_policy
        target = Path(directory) / f"endpoint_update{int(update)}.pt"
        digest = storage.save_checkpoint(target, policy, {
            "record_kind": "nf_exposure_endpoint", "update": int(update),
            "exposures": dict(exposures), "identity": dict(identity)})
        return {"update": int(update), "exposures": dict(exposures),
                "checkpoint": {"path": str(target), "state_sha256": digest,
                               "file_sha256": paths.sha256_file(target)},
                "gate_D": gate.get("D"),
                "preservation_available": bool((preservation or {}).get("available")),
                "evaluation_note": ("full evaluation of this endpoint runs in the audit and "
                                    "coupling stages against frozen, named checkpoints.")}

    def _checkpoint(self, policy, directory, identity, *, update, exposures, gate):
        """An immutable intermediate checkpoint at a declared checkpoint update.

        Listing 250 and 500 in the monitor plan saved nothing: ``on_endpoint``
        only ran for the endpoint updates, and the rolling resume state is
        overwritten at every later check. These are written under their own
        names and are never overwritten.
        """
        from . import her2_policy
        target = Path(directory) / f"checkpoint_update{int(update)}.pt"
        existed = target.is_file()
        digest = storage.save_checkpoint(target, policy, {
            "record_kind": "nf_intermediate_checkpoint", "update": int(update),
            "exposures": dict(exposures), "identity": dict(identity)})
        return {"update": int(update), "checkpoint": str(target), "rewritten": not existed,
                "state_sha256": digest, "file_sha256": paths.sha256_file(target),
                "gate_D": (gate or {}).get("D"),
                "immutability": "written once under its own name; never a rolling state"}

    def save_initial_checkpoint(self, directory, seed, *, block="A", regime="original_split"):
        """The ``update 0`` checkpoint: the parent, named as this arm's own start.

        Block A's declared checkpoint set starts at 0 and Block B's does too. The
        parent file already holds those weights, so this records the pointer and
        its digest rather than copying an 88 MB file twelve times.
        """
        from . import her2_nf_reuse as reuse
        if block != "B":
            checkpoint, entry = self.parent_checkpoint(seed)
        else:
            checkpoint = self._challenge_parent_if_fitted(regime, seed)
            entry = {}
            if checkpoint is None:
                return None
        target = Path(directory) / "checkpoint_update0.json"
        metadata = reuse.checkpoint_metadata(checkpoint)
        record = {"schema_version": contract.NF_SCHEMA, "record_kind": "nf_initial_checkpoint",
                  "update": 0, "points_to": str(checkpoint),
                  "state_sha256": metadata["state_sha256"],
                  "file_sha256": metadata["file_sha256"],
                  "basis": ("update 0 is the parent itself. The pointer and its digest are "
                            "recorded rather than an identical copy per arm.")}
        paths.write_json(target, record)
        return record

    def _snapshot_digest(self):
        marker = self.context.path(spec.FREEZE_MARKER)
        return paths.read_json(marker)["snapshot_sha256"] if marker.is_file() else None

    # -- audits and coupling ---------------------------------------------
    def run_audits(self):
        """Fresh 50k preservation banks per parent, scoring EVERY frozen named checkpoint.

        The named list is the preservation audit list: Block A at u1000 and
        u3750, Block B at u1000, the parents and the reused controls. Every row
        of it is scored, every per-row log ratio is persisted so a paired
        recomputation is possible later, and the parent-context drift is streamed
        over the whole bank rather than a slice.
        """
        freeze = self._finalist_freeze()
        models, banks, seen = [], [], set()
        for entry in freeze["named_checkpoints"]:
            if entry.get("parent_checkpoint") is None:
                continue
            parent_policy = self.policy(Path(entry["parent_checkpoint"]))
            bank = self._final_bank(parent_policy, entry["parent_id"])
            if entry["parent_id"] not in seen:
                banks.append(bank["document"])
                seen.add(entry["parent_id"])
            nf_banks.require_may_influence_selection("final_preservation", freeze_record=freeze,
                                                     where="final preservation audit")
            policy = self.policy(Path(entry["checkpoint"]))
            scores = np.asarray(policy.score(
                bank["index"],
                batch_size=int(self.config["inference"]["score_batch_size"])
            )["sum_log_probability"], dtype=np.float64)
            ratio = np.asarray(bank["parent_scores"], dtype=np.float64) - scores
            drop = metrics.drop_block(bank["parent_scores"], scores)
            strict = contract.both_tail_conventions(ratio)
            target = self.context.path("preservation", f"{entry['name']}.npz")
            paths.write_arrays(target, {"parent_sum_log_probability":
                                        np.asarray(bank["parent_scores"], dtype=np.float64),
                                        "policy_sum_log_probability": scores,
                                        "log_ratio": ratio,
                                        "core_index": np.asarray(bank["index"])})
            models.append({"model": entry["name"], "checkpoint": entry["checkpoint"],
                           "arm": entry.get("arm"), "seed": entry.get("seed"),
                           "update": entry.get("update"), "regime": entry.get("regime"),
                           "reused": entry.get("reused"),
                           "parent_id": entry["parent_id"], "rows": int(bank["index"].shape[0]),
                           "drop": drop, "tail_conventions": strict,
                           "row_scores": str(target.relative_to(self.context.run_root).as_posix()),
                           "parent_context_drift": self._drift_block(parent_policy, policy,
                                                                     bank["index"])})
        return {"schema_version": contract.NF_SCHEMA, "record_kind": "preservation_audit",
                "models": models, "banks": banks,
                "freeze": {key: freeze[key] for key in
                           ("tail_family", "coverage", "missing_required", "frozen_at")
                           if key in freeze},
                "coverage": {"named": len(freeze["named_checkpoints"]), "audited": len(models),
                             "missing": freeze.get("missing_required") or []},
                "worst_upper": _grouped_worst_upper(models),
                "persisted": ("one npz per audited model with the parent scores, the policy "
                              "scores, the per-row log ratio and the bank draws, which is what a "
                              "paired recomputation needs"),
                "bank_role": nf_banks.BANK_ROLES["final_preservation"].document()}

    def _drift_block(self, parent_policy, policy, index, *, rows=None):
        """Per-position ``T``, ``B`` and ``T - B`` over the WHOLE common parent bank.

        Streamed: only ``(10, 20)`` float64 sufficient statistics are retained, so
        a 50k bank costs one conditional chunk at a time instead of two resident
        80 MB blocks. The previous bounded 4,000-row slice was a reduction of a
        required measurement to fit an allocation, which is the wrong direction.
        """
        count = int(rows) if rows is not None else int(np.asarray(index).shape[0])
        sample = np.asarray(index)[:count]
        batch = int(self.config["inference"]["conditional_batch_size"])
        chunk = int(self.config["coupling"].get("drift_chunk_rows", 2000))
        accumulator = coupling.DriftAccumulator()
        for start in range(0, sample.shape[0], chunk):
            block = sample[start:start + chunk]
            teacher = mixture_lib.position_log_conditionals(parent_policy, block,
                                                            batch_size=batch)
            student = mixture_lib.position_log_conditionals(policy, block, batch_size=batch)
            accumulator.add(teacher, student)
        record = accumulator.finish()
        record["rows_used"] = int(sample.shape[0])
        record["chunk_rows"] = chunk
        return record

    def _final_bank(self, parent_policy, parent_id):
        key = f"final:{parent_id}"
        if key in self._cache:
            return self._cache[key]
        digest = _state_digest(parent_policy)
        lineage = self.stream_seed(domain="final_preservation", role="final_preservation",
                                   parent=str(parent_id), model=digest[:16])
        target = self.context.path("banks", f"final_{_slug(parent_id)}.npz")
        existing = nf_banks.load_bank(target, role="final_preservation", parent_id=parent_id,
                                      parent_state_sha256=digest, draw_seed=lineage["seed"])
        if existing is not None:
            block = {"index": existing["index"], "parent_scores": existing["parent_scores"],
                     "document": existing["document"]}
        else:
            bank, document = nf_banks.draw_bank(
                parent_policy, role="final_preservation", parent_id=parent_id,
                parent_state_sha256=digest, seed_lineage_record=lineage,
                rows=int(self.config["banks"]["final_preservation_rows"]),
                batch_size=int(self.config["inference"]["sample_batch_size"]))
            nf_banks.save_bank(target, bank, document)
            block = {"index": bank.index, "parent_scores": bank.sum_log_probability,
                     "document": document}
        self._cache[key] = block
        return block

    def _finalist_freeze(self):
        target = self.context.path("finalist_freeze.json")
        require(target.is_file(),
                f"{target} is absent. The finalist choice and the reported checkpoint names are "
                "frozen BEFORE any final audit bank is scored; without that record the audit "
                "bank would be a development bank.")
        return paths.read_json(target)

    def freeze_finalists(self):
        """Choose the tail family on three-seed DEVELOPMENT results and name the checkpoints.

        The order is the whole point: this record exists before any
        final-preservation or finalist generation bank is drawn, and
        ``require_may_influence_selection`` refuses those banks until it does.

        The rule, in order: all-seed point-rate feasibility first; then higher
        mean macro-AP; a .001 AP tie broken by Y@10k, then Y@1M, then IPO-tail.
        If neither family is feasible, the one minimizing the maximum normalized
        tail violation is audited under the same tie rules, and that is reported
        rather than presented as a feasible choice.
        """
        from . import her2_nf_report as report_lib
        from . import her2_nf_campaign as campaign
        production = campaign.read_stage_record(self.context.run_root, "production") or {}
        production_digest = campaign.production_outcome_digest(self.context.run_root)
        existing = self.context.path("finalist_freeze.json")
        if existing.is_file():
            record = paths.read_json(existing)
            if _freeze_still_stands(record, production_outcome_sha256=production_digest):
                return record
        registry = self.checkpoint_registry()["entries"]
        seeds = [int(value) for value in self.config["block_a"]["parent_seeds"]]
        endpoint = int(self.config["block_a"]["endpoint_updates"][0])
        families = {}
        for family, arm in (("ipo_tail", "IPO_TAIL"), ("dpo_tail", "DPO_TAIL")):
            per_seed = {}
            for seed in seeds:
                entry = registry.get(f"A_{arm}@u{endpoint}_seed{seed}") or {}
                if entry.get("status") != "present":
                    continue
                parent_policy = self.policy(self.parent_checkpoint(seed)[0])
                per_seed[int(seed)] = self._development_metrics(
                    self.policy(entry["checkpoint"]), seed=seed, parent_policy=parent_policy)
            families[family] = per_seed
        choice = _choose_tail_family(families, declared_seeds=seeds)
        # The COUPLING finalists: the declared four arms at u1000, with their
        # parents. The reported PRESERVATION checkpoints are a different, larger
        # list -- A at u1000 and u3750, B at u1000 -- and conflating them dropped
        # every required audit that was not an A-u1000 coupling finalist.
        named, coupling, screens, coupling_missing = [], [], [], []
        finalist_arms = [arm for arm in ("IPO_0", "IPO_FKL", "DPO_FKL", choice["family_arm"])
                         if arm]
        for arm in finalist_arms:
            for seed in seeds:
                entry = registry.get(f"A_{arm}@u{endpoint}_seed{seed}") or {}
                if entry.get("status") != "present":
                    coupling_missing.append({"name": f"A_{arm}@u{endpoint}_seed{seed}",
                                             "reason": entry.get("reason", "not in the registry")})
                    continue
                coupling.append(_freeze_entry(entry, registry))
        if choice["family_arm"] is None:
            coupling_missing.extend({"name": f"tail_family@u{endpoint}_seed{seed}",
                                      "reason": "no_qualified_configuration: neither tail family has an endpoint"}
                                     for seed in seeds)
        for seed in seeds:
            parent = registry.get(f"parent_seed{seed}") or {}
            if parent.get("status") == "present":
                coupling.append(_freeze_entry(parent, registry))
        audit_names = []
        for update in self.config["analysis"]["primary_endpoints"]["block_a"]:
            for arm in [a["id"] for a in self.config["block_a"]["arms"] if not a.get("optional")]:
                audit_names += [f"A_{arm}@u{int(update)}_seed{seed}" for seed in seeds]
            audit_names += [f"continued_sft@u{int(update)}_seed{seed}" for seed in seeds
                            if f"continued_sft@u{int(update)}_seed{seed}" in registry]
        for update in self.config["analysis"]["primary_endpoints"]["block_b"]:
            for regime in self.config["block_b"]["regimes"]:
                for arm in [a["id"] for a in campaign.block_b_arms(
                        self.config, campaign._frozen_if_available(self.context))]:
                    audit_names += [f"B_{regime}_{arm}@u{int(update)}_seed{seed}"
                                    for seed in self.config["block_b"]["parent_seeds"]]
        audit_names += [f"parent_seed{seed}" for seed in seeds]
        audit_names += [f"B_{regime}_parent_seed{seed}"
                        for regime in self.config["block_b"]["regimes"]
                        for seed in self.config["block_b"]["parent_seeds"]]
        missing = []
        for name in sorted(set(audit_names)):
            entry = registry.get(name)
            if entry is None or entry["status"] != "present":
                missing.append({"name": name,
                                "reason": (entry or {}).get("reason", "not in the registry")})
                continue
            named.append(_freeze_entry(entry, registry))
        for name, entry in sorted(registry.items()):
            if entry["status"] == "present":
                screens.append(dict(_freeze_entry(entry, registry), name=f"screen_{name}"))
        return report_lib.finalist_freeze(
            self.context, named_checkpoints=named, coupling_finalists=coupling,
            screen_models=screens, tail_family=choice,
            missing_required=missing, coupling_missing=coupling_missing,
            provisional=production.get("status") != "completed",
            production_outcome_sha256=production_digest,
            reason=("three-seed development results under the declared rule; the other family's "
                    "10k result is retained and reported"))

    def checkpoint_registry(self):
        """ONE registry of every checkpoint this flight reports on, new or reused.

        Searching only the new trajectory directories silently dropped the reused
        historical IPO0/IPO+FKL endpoints and the continued-SFT controls from the
        finalist audit and from the result matrix, which then looked complete
        with the reused controls simply absent. Every expected entry appears here
        with an outcome -- ``present``, ``missing`` or ``not_applicable`` -- and
        the finalist list is built from this and nothing else.
        """
        if "registry" in self._cache:
            return self._cache["registry"]
        from . import her2_nf_campaign as campaign
        entries, expected = {}, []
        block_a = self.config["block_a"]
        historical = self.config["historical"]
        # A historical endpoint stands in for a factorial cell ONLY after preflight
        # verified the reuse. When parity failed, the arm is retrained here; a
        # replacement that stopped before its endpoint keeps its own missing entry
        # and its own stop reason. Substituting anyway put precisely the trajectory
        # that FAILED qualification into the contrasts and the audit, and made a
        # stopped outcome read as an available result.
        preflight = campaign.read_stage_record(self.context.run_root, "preflight") or {}
        reuse_verified = bool(preflight.get("reuse_verified"))
        for seed in block_a["parent_seeds"]:
            parent_path, parent_entry = self.parent_checkpoint(seed)
            self._register(entries, expected, name=f"parent_seed{seed}", kind="parent",
                           path=parent_path, seed=seed, update=0, arm="parent",
                           parent_id=f"parent::seed{seed}", parent_checkpoint=parent_path,
                           state_sha256=parent_entry.get("state_sha256"))
            for arm in block_a["arms"]:
                if arm.get("optional"):
                    continue
                reuse_template = arm.get("reuse_historical")
                substitutable = bool(reuse_template) and reuse_verified
                for update in sorted(set(int(v) for v in block_a["endpoint_updates"])
                                     | set(int(v) for v in block_a["checkpoint_updates"])):
                    if update == 0:
                        continue
                    new = self.context.path("trajectories", f"A_{arm['id']}_seed{seed}",
                                            f"endpoint_update{update}.pt")
                    interim = self.context.path("trajectories", f"A_{arm['id']}_seed{seed}",
                                                f"checkpoint_update{update}.pt")
                    candidate = (new if new.is_file() else
                                 interim if interim.is_file() else None)
                    reused = None
                    if candidate is None and substitutable:
                        old = (self.historical_root() / reuse_template.format(seed=seed)
                               / f"endpoint_update{update}.pt")
                        if old.is_file():
                            candidate, reused = old, "historical"
                    self._register(
                        entries, expected, name=f"A_{arm['id']}@u{update}_seed{seed}",
                        kind="preference_cell", path=candidate, seed=seed, update=update,
                        arm=arm["id"], parent_id=f"parent::seed{seed}",
                        parent_checkpoint=parent_path, reused=reused,
                        missing_reason=("no historical checkpoint exists at this update; the "
                                        "reused paths carry only their established 1000/2000/"
                                        "3750 endpoints and the early ones are MISSING, never "
                                        "reconstructed" if substitutable else
                                        self._missing_checkpoint_reason(f"A_{arm['id']}_seed{seed}")))
            for update in historical.get("continued_sft_updates", (1000, 3750)):
                template = historical.get("continued_sft",
                                          "trajectories/continued_sft_lambda0p1_seed{seed}")
                old = (self.historical_root() / template.format(seed=seed)
                       / f"endpoint_update{int(update)}.pt")
                self._register(entries, expected,
                               name=f"continued_sft@u{int(update)}_seed{seed}",
                               kind="historical_control", path=old if old.is_file() else None,
                               seed=seed, update=int(update), arm="continued_sft_lambda0p1",
                               parent_id=f"parent::seed{seed}", parent_checkpoint=parent_path,
                               reused="historical")
        block_b = self.config["block_b"]
        for regime in block_b["regimes"]:
            for seed in block_b["parent_seeds"]:
                parent = self._challenge_parent_if_fitted(regime, seed)
                self._register(entries, expected, name=f"B_{regime}_parent_seed{seed}",
                               kind="challenge_parent", path=parent, seed=seed, update=0,
                               arm="parent", regime=regime,
                               parent_id=f"parent::{regime}::seed{seed}",
                               parent_checkpoint=parent)
                for arm in campaign.block_b_arms(self.config, campaign._frozen_if_available(self.context)):
                    for update in sorted(set(int(v) for v in block_b["endpoint_updates"])
                                         | set(int(v) for v in block_b["checkpoint_updates"])):
                        if update == 0:
                            continue
                        directory = self.context.path("trajectories",
                                                      f"B_{regime}_{arm['id']}_seed{seed}")
                        new = directory / f"endpoint_update{update}.pt"
                        interim = directory / f"checkpoint_update{update}.pt"
                        candidate = (new if new.is_file() else
                                     interim if interim.is_file() else None)
                        self._register(
                            entries, expected,
                            name=f"B_{regime}_{arm['id']}@u{update}_seed{seed}",
                            kind="preference_cell", path=candidate, seed=seed, update=update,
                            arm=arm["id"], regime=regime,
                            parent_id=f"parent::{regime}::seed{seed}", parent_checkpoint=parent,
                            missing_reason=self._missing_checkpoint_reason(
                                f"B_{regime}_{arm['id']}_seed{seed}"))
        document = {"schema_version": contract.NF_SCHEMA, "record_kind": "checkpoint_registry",
                    "entries": entries, "expected": len(expected),
                    "present": sum(1 for block in entries.values() if block["status"] == "present"),
                    "missing": sorted(name for name, block in entries.items()
                                      if block["status"] == "missing"),
                    "rule": ("one registry for new and reused weights. A missing entry is an "
                             "outcome with a reason, never an omitted row, and no checkpoint is "
                             "reconstructed to fill one.")}
        self._cache["registry"] = document
        return document

    def _missing_checkpoint_reason(self, trajectory):
        from . import her2_nf_campaign as campaign
        terminal = self.context.path("trajectories", trajectory, trajectory_lib.STATUS_JSON)
        if terminal.is_file():
            record = paths.read_json(terminal)
            return f"{record['status']} at update {record.get('updates', 0)}: {record.get('stop_reason')}"
        queue = campaign.build_queue(self.config, frozen=campaign._frozen_if_available(self.context),
                                     require_frozen=False)
        for row in campaign.campaign_state(self.context.run_root, queue)["rows"]:
            if row["trajectory"] == trajectory:
                return row["observed_status"]
        return "no checkpoint exists at this update"

    def _register(self, entries, expected, *, name, kind, path, seed, update, arm,
                  parent_id=None, parent_checkpoint=None, regime=None, reused=None,
                  state_sha256=None, missing_reason=None):
        expected.append(name)
        block = {"name": name, "kind": kind, "arm": arm, "seed": int(seed), "update": int(update),
                 "regime": regime, "parent_id": parent_id,
                 "parent_checkpoint": None if parent_checkpoint is None else str(parent_checkpoint),
                 "reused": reused}
        if path is None or not Path(path).is_file():
            block.update(status="missing", checkpoint=None,
                         reason=missing_reason or "no checkpoint exists at this update")
        else:
            metadata = nf_reuse.checkpoint_metadata(path)
            require(state_sha256 is None or state_sha256 == metadata["state_sha256"],
                    f"{name}: checkpoint changed from its declared state")
            block.update(status="present", checkpoint=str(path),
                         state_sha256=metadata["state_sha256"], file_sha256=metadata["file_sha256"])
        entries[name] = block
        return block

    def _endpoint_checkpoint(self, trajectory, update):
        for name in (f"endpoint_update{int(update)}.pt", f"checkpoint_update{int(update)}.pt"):
            target = self.context.path("trajectories", trajectory, name)
            if target.is_file():
                return target
        return None

    def run_coupling(self):
        """Screens for every eligible model, two independent 50k banks for the finalists.

        The screens cover every checkpoint in the registry -- every method and
        every endpoint, not three parents -- plus the primary ``alpha = .89``
        mixture and, where the challenge has fitted parents, its models too. The
        finalist banks go to the declared coupling list.
        """
        freeze = self._finalist_freeze()
        screens, finalists, mixtures = [], [], []
        for entry in freeze["screen_models"]:
            policy = self.policy(Path(entry["checkpoint"]))
            screens.append(self._coupling_block(policy, entry, role="q_generation_screen",
                                                repeats=1))
            del policy
            self.release_transient()
        for entry in freeze.get("coupling_finalists") or []:
            finalists.append(self._coupling_block(policy=self.policy(Path(entry["checkpoint"])),
                                                  entry=entry, role="q_generation_finalist",
                                                  repeats=int(self.config["banks"]
                                                              ["finalist_banks_per_model"])))
            self.release_transient()
        mixture_entries = list(freeze.get("coupling_finalists") or [])
        mixture_entries += [entry for entry in freeze["named_checkpoints"]
                            if entry.get("regime") in self.config["block_b"]["regimes"]]
        for entry in mixture_entries:
            if entry.get("arm") != "IPO_0" or not entry.get("parent_checkpoint"):
                continue
            mixtures.append(self._mixture_coupling_block(entry))
            self.release_transient()
        return {"schema_version": contract.NF_SCHEMA, "record_kind": "coupling_audit",
                "screens": screens, "finalists": finalists, "mixture_diagnostics": mixtures,
                "banks": sum(len(block["banks"]) for block in screens + finalists),
                "coverage": {"screened_models": len(screens),
                             "finalist_models": len(finalists),
                             "mixture_diagnostics": len(mixtures),
                             "rule": ("every eligible checkpoint receives a screen; the declared "
                                      "finalists receive two independent 50k banks each; the "
                                      "primary .89 mixture receives a matched 10k diagnostic per "
                                      "seed, computed from the TRUE mixture conditionals.")},
                "uncertainty": ("independent model-specific banks describe repeat variation; the "
                                "bootstrap describes Monte Carlo sensitivity. Neither is an extra "
                                "training seed and the raw three-seed contrasts are reported.")}

    def _coupling_block(self, policy, entry, *, role, repeats):
        blocks, banks = [], []
        for repeat in range(int(repeats)):
            lineage = self.stream_seed(domain=role, role=role, regime=entry.get("regime"),
                                       parent=entry.get("parent_id"), model=entry["name"],
                                       repeat=repeat)
            bank, document = nf_banks.draw_bank(
                policy, role=role, parent_id=entry.get("parent_id", entry["name"]),
                # The CHECKPOINT drew these rows; its lineage parent did not.
                parent_state_sha256=_state_digest(policy),
                lineage_parent_state_sha256=entry.get("parent_state_sha256"),
                seed_lineage_record=lineage, retain_conditionals=True,
                batch_size=int(self.config["inference"]["sample_batch_size"]),
                conditional_batch=int(self.config["inference"]["conditional_batch_size"]))
            banks.append(document)
            pairwise = coupling.pairwise_mutual_information(bank.index)
            floor = coupling.permutation_floor(
                bank.index,
                seed=self.stream_seed(domain="permutation_floor", role="permutation",
                                      model=entry["name"], repeat=repeat)["seed"],
                draws=int(self.config["coupling"]["permutation_draws"]))
            total = coupling.total_correlation(bank.conditionals)
            sizes = [int(value) for value in self.config["coupling"]["bank_size_sensitivity"]
                     if int(value) <= bank.rows]
            sensitivity = coupling.bank_size_sensitivity(
                bank.conditionals, sizes=sizes,
                seed=self.stream_seed(domain="bootstrap", role="bank_size", model=entry["name"],
                                      repeat=repeat)["seed"]) if sizes else None
            diversity = dict(_diversity_block(bank.index), known_panel=self._panel_coverage(bank.index))
            # The bootstrap runs on the FINALIST banks only. It is a resampling of
            # one bank and costs a full 45-pair pass per draw, so spending it on
            # every exploratory screen would buy Monte Carlo error bars on
            # numbers the flight already calls exploratory.
            bootstrap = (_bank_bootstrap(
                bank.index,
                seed=self.stream_seed(domain="bootstrap", role="mi_sum", model=entry["name"],
                                      repeat=repeat)["seed"],
                draws=int(self.config["coupling"].get("bootstrap_draws", 50)))
                if role == "q_generation_finalist" else
                {"skipped": "screens are exploratory; the bootstrap runs on the finalists"})
            tc_bootstrap = (coupling.bootstrap_total_correlation(
                bank.conditionals,
                seed=self.stream_seed(domain="bootstrap", role="total_correlation",
                                      model=entry["name"], repeat=repeat)["seed"],
                draws=int(self.config["coupling"].get("bootstrap_draws", 50)))
                if role == "q_generation_finalist" else None)
            blocks.append({"repeat": repeat, "rows": bank.rows, "bootstrap": bootstrap,
                           "total_correlation_bootstrap": tc_bootstrap,
                           "draw_seed": int(lineage["seed"]),
                           "pairwise_mi_sum": pairwise["sum"], "per_pair": pairwise["per_pair"],
                           "permutation_floor": floor, "diversity": diversity,
                           "bank_size_sensitivity": sensitivity,
                           "total_correlation": {k: v for k, v in total.items()
                                                 if k != "log_marginals"},
                           "log_marginals_sha256": paths.array_digest(total["log_marginals"])})
            paths.write_arrays(
                self.context.path("coupling", f"{_slug(entry['name'])}_{role}_{repeat}.npz"),
                {"index": bank.index, "log_conditionals": bank.conditionals.astype(np.float64),
                 "log_marginals": total["log_marginals"],
                 "sum_log_probability": np.asarray(bank.sum_log_probability, dtype=np.float64)})
        repeat_agreement = None
        if len(blocks) > 1:
            values = [block["total_correlation"]["total_correlation"] for block in blocks]
            repeat_agreement = {"values": values,
                                "range": float(max(values) - min(values)),
                                "meaning": ("independent banks from the SAME model. This is "
                                            "Monte Carlo repeat variation, not training-seed "
                                            "variation.")}
        return {"model": entry["name"], "role": role, "arm": entry.get("arm"),
                "seed": entry.get("seed"), "update": entry.get("update"),
                "regime": entry.get("regime"), "repeats": blocks, "banks": banks,
                "repeat_agreement": repeat_agreement,
                "storage_note": ("native log conditionals are persisted at float64 and their "
                                 "float64 derived marginals are hashed; the precision choice is "
                                 "declared, not silent.")}

    def _panel_coverage(self, index):
        if "panel_membership" not in self._cache:
            train = set(self.split("train").seq.tolist())
            known = train | set(self.split("val").seq.tolist())
            self._cache["panel_membership"] = train, known
        train, known = self._cache["panel_membership"]
        cores = data_lib.decode_cores(index)
        n = len(cores)
        train_hits = sum(core in train for core in cores)
        known_hits = sum(core in known for core in cores)
        return {"draws": n, "training_hits": train_hits, "training_hit_rate": train_hits / n,
                "known_panel_hits": known_hits, "known_panel_hit_rate": known_hits / n,
                "outside_known_panel": n - known_hits,
                "denominator": "all generated draws, including repeats",
                "known_panel": "train plus validation sequences only; no sealed-test labels",
                "interpretation": "outside this panel has unknown measured functionality"}

    def _mixture_coupling_block(self, entry):
        """A matched 10k coupling diagnostic for the primary ``alpha = .89`` mixture.

        Drawn as a mixture -- one component per sequence -- and scored with the
        TRUE posterior-weighted mixture conditionals. Substituting either
        component's conditionals would answer a different question, and the
        component indicator can itself induce dependence, which is why this is
        reported apart from the single-model screens.
        """
        from . import her2_nf_report as report_lib
        alpha = float(mixture_lib.PRIMARY_ALPHA)
        parent_policy = self.policy(Path(entry["parent_checkpoint"]))
        policy = self.policy(Path(entry["checkpoint"]))
        rows = int(nf_banks.BANK_ROLES["q_generation_screen"].rows)
        lineage = self.stream_seed(domain="mixture_component", role="mixture",
                                   parent=entry.get("parent_id"), model=entry["name"])
        parent_seed = self.stream_seed(domain="mixture_component", role="mixture_parent",
                                       parent=entry.get("parent_id"), model=entry["name"])
        policy_seed = self.stream_seed(domain="mixture_component", role="mixture_policy",
                                       parent=entry.get("parent_id"), model=entry["name"])
        drawn = mixture_lib.sample_mixture(
            parent_policy, policy, count=rows, alpha=alpha,
            component_seed=lineage["seed"], parent_seed=parent_seed["seed"],
            policy_seed=policy_seed["seed"],
            batch_size=int(self.config["inference"]["sample_batch_size"]))
        vectors = mixture_lib.conditional_vectors_for_mixture(
            parent_policy, policy, drawn["index"], alpha=alpha,
            batch_size=int(self.config["inference"]["conditional_batch_size"]))
        total = coupling.total_correlation(vectors["log_conditionals"])
        pairwise = coupling.pairwise_mutual_information(drawn["index"])
        floor = coupling.permutation_floor(
            drawn["index"], draws=int(self.config["coupling"]["permutation_draws"]),
            seed=self.stream_seed(domain="permutation_floor", role="mixture",
                                  model=entry["name"])["seed"])
        scored = mixture_lib.score_mixture(
            parent_policy, policy, drawn["index"], alpha=alpha,
            batch_size=int(self.config["inference"]["score_batch_size"]))
        target = self.context.path("coupling", f"mixture_{_slug(entry['name'])}.npz")
        paths.write_arrays(target, {
            "index": drawn["index"], "component": drawn["component"],
            "mixture_sum_log_probability": scored["mixture_sum_log_probability"],
            "parent_sum_log_probability": scored["parent_sum_log_probability"],
            "policy_sum_log_probability": scored["policy_sum_log_probability"],
            "log_conditionals": vectors["log_conditionals"],
            "log_marginals": total["log_marginals"]})
        return {"model": f"mixture_alpha{alpha:g}::{entry['name']}", "alpha": alpha,
                "rows": rows, "seed": entry.get("seed"),
                "draws_from_parent": drawn["draws_from_parent"],
                "draws_from_policy": drawn["draws_from_policy"],
                "pairwise_mi_sum": pairwise["sum"],
                "per_pair": pairwise["per_pair"], "permutation_floor": floor,
                "diversity": dict(_diversity_block(drawn["index"]),
                                  known_panel=self._panel_coverage(drawn["index"])),
                "total_correlation": {k: v for k, v in total.items() if k != "log_marginals"},
                "floor": mixture_lib.floor_record(alpha),
                "row_scores": str(target.relative_to(self.context.run_root).as_posix()),
                "diagnostic": report_lib.mixture_diagnostic(
                    vectors["parent_log_conditionals"], vectors["policy_log_conditionals"],
                    (vectors["posterior_log_weights"]["parent"],
                     vectors["posterior_log_weights"]["policy"]),
                    label=f"mixture alpha={alpha:g} over {entry['name']}")}

    # -- discriminative and statistical comparators ----------------------
    def fit_comparators(self):
        """CNN single seeds, the CNN ensemble, the additive model and the pairwise model.

        All four are rebuilt per challenge population using only allowed
        training rows, and every selection -- the CNN epoch, the additive ridge,
        the interaction classifier's regularization -- reads ``C`` and nothing
        else. Their scores are ranking scores: the CNN is trained on all three
        classes, which is a different training signal from high-only stage 1,
        and none of these is ever treated as a generative probability.
        """
        from . import her2_baselines as baselines
        record = self.context.path("comparators.json")
        if record.is_file():
            document = paths.read_json(record)
            for regime in list(document["regimes"]):
                self._load_comparator_models(regime)
            return document
        block_data = self.load_populations()
        val_frame = self.split("val")
        calibration_rows = np.asarray(block_data["calibration_rows"])
        calibration_index = self.index("val")[calibration_rows]
        calibration_classes = np.asarray(val_frame["class"])[calibration_rows]
        calibration_targets = baselines.class_targets(calibration_classes)
        train_frame = self.split("train")
        out = {}
        # The ORIGINAL split gets its own comparator set: the challenge table is
        # not the Block-A table, and an original-split CNN/additive/pairwise
        # comparison was simply missing. Its development population is the
        # original validation split, which is disclosed as containing E.
        populations = {"original_split": (np.arange(len(train_frame)),
                                          self.index("val"),
                                          np.asarray(val_frame["class"]))}
        for regime in self.config["block_b"]["regimes"]:
            populations[regime] = (
                np.asarray(block_data["purge_rows" if regime == "purge" else "match_rows"]),
                calibration_index, calibration_classes)
        for regime, (rows, development_index, development_classes) in populations.items():
            index = self.index("train")[rows]
            classes = np.asarray(train_frame["class"])[rows]
            out[regime] = self._fit_one_comparator_set(
                regime, index=index, classes=classes, development_index=development_index,
                development_classes=development_classes,
                development_targets=(calibration_targets if regime != "original_split"
                                     else baselines.class_targets(development_classes)))
        document = {"schema_version": contract.NF_SCHEMA, "record_kind": "comparators",
                    "regimes": out,
                    "populations": {"original_split": ("the full original training rows, with the "
                                                       "original validation split for selection. "
                                                       "That population contains E and the "
                                                       "disclosure stands."),
                                    "purge": "T_purge only; selection on C",
                                    "matched": "T_match only; selection on C"},
                    "departures": list(baselines.CNN_DEPARTURES_FROM_UPSTREAM)}
        paths.write_json(record, _jsonable(document))
        return document

    def _fit_one_comparator_set(self, regime, *, index, classes, development_index,
                                development_classes, development_targets):
        """One regime's four comparators, every model state written to disk."""
        import torch

        from . import her2_baselines as baselines
        from . import her2_nf_reuse as reuse
        settings = dict(self.config["comparators"]["cnn"])
        targets = baselines.class_targets(classes)
        directory = self.context.path("comparators", regime)
        directory.mkdir(parents=True, exist_ok=True)
        models, seeds = [], []
        historical = {int(entry["seed"]): entry
                      for entry in (self.config["comparators"].get("historical") or [])}
        for seed in self.config["comparators"]["cnn_seeds"]:
            target = directory / f"cnn_seed{seed}.pt"
            model = baselines.MasonCNN().to(self.device)
            if target.is_file():
                state = torch.load(target, map_location="cpu", weights_only=True)
                model.load_state_dict(state["state"], strict=True)
                models.append(model.to(self.device))
                seeds.append(dict(state["record"], reloaded=True))
                continue
            entry = historical.get(int(seed))
            if regime == "original_split" and entry:
                # The historical CNNs were fitted on exactly this population and
                # are hash-verified. Refitting them would be a different model
                # with the same name.
                block = reuse.load_historical_cnn(
                    self.context.repository_root / entry["logical"],
                    expected_sha256=entry.get("sha256"))
                model.load_state_dict(block["state"], strict=True)
                history = {"seed": int(seed), "source": "historical",
                           "file_sha256": block["file_sha256"],
                           "selection_provenance": block["selection_provenance"]}
            else:
                fitted, record = baselines.train_cnn(
                    index, targets, development_index, development_targets, settings,
                    seed=int(seed), device=self.device, directory=directory)
                model = fitted
                history = {"seed": int(seed), "source": "fitted",
                           **{k: v for k, v in record.items() if k != "history"}}
            torch.save({"schema_version": contract.NF_SCHEMA, "state": model.state_dict(),
                        "record": history}, target)
            models.append(model.to(self.device))
            seeds.append(history)
        additive, additive_record = baselines.fit_additive_linear(
            index, targets, ridge=float(self.config["comparators"]["additive_ridge"]))
        torch.save({"schema_version": contract.NF_SCHEMA, "weight": additive.weight,
                    "bias": additive.bias, "record": additive_record},
                   directory / "additive.pt")
        classifier = dict(self.config["coupling"]["classifier"])
        pairwise = coupling.fit_interaction_classifier(
            index, classes, development_index=development_index,
            development_classes=development_classes, pairwise=True,
            regularization_grid=tuple(classifier["regularization_grid"]),
            max_iterations=int(classifier["max_iterations"]))
        main_only = coupling.fit_interaction_classifier(
            index, classes, development_index=development_index,
            development_classes=development_classes, pairwise=False,
            regularization_grid=tuple(classifier["regularization_grid"]),
            max_iterations=int(classifier["max_iterations"]))
        for name, fit in (("pairwise", pairwise), ("main_only", main_only)):
            _save_sklearn(directory / f"{name}.npz", fit)
        self._cache[f"comparators:{regime}"] = {
            "cnn_models": models, "additive": additive, "pairwise": pairwise,
            "main_only": main_only}
        return {"training_rows": int(index.shape[0]),
                "cnn_seeds": seeds,
                "cnn_ensemble": {"members": len(models),
                                 "status": "a separate entry from the single-model mean, never "
                                           "averaged with it"},
                "additive": additive_record,
                "pairwise": {k: v for k, v in pairwise.items() if k != "model"},
                "main_effects_only": {k: v for k, v in main_only.items() if k != "model"},
                "persisted": sorted(p.name for p in directory.glob("*.pt"))
                + sorted(p.name for p in directory.glob("*.npz")),
                "selection": ("every choice reads the declared development population for this "
                              "regime: C for the challenge regimes, the original validation split "
                              "for the original-split set"),
                "status": ("discriminative ranking comparators. The CNN's all-class supervision "
                           "is a different training signal from high-only stage 1, and none of "
                           "these scores is a generative probability.")}

    def _load_comparator_models(self, regime):
        """Reload the persisted comparator states for a regime.

        Storing the fitted models only in memory and returning early from the
        cached JSON on restart meant ``comparator_scores_on`` demanded a cache
        that the restart had just skipped building: the comparator half of the
        challenge table simply disappeared after any interruption.
        """
        import torch

        from . import her2_baselines as baselines
        key = f"comparators:{regime}"
        if key in self._cache:
            return self._cache[key]
        directory = self.context.path("comparators", regime)
        if not directory.is_dir():
            return None
        models = []
        for seed in self.config["comparators"]["cnn_seeds"]:
            target = directory / f"cnn_seed{seed}.pt"
            if not target.is_file():
                return None
            state = torch.load(target, map_location="cpu", weights_only=True)
            model = baselines.MasonCNN()
            model.load_state_dict(state["state"], strict=True)
            models.append(model.to(self.device))
        additive_path = directory / "additive.pt"
        if not additive_path.is_file():
            return None
        stored = torch.load(additive_path, map_location="cpu", weights_only=True)
        additive = baselines.AdditiveLinear(stored["weight"], stored["bias"])
        pairwise = _load_sklearn(directory / "pairwise.npz")
        main_only = _load_sklearn(directory / "main_only.npz")
        if pairwise is None or main_only is None:
            return None
        self._cache[key] = {"cnn_models": models, "additive": additive,
                            "pairwise": pairwise, "main_only": main_only}
        return self._cache[key]

    def comparator_scores_on(self, regime, index):
        """Every comparator's ranking score on a frozen panel. Scored after the freeze."""
        from . import her2_baselines as baselines
        block = self._cache.get(f"comparators:{regime}") or self._load_comparator_models(regime)
        require(block is not None,
                f"the comparators for {regime!r} are neither fitted in this session nor readable "
                f"from {self.context.path('comparators', regime)}. Fit them first so the "
                "selection order stays visible.")
        ensemble, per_seed = baselines.ensemble_probabilities(block["cnn_models"], index,
                                                              device=self.device)
        high = baselines.CLASS_ORDER.index("high")
        return {"cnn_single": [values[:, high] for values in per_seed],
                "cnn_ensemble": ensemble[:, high],
                "additive": block["additive"].positive_scores(index),
                "pairwise": coupling.classifier_scores(block["pairwise"], index)["p_high"],
                "main_effects_only": coupling.classifier_scores(block["main_only"],
                                                               index)["p_high"],
                "status": "ranking scores only; never exponentiated and never a yield input"}

    # -- the challenge estimands, scored last ----------------------------
    def evaluate_challenge(self):
        """Score ``E`` with every Block-B endpoint, after the choices are frozen.

        This is the only place a challenge model touches ``E``. The endpoint is
        the configured ``u1000``, frozen by the source snapshot before any of
        this ran, and the parent selection, the monitoring and the comparator
        regularization all used ``C``.

        The primary endpoint is high-versus-rest AP on the fixed ``E``: after an
        exact purge every ``E`` row is distance >= 3 from ``T_purge``, so the old
        d1/d2/far macro is a different statistic and appears only as a labelled
        diagnostic in the ORIGINAL bins.
        """
        freeze = self._finalist_freeze()          # refuses to run before the freeze exists
        from . import her2_nf_campaign as campaign
        arms = campaign.block_b_arms(self.config, campaign._frozen_if_available(self.context))
        dpo_arm = next((arm["id"] for arm in arms if arm["task"] == "dpo"), "DPO_FKL")
        block_data = self.load_populations()
        val_frame = self.split("val")
        rows = np.asarray(block_data["evaluation_rows"])
        index = self.index("val")[rows]
        classes = np.asarray(val_frame["class"])[rows]
        positive = classes == "high"
        original = np.asarray(block_data["manifest"]["evaluation_panel"]
                              ["original_train_distance"])
        bins = np.where(original < 0, ">=3", original.astype(str))
        endpoint = int(self.config["block_b"]["endpoint_updates"][0])
        records, per_arm, mixture_curves, decompositions = [], {}, [], []
        for regime in self.config["block_b"]["regimes"]:
            for seed in self.config["block_b"]["parent_seeds"]:
                parent = self._challenge_parent_if_fitted(regime, seed)
                parent_density = None
                parent_marginals = self._screen_marginals(f"screen_B_{regime}_parent_seed{seed}")
                entries = [("parent", parent, 0)] if parent else []
                for arm in arms:
                    checkpoint = self._endpoint_checkpoint(
                        f"B_{regime}_{arm['id']}_seed{seed}", endpoint)
                    if checkpoint is not None:
                        entries.append((arm["id"], checkpoint, endpoint))
                for name, checkpoint, update in entries:
                    scored = self.policy(checkpoint).score(
                        index, batch_size=int(self.config["inference"]["score_batch_size"]))
                    if name == "parent":
                        parent_density = np.asarray(scored["sum_log_probability"], dtype=np.float64)
                    target = self.context.path(
                        "row_scores", f"challenge_{regime}_{name}_seed{seed}_u{update}.npz")
                    paths.write_arrays(target, {
                        "row_id": contract.index_hashes(index).astype("S64"),
                        "source_split": np.asarray("val:E"), "class": classes.astype("S4"),
                        "model": np.asarray(f"B_{regime}_{name}_seed{seed}@u{update}"),
                        "model_state_sha256": np.asarray(nf_reuse.checkpoint_metadata(checkpoint)["state_sha256"]),
                        "evaluation_rows": rows,
                        "sum_log_probability": np.asarray(scored["sum_log_probability"],
                                                          dtype=np.float64),
                        "mean_log_probability": np.asarray(scored["mean_log_probability"],
                                                           dtype=np.float64),
                        "positive": positive.astype(np.int8),
                        "original_train_distance": np.asarray(original)})
                    block = {
                        "regime": regime, "seed": int(seed), "arm": name, "update": update,
                        "primary": metrics.rank_block(scored["mean_log_probability"], positive),
                        "high_versus_mid": _versus(scored["mean_log_probability"], classes, "mid"),
                        "high_versus_low": _versus(scored["mean_log_probability"], classes, "low"),
                        "original_bin_diagnostics": metrics.stratified_rank_block(
                            scored["mean_log_probability"], positive, bins,
                            categories=proximity.ORIGINAL_STRATA),
                        "class_mass": _class_mass(scored["sum_log_probability"], classes),
                        "yield": metrics.yield_curve(
                            scored["sum_log_probability"][positive]),
                        "row_scores": str(target.relative_to(self.context.run_root).as_posix()),
                        "panel_note": ("high identities of E only. Its absolute yield is NOT "
                                       "comparable with the full original-validation yield.")}
                    records.append(block)
                    per_arm.setdefault(f"{name}@{regime}", {})[int(seed)] = \
                        block["primary"]["average_precision"]
                    if name != "parent" and parent_density is not None and parent_marginals is not None:
                        marginals = self._screen_marginals(f"screen_B_{regime}_{name}@u{update}_seed{seed}")
                        if marginals is not None:
                            decomposition = coupling.decomposition_report(
                                index, parent_density, scored["sum_log_probability"],
                                parent_marginals, marginals, positive, classes)
                            decompositions.append(dict(decomposition, regime=regime, seed=int(seed),
                                                       arm=name, update=update))
                    if name == "IPO_0" and parent_density is not None:
                        density = np.asarray(scored["sum_log_probability"], dtype=np.float64)
                        mixed = mixture_lib.mixture_log_probability(parent_density, density,
                                                                     mixture_lib.PRIMARY_ALPHA)
                        mix_target = self.context.path("row_scores",
                            f"challenge_{regime}_mixture089_seed{seed}_u{update}.npz")
                        paths.write_arrays(mix_target, {
                            "row_id": contract.index_hashes(index).astype("S64"),
                            "source_split": np.asarray("val:E"), "class": classes.astype("S4"),
                            "parent_sum_log_probability": parent_density,
                            "policy_sum_log_probability": density, "sum_log_probability": mixed,
                            "mean_log_probability": mixed / 10, "alpha": np.asarray(.89),
                            "parent_checkpoint": np.asarray(str(parent)),
                            "policy_checkpoint": np.asarray(str(checkpoint))})
                        mixed_record = {
                            "regime": regime, "seed": int(seed), "arm": "mixture089", "update": update,
                            "primary": metrics.rank_block(mixed / 10, positive),
                            "high_versus_mid": _versus(mixed / 10, classes, "mid"),
                            "high_versus_low": _versus(mixed / 10, classes, "low"),
                            "class_mass": _class_mass(mixed, classes),
                            "yield": metrics.yield_curve(mixed[positive]),
                            "row_scores": mix_target.relative_to(self.context.run_root).as_posix(),
                            "alpha": mixture_lib.PRIMARY_ALPHA,
                            "panel_note": "fixed E with this regime's own freshly fitted parent"}
                        records.append(mixed_record)
                        per_arm.setdefault(f"mixture089@{regime}", {})[int(seed)] = \
                            mixed_record["primary"]["average_precision"]
                        curve = mixture_lib.alpha_curve(
                            parent_density, density, labels=positive, yield_budgets=metrics.YIELD_BUDGETS,
                            yield_function=metrics.expected_distinct_yield)
                        mixture_curves.append(dict(curve, label=f"B_{regime}_IPO_0_seed{seed}@u{update}",
                                                    regime=regime, seed=int(seed), panel="fixed E"))
        comparators = []
        for regime in self.config["block_b"]["regimes"]:
            if (f"comparators:{regime}" not in self._cache
                    and self._load_comparator_models(regime) is None):
                continue
            scores = self.comparator_scores_on(regime, index)
            comparators.append({
                "regime": regime,
                "cnn_single_mean": float(np.mean([
                    metrics.average_precision(values, positive)
                    for values in scores["cnn_single"]])),
                "cnn_single_per_seed": [metrics.average_precision(values, positive)
                                        for values in scores["cnn_single"]],
                "cnn_ensemble": metrics.rank_block(scores["cnn_ensemble"], positive),
                "additive": metrics.rank_block(scores["additive"], positive),
                "pairwise": metrics.rank_block(scores["pairwise"], positive),
                "main_effects_only": metrics.rank_block(scores["main_effects_only"], positive),
                "reading": ("whether the pairwise model beats the additive one on the SAME rows "
                            "is the task-relevant dependence question. A higher MI sum is not "
                            "that answer.")})
        return {"schema_version": contract.NF_SCHEMA, "record_kind": "challenge_evaluation",
                "comparators": comparators,
                "endpoint": endpoint, "panel_rows": int(rows.size),
                "prevalence": float(positive.mean()), "records": records,
                "per_arm_average_precision": per_arm,
                "parent_contrasts": _parent_ap_contrasts(per_arm),
                "mixture_curves": mixture_curves, "decomposition": decompositions,
                "difference_in_differences": _difference_in_differences(per_arm, dpo_arm=dpo_arm),
                "freeze": {"named_checkpoints": len(freeze.get("named_checkpoints") or [])},
                "ordering": ("E is scored only here, after the challenge choices and the endpoint "
                             "were frozen. No challenge-model score on E influenced any "
                             "selection."),
                "limitation": ("matching class and WT distance controls quantity and coarse "
                               "composition; it does not randomize every residue feature or "
                               "library provenance.")}

    def _challenge_parent_if_fitted(self, regime, seed):
        record = self.context.path("parents", f"{regime}_seed{seed}", "selection.json")
        return Path(paths.read_json(record)["selected"]["path"]) if record.is_file() else None

    def report_evidence(self):
        """Every number the report needs that only the runtime can produce.

        Scored over the WHOLE checkpoint registry -- every arm at every required
        endpoint, the parents, the reused historical IPO0/IPO+FKL endpoints and
        the continued-SFT controls -- not one arm at one endpoint. Each model's
        row-indexed score vector is persisted so the contrasts can be recomputed
        against the same rows, the class masses are exact sums of ``exp(ell)``
        over the declared assayed panel, and the coupling decomposition runs from
        the generated-bank marginals that the coupling stage retained.
        """
        registry = self.checkpoint_registry()
        evidence = {"metric_records": {}, "yield_curves": {}, "coupling": [],
                    "decomposition": [], "class_mass": {}, "challenge": None,
                    "comparators": None, "registry": registry, "row_scores": {},
                    "preservation_contrasts": {}, "mixtures": None,
                    "comparator_rankings": None, "rankings": {}}
        if self.context.path("split_manifest.json").is_file():
            # Coerced on the way in: the report is written with allow_nan=False
            # json, and a numpy scalar reaching it raises at the very end of the
            # flight rather than here.
            evidence["comparators"] = _jsonable(self.fit_comparators())
        val_frame, val_index = self.split("val"), self.index("val")
        classes = np.asarray(val_frame["class"])
        positive = classes == "high"
        strata = self.original_strata()["labels"]
        batch = int(self.config["inference"]["score_batch_size"])
        parents = {}
        row_ids = contract.index_hashes(val_index).astype("S64")
        ordered = sorted(registry["entries"].items(),
                         key=lambda item: (item[1]["kind"] != "parent", item[0]))
        for name, entry in ordered:
            if entry["status"] != "present" or entry.get("regime") is not None:
                continue                      # Block B is scored on E, not on old validation
            scored = self.policy(Path(entry["checkpoint"])).score(val_index, batch_size=batch)
            summed = np.asarray(scored["sum_log_probability"], dtype=np.float64)
            ranking = np.asarray(scored["mean_log_probability"], dtype=np.float64)
            target = self.context.path("row_scores", f"{_slug(name)}.npz")
            paths.write_arrays(target, {
                "row_id": row_ids, "source_split": np.asarray("val"),
                "model": np.asarray(name), "class": classes.astype("S4"),
                "sum_log_probability": summed, "mean_log_probability": ranking,
                "positive": positive.astype(np.int8), "original_train_distance_label":
                    np.asarray([str(value) for value in strata], dtype="<U8")})
            evidence["row_scores"][name] = str(
                target.relative_to(self.context.run_root).as_posix())
            block = metrics.stratified_rank_block(ranking, positive, strata,
                                                  categories=proximity.ORIGINAL_STRATA)
            evidence["rankings"][name] = block
            key = f"{entry['arm']}@{entry['update']}"
            evidence["metric_records"].setdefault(key, {})[int(entry["seed"])] = \
                block["macro_average_precision"]
            evidence["class_mass"][name] = _class_mass(summed, classes)
            if entry["kind"] == "parent":
                parents[int(entry["seed"])] = summed
            require(int(entry["seed"]) in parents,
                    f"{name}: the matching parent scores must exist before yield comparison")
            evidence["yield_curves"][name] = {
                "log_probabilities": summed[positive],
                "control_log_probabilities": parents[int(entry["seed"])][positive],
                "control_label": f"parent_seed{entry['seed']}"}
            if entry["arm"] == "IPO_0":
                mixed = mixture_lib.mixture_log_probability(
                    parents[int(entry["seed"])], summed, mixture_lib.PRIMARY_ALPHA)
                mixed_rank = metrics.stratified_rank_block(
                    mixed, positive, strata, categories=proximity.ORIGINAL_STRATA)
                evidence["metric_records"].setdefault(
                    f"mixture089@{entry['update']}", {})[int(entry["seed"])] = \
                    mixed_rank["macro_average_precision"]
                evidence["rankings"][f"mixture089::{name}"] = mixed_rank
        evidence["mixtures"] = self._mixture_evidence(registry, positive=positive,
                                                      parents=parents)
        evidence["decomposition"] = self._decomposition_evidence(
            registry, val_index=val_index, positive=positive, classes=classes)
        evidence["comparator_rankings"] = self._comparator_rankings(val_index, positive)
        evidence["preservation_contrasts"] = _read_optional_json(
            self.context.path("preservation_audit.json"))
        coupling_path = self.context.path("coupling.json")
        if coupling_path.is_file():
            document = paths.read_json(coupling_path)
            evidence["coupling"] = (document["screens"] + document["finalists"]
                                    + document.get("mixture_diagnostics", []))
        if self.context.path("finalist_freeze.json").is_file():
            evidence["challenge"] = _jsonable(self.evaluate_challenge())
            evidence["mixtures"]["curves"].extend(evidence["challenge"].get("mixture_curves") or [])
            evidence["decomposition"].extend(evidence["challenge"].get("decomposition") or [])
            paths.write_json(self.context.path("challenge_evaluation.json"),
                             evidence["challenge"])
        paths.write_json(self.context.path("report_evidence.json"),
                         _jsonable({key: value for key, value in evidence.items()
                                    if key != "yield_curves"}))
        return evidence

    def _mixture_evidence(self, registry, *, positive, parents):
        """The full alpha grid for every eligible checkpoint, from the saved row scores.

        Every historical IPO0 seed at both required endpoints, every eligible new
        DPO0 checkpoint and the challenge IPO0 checkpoints, not one hand-listed
        pair. Computed from the persisted row scores, so this costs no extra
        forward pass.
        """
        curves = []
        for name, entry in sorted(registry["entries"].items()):
            if (entry["status"] != "present" or entry["arm"] not in ("IPO_0", "DPO_0")
                    or entry.get("regime") is not None):
                continue
            stored = self.context.path("row_scores", f"{_slug(name)}.npz")
            parent = parents.get(int(entry["seed"]))
            if not stored.is_file() or parent is None:
                continue
            policy_scores = np.asarray(paths.read_arrays(stored)["sum_log_probability"],
                                       dtype=np.float64)
            curve = mixture_lib.alpha_curve(parent, policy_scores, labels=positive,
                                            yield_budgets=metrics.YIELD_BUDGETS,
                                            yield_function=metrics.expected_distinct_yield)
            parent_yield = metrics.expected_distinct_yield(parent[positive], 10_000)
            policy_yield = metrics.expected_distinct_yield(policy_scores[positive], 10_000)
            curve.update(label=name, seed=int(entry["seed"]), arm=entry["arm"],
                         update=int(entry["update"]), regime=entry.get("regime"),
                         parent_yield_10k=parent_yield, policy_yield_10k=policy_yield,
                         concavity_bound_at_primary=metrics.mixture_yield_bound(
                             parent_yield, policy_yield, mixture_lib.PRIMARY_ALPHA))
            curves.append(curve)
        return {"curves": curves, "grid": list(mixture_lib.ALPHA_GRID),
                "coverage": ("every eligible IPO0 and DPO0 checkpoint in the registry, at every "
                             "required endpoint. Each grid point is persisted; no alpha is "
                             "presented as prespecified.")}

    def _decomposition_evidence(self, registry, *, val_index, positive, classes):
        """Full / marginal-only / residual ranking and the two declared hybrids.

        The generated marginals come from each model's own retained screen bank,
        which the coupling stage wrote. Without them there is no ``g``, and the
        decomposition section stayed empty in every report.
        """
        blocks = []
        parents = {}
        for name, entry in sorted(registry["entries"].items()):
            if entry["kind"] != "parent" or entry["status"] != "present":
                continue
            stored = self._screen_marginals(f"screen_{name}")
            scores = self.context.path("row_scores", f"{_slug(name)}.npz")
            if stored is None or not scores.is_file():
                continue
            parents[int(entry["seed"])] = {
                "log_marginals": stored,
                "scores": np.asarray(paths.read_arrays(scores)["sum_log_probability"],
                                     dtype=np.float64)}
        for name, entry in sorted(registry["entries"].items()):
            if (entry["status"] != "present" or entry["kind"] == "parent"
                    or entry.get("regime") is not None):
                continue
            parent = parents.get(int(entry["seed"]))
            stored = self._screen_marginals(f"screen_{name}")
            scores = self.context.path("row_scores", f"{_slug(name)}.npz")
            if parent is None or stored is None or not scores.is_file():
                continue
            policy_scores = np.asarray(paths.read_arrays(scores)["sum_log_probability"],
                                       dtype=np.float64)
            block = coupling.decomposition_report(
                val_index, parent["scores"], policy_scores, parent["log_marginals"], stored,
                positive, classes)
            block.update(model=name, arm=entry["arm"], seed=int(entry["seed"]),
                         update=int(entry["update"]),
                         control=f"parent_seed{entry['seed']}")
            blocks.append(block)
        return blocks

    def _screen_marginals(self, name):
        """A model's generated log marginals, from the screen bank the coupling stage wrote."""
        target = self.context.path("coupling", f"{_slug(name)}_q_generation_screen_0.npz")
        if not target.is_file():
            return None
        return np.asarray(paths.read_arrays(target)["log_marginals"], dtype=np.float64)

    def _comparator_rankings(self, val_index, positive):
        """The original-split comparator table: CNN single/ensemble, additive, pairwise."""
        block = self._cache.get("comparators:original_split") or self._load_comparator_models(
            "original_split")
        if block is None:
            return None
        scores = self.comparator_scores_on("original_split", val_index)
        return {"population": ("the original validation split, which contains E. That exposure is "
                               "disclosed and is why the challenge panel is scored separately."),
                "cnn_single_per_seed": [metrics.average_precision(values, positive)
                                        for values in scores["cnn_single"]],
                "cnn_single_mean": float(np.mean([metrics.average_precision(values, positive)
                                                  for values in scores["cnn_single"]])),
                "cnn_ensemble": metrics.rank_block(scores["cnn_ensemble"], positive),
                "additive": metrics.rank_block(scores["additive"], positive),
                "pairwise": metrics.rank_block(scores["pairwise"], positive),
                "main_effects_only": metrics.rank_block(scores["main_effects_only"], positive),
                "reading": ("whether the pairwise model beats the additive one on the SAME rows "
                            "is the task-relevant dependence question. A higher MI sum is not "
                            "that answer, and the ensemble is a separate entry from the "
                            "single-model mean.")}

    # -- the miniature end-to-end smoke path -----------------------------
    def smoke(self, *, namespace=None):
        """A miniature but COMPLETE native pass: train, interrupt, resume, score, report.

        Everything runs through the production entry points at small bank sizes,
        inside an explicit smoke namespace that no production stage ever reads.
        Nothing is deleted: a namespace that already carries results is a
        collision, and the run refuses rather than erasing an existing one to
        make itself pass.

        The pass covers, in order:

        * one trajectory to a declared endpoint, with intermediate checkpoints;
        * the same trajectory again with a REAL interruption injected inside the
          endpoint publication, resumed to the same weights, the same optimizer,
          the same scheduler, the same RNG, the same cadence counters and the
          same published endpoint;
        * a deliberate gate stop, a wrong-identity refusal, a corrupted-payload
          refusal, a truncated-journal repair and the forbidden-row guard;
        * scoring, the class masses, one comparator fit, a coupling screen with
          retained conditionals, and a numeric report block.

        Synthetic orchestration cannot catch a native API error -- the wrong
        ``raw_root``, an unreadable historical schema, a bank whose shape moved.
        This is the cheapest place those surface.
        """
        started = time.perf_counter()
        checks, failures = [], []

        def record(name, passed, detail=None):
            checks.append({"check": name, "passed": bool(passed), "detail": detail})
            if not passed:
                failures.append(name)

        space = str(namespace or self.config["smoke"].get("namespace") or "smoke")
        root = self.context.path(space)
        require(not root.exists() or not any(root.iterdir()),
                f"{root} already holds a smoke run"
                + (" that completed" if (root / "smoke_complete.json").is_file() else "")
                + ". This path never erases an existing run to make itself pass: choose a fresh "
                  "namespace, or move that directory aside deliberately if you want to repeat it. "
                  "Nothing here is deleted for you.")
        seed = int(self.config["block_a"]["parent_seeds"][0])
        updates = int(self.config["smoke"]["updates"])
        half = max(1, updates // 2)
        small = {"bank_rows": int(self.config["smoke"].get("bank_rows", 256)),
                 "monitor_rows": int(self.config["smoke"].get("monitor_rows", 128)),
                 "namespace": space,
                 # ONE fixed max-horizon stream for every arm in this namespace,
                 # so the interrupted run and the reference run resolve the same
                 # stream and a strict resume has no reason to refuse.
                 "max_horizon": updates}
        arm = {"block": "A", "regime": "original_split", "seed": seed, "task": "ipo",
               "preservation": "none", "coefficients": {"tau": 0.1}, "parent_seed": seed}

        reference_dir = root / "reference"
        first = self.run_one_trajectory(trajectory=f"{space}_reference", directory=reference_dir,
                                        updates=updates, endpoints=(half, updates),
                                        checkpoints=(half,), **arm, **small)
        record("end_to_end_completes", first["status"] == "completed", first["status"])
        progress = trajectory_lib.durable_progress(reference_dir)
        record("journals_agree_with_loop", progress["updates"] == updates, progress["updates"])
        record("intermediate_checkpoint_written",
               (reference_dir / f"endpoint_update{half}.pt").is_file()
               or (reference_dir / f"checkpoint_update{half}.pt").is_file(),
               sorted(p.name for p in reference_dir.glob("*.pt")))
        record("every_declared_endpoint_published",
               sorted(int(v) for v in progress["endpoints_reached_updates"]) == [half, updates],
               progress["endpoints_reached_updates"])
        reference_digest = _state_digest(first["policy"])
        analysis = self._smoke_analysis(root, space, first, record)
        reference_document = first["document"]
        del first
        self.release_transient()

        interrupted = self._interrupted_resume(root, space, arm, small, updates=updates,
                                               half=half, record=record,
                                               reference=reference_document,
                                               reference_digest=reference_digest)
        self._refusal_checks(interrupted["directory"], seed, record)

        stopped = self._deliberate_gate_stop(root / "gate", seed, namespace=space)
        record("deliberate_gate_stop", stopped["status"] == trajectory_lib.STATUS_STOPPED,
               stopped["status"])
        record("failed_state_written", (root / "gate" / trajectory_lib.FAILED_STATE).is_file())
        record("stop_is_an_outcome_not_a_failure",
               stopped["status"] != trajectory_lib.STATUS_FAILED
               and stopped.get("stop_reason") is not None, stopped.get("stop_reason"))

        try:
            self.forbidden_evaluation_rows().check(
                [self.split("val").seq.iloc[int(row)]
                 for row in self.load_populations()["evaluation_rows"][:4]],
                where=f"{space}: challenge loader")
            record("evaluation_rows_refused_in_training", False,
                   "an E row resolved inside a challenge population")
        except ValueError:
            record("evaluation_rows_refused_in_training", True)
        except Exception as error:                              # noqa: BLE001 - reported
            record("evaluation_rows_refused_in_training", False,
                   f"the guard could not run: {type(error).__name__}: {error}")

        document = {"schema_version": contract.NF_SCHEMA, "record_kind": "smoke",
                    "namespace": space, "checks": checks, "failures": failures,
                    "passed": not failures, "analysis": analysis,
                    "wall_seconds": time.perf_counter() - started,
                    "scope": ("miniature bank sizes inside an explicit namespace. The SHAPE of "
                              "the workflow is the production one: the same trajectory loop, the "
                              "same resume, the same scoring, comparator, coupling and report "
                              "entry points."),
                    "purpose": ("validate orchestration, the native APIs and the exact negative "
                                "cases BEFORE the launch. An orchestration defect found after "
                                "twenty hours of training is not a cheap one.")}
        paths.write_json(root / "smoke_complete.json", _jsonable(document))
        return document

    def _interrupted_resume(self, root, space, arm, small, *, updates, half, record,
                            reference, reference_digest):
        """Interrupt inside the endpoint publication, resume, and compare everything.

        The injected failure fires AFTER the resume state carrying the endpoint's
        weights is written and BEFORE its checkpoint and journal record exist --
        the exact window in which an endpoint used to be skipped permanently.
        """
        import torch

        directory = root / "interrupted"
        boom = {"count": 0}
        real_endpoint = self._endpoint

        def exploding(policy, target, identity, **kwargs):
            if int(kwargs["update"]) == half and boom["count"] == 0:
                boom["count"] += 1
                raise RuntimeError("injected interruption inside the endpoint publication")
            return real_endpoint(policy, target, identity, **kwargs)

        self._endpoint = exploding
        try:
            self.run_one_trajectory(trajectory=f"{space}_interrupted", directory=directory,
                                    updates=updates, endpoints=(half, updates),
                                    checkpoints=(half,), **arm, **small)
            record("interruption_actually_fired", False, "the injected failure did not raise")
        except RuntimeError as error:
            record("interruption_actually_fired", "injected interruption" in str(error),
                   str(error)[:160])
        finally:
            self._endpoint = real_endpoint
        storage.collect_unused()
        state = storage.load_cpu(directory / trajectory_lib.RESUME_STATE)
        record("pending_endpoint_recorded",
               (state["progress"].get("pending_endpoint") or {}).get("update") == half,
               state["progress"].get("pending_endpoint"))
        del state
        (directory / trajectory_lib.STATUS_JSON).unlink(missing_ok=True)
        resumed = self.run_one_trajectory(trajectory=f"{space}_interrupted", directory=directory,
                                          updates=updates, endpoints=(half, updates),
                                          checkpoints=(half,), **arm, **small)
        document = resumed["document"]
        reference_weights = storage.load_cpu(root / "reference" / f"endpoint_update{updates}.pt",
                                             weights_only=True)
        continuation_agrees = _native_tree_close(resumed["policy"].model.state_dict(),
                                                 reference_weights["state"])
        del reference_weights
        record("resume_reproduces_uninterrupted_weights",
               continuation_agrees,
               {"uninterrupted": reference_digest, "resumed": _state_digest(resumed["policy"]),
                "comparison": "all state tensors, atol=1e-7, rtol=1e-6; exact checkpoint "
                              "restoration is separately verified before continuation. Native "
                              "SDPA reductions need not be bitwise identical across runs."})
        record("interrupted_endpoint_was_published",
               sorted(int(value) for value in document["endpoints_reached"]) == [half, updates],
               sorted(document["endpoints_reached"]))
        record("resume_reconciled_the_pending_endpoint",
               (document.get("resume_reconciliation") or {}).get("published_endpoint") == half,
               document.get("resume_reconciliation"))
        for field in ("checks", "sentinels", "clipped_updates", "stream_position",
                      "replay_position", "exposures", "updates"):
            record(f"resume_matches_uninterrupted_{field}",
                   document.get(field) == reference.get(field),
                   {"uninterrupted": reference.get(field), "resumed": document.get(field)})
        del resumed
        storage.collect_unused()
        after = storage.load_cpu(directory / trajectory_lib.RESUME_STATE)
        before = storage.load_cpu(root / "reference" / trajectory_lib.RESUME_STATE)
        record("resumed_optimizer_matches",
               _native_tree_close(after["optimizer"], before["optimizer"]),
               {"resumed": trajectory_lib._optimizer_step_count(after["optimizer"]),
                "uninterrupted": trajectory_lib._optimizer_step_count(before["optimizer"])})
        record("resumed_scheduler_matches",
               _nested_digest(after["scheduler"]) == _nested_digest(before["scheduler"]),
               {"resumed": after["scheduler"]["last_epoch"],
                "uninterrupted": before["scheduler"]["last_epoch"]})
        record("resumed_rng_matches",
               _nested_digest(after["rng"]) == _nested_digest(before["rng"]))
        record("whole_payload_digest_present", after.get("payload_sha256") is not None)
        del after, before
        storage.collect_unused()
        journal = directory / trajectory_lib.UPDATES_JSONL
        with journal.open("a", encoding="utf-8") as stream:
            stream.write('{"record_kind": "update", "update": 999, "trunc')
        repair = trajectory_lib.repair_truncated_journal(journal)
        record("truncated_journal_repaired_before_append", repair["repaired"], repair)
        record("repaired_journal_reads_clean",
               trajectory_lib.durable_progress(directory)["updates"] == updates)
        return {"directory": directory, "document": document}

    def _refusal_checks(self, directory, seed, record):
        """Wrong identity, tampered weights and a tampered optimizer are all refused."""
        import torch
        checkpoint, _ = self.parent_checkpoint(seed)

        def attempt(identity):
            policy = self.policy(checkpoint)
            optimizer, scheduler = self.optimizer_and_scheduler(policy)
            return trajectory_lib.TrajectoryStateStore(directory, identity=identity).load(
                policy=policy, optimizer=optimizer, scheduler=scheduler)

        try:
            attempt({"trajectory": "someone_else"})
            record("wrong_identity_refused", False, "the load accepted another trajectory's state")
        except ValueError as error:
            record("wrong_identity_refused", True, str(error)[:200])
        target = directory / trajectory_lib.RESUME_STATE
        original = torch.load(target, map_location="cpu", weights_only=False)
        identity = dict(original["identity"])
        tampered = dict(original)
        key = sorted(tampered["state"])[0]
        tampered["state"] = dict(tampered["state"])
        tampered["state"][key] = tampered["state"][key] + 1.0
        torch.save(tampered, target)
        try:
            attempt(identity)
            record("corrupt_weights_refused", False, "a tampered state was resumed from")
        except ValueError as error:
            record("corrupt_weights_refused", True, str(error)[:200])
        # A damaged OPTIMIZER with intact weights: the failure a model-only
        # digest cannot see, and the reason the payload digest exists.
        tampered = dict(original)
        tampered["scheduler"] = dict(tampered["scheduler"])
        tampered["scheduler"]["last_epoch"] = int(tampered["scheduler"]["last_epoch"]) + 7
        torch.save(tampered, target)
        try:
            attempt(identity)
            record("corrupt_optimizer_half_refused", False,
                   "a state with intact weights and a tampered scheduler was resumed from")
        except ValueError as error:
            record("corrupt_optimizer_half_refused", True, str(error)[:200])
        torch.save(original, target)
        return True

    def _smoke_analysis(self, root, space, first, record):
        """Score, decompose, screen and report -- the analysis half, at smoke size."""
        rows = int(self.config["smoke"].get("analysis_rows", 256))
        val_frame, val_index = self.split("val"), self.index("val")[:rows]
        classes = np.asarray(val_frame["class"])[:rows]
        positive = classes == "high"
        policy = first["policy"]
        scored = policy.score(val_index,
                              batch_size=int(self.config["inference"]["score_batch_size"]))
        summed = np.asarray(scored["sum_log_probability"], dtype=np.float64)
        ranking = np.asarray(scored["mean_log_probability"], dtype=np.float64)
        target = root / "row_scores.npz"
        paths.write_arrays(target, {"sum_log_probability": summed,
                                    "mean_log_probability": ranking,
                                    "positive": positive.astype(np.int8)})
        record("row_scores_are_numeric_and_finite",
               bool(np.isfinite(summed).all()) and summed.size == rows, int(summed.size))
        ranking_block = metrics.rank_block(ranking, positive)
        record("ranking_metric_is_a_number",
               isinstance(ranking_block.get("average_precision"), float),
               ranking_block.get("average_precision"))
        mass = _class_mass(summed, classes)
        record("class_mass_is_exact_and_positive",
               all(block["mass"] >= 0 for name, block in mass.items()
                   if not name.startswith("_")), mass["_panel"]["mass"])
        parent_policy = self.policy(self.parent_checkpoint(
            int(self.config["block_a"]["parent_seeds"][0]))[0])
        parent_scores = np.asarray(parent_policy.score(
            val_index, batch_size=int(self.config["inference"]["score_batch_size"])
        )["sum_log_probability"], dtype=np.float64)
        curve = mixture_lib.alpha_curve(parent_scores, summed, labels=positive,
                                        yield_budgets=(10_000,),
                                        yield_function=metrics.expected_distinct_yield)
        record("mixture_curve_covers_the_declared_grid",
               len(curve["points"]) == len(mixture_lib.ALPHA_GRID), len(curve["points"]))
        draws = int(self.config["smoke"].get("generation_draws", 256))
        lineage = self.stream_seed(domain="smoke", role="screen", model=f"{space}_reference")
        bank, bank_document = nf_banks.draw_bank(
            policy, role="q_generation_screen", parent_id=f"{space}::reference",
            parent_state_sha256=_state_digest(policy), seed_lineage_record=lineage, rows=draws,
            retain_conditionals=True,
            batch_size=int(self.config["inference"]["sample_batch_size"]),
            conditional_batch=int(self.config["inference"]["conditional_batch_size"]))
        total = coupling.total_correlation(bank.conditionals)
        pairwise = coupling.pairwise_mutual_information(bank.index)
        parent_lineage = self.stream_seed(domain="smoke", role="parent_marginals", model=space)
        parent_bank, parent_bank_document = nf_banks.draw_bank(
            parent_policy, role="q_generation_screen", parent_id=f"{space}::parent",
            parent_state_sha256=_state_digest(parent_policy), seed_lineage_record=parent_lineage,
            rows=draws, retain_conditionals=True,
            batch_size=int(self.config["inference"]["sample_batch_size"]),
            conditional_batch=int(self.config["inference"]["conditional_batch_size"]))
        parent_total = coupling.total_correlation(parent_bank.conditionals)
        record("independent_parent_and_policy_marginal_banks",
               parent_lineage["seed"] != lineage["seed"],
               {"parent_seed": parent_lineage["seed"], "policy_seed": lineage["seed"]})
        record("coupling_statistics_are_finite",
               np.isfinite(total["total_correlation"]) and np.isfinite(pairwise["sum"]),
               {"total_correlation": total["total_correlation"], "pairwise_sum": pairwise["sum"]})
        decomposition = coupling.decomposition_report(
            val_index, parent_scores, summed, parent_total["log_marginals"], total["log_marginals"],
            positive, classes)
        record("decomposition_identity_holds",
               all(abs(block["additivity_residual"]) < 1e-9
                   for block in decomposition["within_class_mean_changes"].values()),
               {name: block["additivity_residual"]
                for name, block in decomposition["within_class_mean_changes"].items()})
        drift = self._drift_block(parent_policy, policy, parent_bank.index,
                                  rows=min(draws, 64))
        record("drift_satisfies_B_le_T", bool(drift["B_le_T_holds"]), drift["sum_T"])
        comparator = self._comparator_probe(rows=int(self.config["smoke"].get(
            "comparator_rows", 600)))
        record("comparator_fit_selects_a_regularization",
               comparator.get("fit") is not None, comparator)
        document = {"rows": rows, "ranking": ranking_block, "class_mass": mass,
                    "mixture_curve": curve, "coupling": {
                        "total_correlation": {k: v for k, v in total.items()
                                              if k != "log_marginals"},
                        "pairwise_mi_sum": pairwise["sum"], "bank": bank_document,
                        "parent_bank": parent_bank_document},
                    "decomposition": decomposition, "parent_context_drift": drift,
                    "comparator": comparator,
                    "row_scores": str(target.relative_to(self.context.run_root).as_posix()),
                    "basis": ("real numbers from the real model at smoke sizes. Nothing here is "
                              "a production measurement and nothing here is reported as one.")}
        paths.write_json(root / "smoke_analysis.json", _jsonable(document))
        record("numeric_report_block_is_nonempty",
               bool(document["ranking"]) and bool(document["class_mass"])
               and bool(document["mixture_curve"]["points"]))
        return document

    def _deliberate_gate_stop(self, directory, seed, *, namespace="smoke"):
        """Force a breach with an impossible reference; the arm must stop, not crash."""
        checkpoint, entry = self.parent_checkpoint(seed)
        parent_policy = self.policy(checkpoint)
        population = preferences.build_population(self.split("train"), "train")
        pairing = preferences.PreferencePairing(
            population, seed=int(self.config["block_a"]["pairing"]["seed_base"]) + int(seed))
        pairs = pairing.fixed_validation_pairs(count=monitor_lib.SENTINEL_PAIRS)
        parent_chosen = np.asarray(preferences.score_sequences(parent_policy,
                                                               pairs["chosen_index"],
                                                               progress_every=0))
        gate = monitor_lib.HighRowGate(row_index=pairs["chosen_index"],
                                       parent_log_probability=parent_chosen + 10.0,
                                       threshold=1.0, label="deliberate breach probe")
        policy = self.policy(checkpoint)
        optimizer, scheduler = self.optimizer_and_scheduler(policy)
        stream = streams_lib.resolve_task_stream(
            pairing, seed=int(seed), exposures=8 * 4, batch_rows=8,
            pairing_seed=int(self.config["block_a"]["pairing"]["seed_base"]) + int(seed))
        reference = self._reference_cache(parent_policy, population, seed=seed,
                                          parent_file_sha256=entry.get("file_sha256"))
        plan = monitor_lib.MonitorPlan(endpoints=(4,), checkpoints=())
        name = f"{namespace}_gate"
        document = trajectory_lib.run_trajectory(
            row={"trajectory": name, "block": "A", "task": "ipo", "seed": int(seed)},
            directory=directory, policy=policy, optimizer=optimizer, scheduler=scheduler,
            stream=stream, replay_order=None, plan=plan, endpoints=(4,), batch_rows=8,
            microbatch_rows=8,
            task_batch=lambda chosen, rejected: self._task_microbatch(
                policy, population, reference, task="ipo", coefficients={"tau": 0.1},
                chosen_rows=chosen, rejected_rows=rejected),
            preservation_batch=None,
            full_check=lambda **kwargs: gate.evaluate(policy, update=kwargs["update"],
                                                      reason=kwargs["reason"]),
            sentinel_check=lambda **kwargs: {"record_kind": "sentinel_check",
                                             "request_full_check": False, "update":
                                                 kwargs["update"]},
            on_endpoint=lambda **kwargs: {"update": kwargs["update"]},
            gradient_clip=1.0, identity={"trajectory": name}, uses_rejected=True,
            preservation_family="none", preservation_lambda=0.0, max_updates=4, resume=False)
        return document


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _state_digest(policy):
    return storage.state_digest(policy.model)


def _nested_digest(value):
    import hashlib
    digest = hashlib.sha256()
    trajectory_lib._feed_digest(digest, value)
    return digest.hexdigest()


def _native_tree_close(left, right):
    """Independent native trajectories: tight floating tolerance, exact structure/scalars."""
    import torch
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            return False
        a, b = left.detach().cpu(), right.detach().cpu()
        return bool(torch.allclose(a, b, atol=1e-7, rtol=1e-6)) if a.is_floating_point() else bool(torch.equal(a, b))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _native_tree_close(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _native_tree_close(a, b) for a, b in zip(left, right))
    return left == right


def _versus(scores, classes, other):
    """High versus one named class only. ``None`` when that class is absent."""
    mask = (classes == "high") | (classes == other)
    if not mask.any() or (classes[mask] == "high").all():
        return {"rows": int(mask.sum()), "average_precision": None,
                "reason": f"the panel carries no {other} rows to contrast against"}
    return metrics.rank_block(np.asarray(scores)[mask], classes[mask] == "high",
                              prevalence_note=False)


def _difference_in_differences(per_arm, *, dpo_arm="DPO_FKL"):
    """``[DPO_FKL - IPO_FKL]_purge - [DPO_FKL - IPO_FKL]_matched`` at the common endpoint."""
    needed = [f"{dpo_arm}@purge", "IPO_FKL@purge", f"{dpo_arm}@matched", "IPO_FKL@matched"]
    missing = [name for name in needed if name not in per_arm]
    if missing:
        return {"available": False, "missing": missing,
                "reason": "the main proximity contrast needs all four cells at the same endpoint"}
    seeds = sorted(set.intersection(*[set(per_arm[name]) for name in needed]))
    if len(seeds) < 2:
        return {"available": False, "paired_seeds": seeds,
                "reason": "fewer than two paired seeds carry all four cells"}
    differences = [(per_arm[f"{dpo_arm}@purge"][seed] - per_arm["IPO_FKL@purge"][seed])
                   - (per_arm[f"{dpo_arm}@matched"][seed] - per_arm["IPO_FKL@matched"][seed])
                   for seed in seeds]
    return {"available": True, "paired_seeds": seeds,
            **metrics.paired_t(np.asarray(differences)),
            "estimand": f"[{dpo_arm} - IPO_FKL]_purge - [{dpo_arm} - IPO_FKL]_matched, high-v-rest AP "
                        "on the identical E under both training regimes",
            "reading": ("a method's advantage collapsing after neighbourhood removal is a "
                        "useful negative result. With three seeds this interval has little "
                        "power and failing to exclude zero is not equivalence.")}


def _parent_ap_contrasts(per_arm):
    contrasts = []
    for name, values in sorted(per_arm.items()):
        arm, regime = name.rsplit("@", 1)
        if arm == "parent":
            continue
        parent = per_arm.get(f"parent@{regime}", {})
        seeds = sorted(set(values) & set(parent))
        differences = [values[seed] - parent[seed] for seed in seeds]
        block = (metrics.paired_t(np.asarray(differences)) if len(seeds) >= 2 else
                 {"reason": "fewer than two paired seeds at the common endpoint"})
        contrasts.append(dict(block, arm=arm, regime=regime, paired_seeds=seeds,
                               available=len(seeds) >= 2, estimand="high-v-rest AP minus own parent on E"))
    return contrasts


def _pair_reference(parent_policy, pairs, identity):
    from . import her2_guard as guard
    return guard.build_parent_reference(
        parent_policy, pairs,
        guard.parent_reference_identity(
            parent_checkpoint_sha256=identity.get("parent_checkpoint_sha256", "unrecorded"),
            parent_state_sha256=_state_digest(parent_policy),
            config_sha256=identity.get("config_sha256", "unrecorded"),
            scaffold_prefix=identity.get("scaffold_prefix", ""),
            chosen_index=pairs["chosen_index"], rejected_index=pairs["rejected_index"]))


def _freeze_still_stands(record, *, production_outcome_sha256):
    """May an existing finalist freeze be reused exactly as it is?

    A freeze taken while production was still incomplete is PROVISIONAL: it names
    only the checkpoints that existed then. Once production advances it has to be
    retaken -- still before any new bank is drawn, so the pre-registration
    ordering holds -- or the report publishes a finalist list chosen over a queue
    that has since moved. A freeze taken over a terminal queue is final and is
    never retaken.
    """
    if not record.get("provisional"):
        return True
    return record.get("production_outcome_sha256") == production_outcome_sha256


def _freeze_entry(entry, registry):
    """One registry row, in the shape the freeze record and the audits read."""
    parent_name = (f"B_{entry['regime']}_parent_seed{entry['seed']}" if entry.get("regime")
                   else f"parent_seed{entry['seed']}")
    parent = registry.get(parent_name) or {}
    return {"name": entry["name"], "checkpoint": entry["checkpoint"],
            "parent_checkpoint": entry.get("parent_checkpoint") or parent.get("checkpoint"),
            "parent_id": entry["parent_id"], "seed": int(entry["seed"]),
            "update": int(entry["update"]), "arm": entry["arm"], "regime": entry.get("regime"),
            "kind": entry["kind"], "reused": entry.get("reused"),
            "state_sha256": entry.get("state_sha256"),
            "parent_state_sha256": parent.get("state_sha256")}


def _grouped_worst_upper(models):
    grouped = {}
    for entry in models:
        for event, interval in entry["drop"]["wilson"].items():
            key = f"{entry.get('regime') or 'original_split'}::{entry['arm']}@u{entry['update']}::{event}"
            grouped.setdefault(key, []).append(interval)
    return {key: metrics.worst_upper(intervals) for key, intervals in sorted(grouped.items())}


def _choose_tail_family(families, *, criteria=None, tie=0.001, declared_seeds=()):
    """The declared tail-family rule, as data rather than prose.

    Feasibility first and across ALL **declared** seeds; then mean macro-AP; a
    .001 AP tie broken by Y@10k, then Y@1M, then IPO-tail. A family that carries
    results for one seed out of three is not all-seed feasible -- it was being
    accepted as one, because ``all()`` over a one-element dictionary is true --
    and its missing seeds stay visible in the summary. When neither family is
    feasible the one minimizing the maximum normalized tail violation is chosen
    and the record says so.
    """
    from . import her2_nf_calibration as calibration_rules

    table = dict(criteria or calibration_rules.POINT_RATE_CRITERIA)
    required = sorted(int(value) for value in declared_seeds)
    summary = {}
    for family, per_seed in sorted(dict(families).items()):
        if not per_seed:
            summary[family] = {"seeds": 0, "feasible": False, "missing_seeds": required,
                               "reason": "no development results for this family"}
            continue
        missing = [seed for seed in required if seed not in per_seed]
        feasible = (not missing) and all(
            calibration_rules.is_feasible(block, criteria=table)["feasible"]
            for block in per_seed.values())
        violation = max(max(float(block.get(f"{name}_rate") or 0.0) / float(limit)
                            for name, limit in table.items())
                        for block in per_seed.values())
        summary[family] = {
            "seeds": len(per_seed), "declared_seeds": required, "missing_seeds": missing,
            "feasible": bool(feasible),
            "all_seed_results": not missing,
            "reason": (None if not missing else
                       f"results for {len(per_seed)} of {len(required)} declared seeds; the "
                       "missing ones are stops or failures and stay visible rather than being "
                       "read as all-seed feasibility"),
            "mean_macro_average_precision": float(np.mean(
                [block["macro_average_precision"] for block in per_seed.values()])),
            "mean_yield_10k": float(np.mean([block["yield_10k"] for block in per_seed.values()])),
            "mean_yield_1m": float(np.mean([block["yield_1m"] for block in per_seed.values()])),
            "max_normalized_tail_violation": violation}
    usable = [name for name, block in summary.items() if block.get("feasible")]
    fallback = False
    if not usable:
        usable = [name for name, block in summary.items() if block.get("seeds")]
        fallback = True
    if not usable:
        return {"family": None, "family_arm": None, "summary": summary,
                "outcome": "no tail family has development results; neither is audited"}
    if fallback:
        best = min(summary[name]["max_normalized_tail_violation"] for name in usable)
        usable = [name for name in usable
                  if summary[name]["max_normalized_tail_violation"] == best]
        rule = "neither family is feasible; minimizing the maximum normalized tail violation"
    else:
        rule = "all-seed point-rate feasibility, then mean macro-AP"
    best_ap = max(summary[name]["mean_macro_average_precision"] for name in usable)
    tied = [name for name in usable
            if best_ap - summary[name]["mean_macro_average_precision"] <= float(tie)]
    broken_by = None
    if len(tied) > 1:
        broken_by = "Y@10k"
        best_yield = max(summary[name]["mean_yield_10k"] for name in tied)
        tied = [name for name in tied if summary[name]["mean_yield_10k"] == best_yield]
    if len(tied) > 1:
        broken_by = "Y@1M"
        best_million = max(summary[name]["mean_yield_1m"] for name in tied)
        tied = [name for name in tied if summary[name]["mean_yield_1m"] == best_million]
    if len(tied) > 1:
        broken_by = "IPO-tail"
        tied = ["ipo_tail"] if "ipo_tail" in tied else tied[:1]
    chosen = tied[0]
    return {"family": chosen, "family_arm": {"ipo_tail": "IPO_TAIL",
                                             "dpo_tail": "DPO_TAIL"}[chosen],
            "feasible": not fallback, "tie_broken_by": broken_by, "rule": rule,
            "summary": summary,
            "retained": "the other family's 10k development result is retained and reported"}


def _read_optional_json(path):
    return paths.read_json(path) if Path(path).is_file() else None


def _as_bytes(value):
    """Raw bytes of an RNG state blob, whichever container torch handed back."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").to(torch.uint8).numpy().tobytes()
    return np.asarray(value, dtype=np.uint8).tobytes()


def _class_mass(sum_log_probability, classes):
    """Exact generated mass on each class of the declared assayed panel.

    ``sum exp(ell)`` over the rows of each class -- a probability, under the one
    modelled event. A draw-hit rate over a generated bank is a DIFFERENT
    statistic and is reported under its own name in the diversity block; the two
    were being conflated, and only one of them is what the specification calls
    class mass.
    """
    values = np.asarray(sum_log_probability, dtype=np.float64)
    labels = np.asarray(classes)
    out = {}
    for name in sorted(set(labels.tolist())):
        mask = labels == name
        out[str(name)] = {"rows": int(mask.sum()),
                          "mass": float(np.exp(values[mask]).sum()),
                          "mean_sum_log_probability": float(values[mask].mean())}
    total = float(np.exp(values).sum())
    out["_panel"] = {"rows": int(values.size), "mass": total,
                     "denominator": "the declared assayed evaluation panel, all classes",
                     "purity": {name: (block["mass"] / total if total > 0 else None)
                                for name, block in out.items() if not name.startswith("_")},
                     "definition": ("sum of exp(ell) over each class. This is generated "
                                    "probability mass, not a draw-hit rate and not a yield "
                                    "input.")}
    return out


def _slug(text):
    """A filesystem-safe name that still reads as the model it names."""
    return "".join(character if character.isalnum() or character in "-_." else "_"
                   for character in str(text))


def _diversity_block(index):
    """Diversity at a fixed draw count, beside the dependence statistics."""
    values = np.asarray(index)
    hashes = contract.index_hashes(values)
    unique, counts = np.unique(hashes, return_counts=True)
    entropy = []
    for position in range(values.shape[1]):
        counts_at = np.bincount(values[:, position].astype(np.int64),
                                minlength=len(data_lib.CANONICAL)).astype(np.float64)
        probabilities = counts_at / counts_at.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)
        entropy.append(float(-terms.sum()))
    return {"rows": int(values.shape[0]), "unique_cores": int(unique.size),
            "unique_fraction": float(unique.size) / float(values.shape[0]),
            "max_single_core_frequency": float(counts.max()) / float(values.shape[0]),
            "site_entropy_nats": entropy, "sum_site_entropy_nats": float(sum(entropy)),
            "basis": "a fixed draw count; duplicates are retained and never filtered"}


def _bank_bootstrap(index, *, seed, draws=200):
    """Whole-bank resampling of the 45-pair MI sum.

    Monte Carlo sensitivity of a statistic computed on ONE bank. It is not an
    extra trained seed, it does not remove the estimator's downward bias, and the
    record says both. The independent repeat bank answers the other question.
    """
    values = np.asarray(index)
    generator = np.random.default_rng(int(seed))
    sums = []
    for _ in range(int(draws)):
        rows = generator.integers(0, values.shape[0], values.shape[0])
        sums.append(coupling.pairwise_mutual_information(values[rows])["sum"])
    array = np.asarray(sums, dtype=np.float64)
    return {"draws": int(draws), "seed": int(seed), "rows": int(values.shape[0]),
            "mean": float(array.mean()),
            "quantiles": {"0.025": float(np.quantile(array, 0.025)),
                          "0.5": float(np.quantile(array, 0.5)),
                          "0.975": float(np.quantile(array, 0.975))},
            "status": ("whole-bank resampling of ONE bank. Monte Carlo sensitivity, not training "
                       "variation and not a bias correction; the independent repeat bank answers "
                       "the repeat question and the three trained seeds answer the seed one.")}


def _save_sklearn(path, fit):
    """Persist a fitted sklearn classifier as its coefficients plus its record."""
    model = fit["model"]
    paths.write_arrays(Path(path), {"coef": np.asarray(model.coef_, dtype=np.float64),
                                    "intercept": np.asarray(model.intercept_, dtype=np.float64),
                                    "classes": np.asarray(model.classes_, dtype=np.int64)})
    paths.write_json(Path(path).with_suffix(".json"),
                     _jsonable({k: v for k, v in fit.items() if k != "model"}))
    return str(path)


def _load_sklearn(path):
    """Rebuild a fitted classifier from persisted coefficients. ``None`` if absent."""
    from sklearn.linear_model import LogisticRegression

    target, sidecar = Path(path), Path(path).with_suffix(".json")
    if not (target.is_file() and sidecar.is_file()):
        return None
    arrays = paths.read_arrays(target)
    record = paths.read_json(sidecar)
    model = LogisticRegression(penalty="l2", C=float(record["selected_C"]), solver="saga")
    model.coef_ = np.asarray(arrays["coef"], dtype=np.float64)
    model.intercept_ = np.asarray(arrays["intercept"], dtype=np.float64)
    model.classes_ = np.asarray(arrays["classes"], dtype=np.int64)
    model.n_features_in_ = int(model.coef_.shape[1])
    return dict(record, model=model)


def _read_journal_head(path, count):
    import json
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_kind") == "update":
                records.append(record)
            if len(records) >= int(count):
                break
    return records


def _jsonable(node):
    if isinstance(node, dict):
        return {str(key): _jsonable(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_jsonable(value) for value in node]
    if isinstance(node, np.generic):
        return node.item()
    if isinstance(node, np.ndarray):
        return node.tolist()
    if isinstance(node, Path):
        return str(node)
    return node
