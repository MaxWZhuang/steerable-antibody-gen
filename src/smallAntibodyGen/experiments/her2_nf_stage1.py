"""Fresh-parent training with atomic, population-bound resumable state."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from . import her2_nf_storage as storage
from . import her2_nf_trajectory as trajectory
from . import her2_policy
from . import her2_support_paths as paths
from .her2_runtime import require


def fit(policy, *, index, seed, directory, plan_config, optimization, selection_index,
        selection_labels, source_sha256=None, after_update=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    batch = int(plan_config["batch_size"])
    micro = int(plan_config.get("microbatch_rows") or batch)
    epochs = int(plan_config["epochs"])
    rows = len(index)
    require(rows > 0 and batch % micro == 0, "Invalid stage-1 rows or accumulation")
    positives = np.asarray(selection_labels, dtype=bool)
    require(len(positives) == len(selection_index) and positives.any(), "No high C selection rows")
    steps_per_epoch = int(np.ceil(rows / batch))
    total_steps = steps_per_epoch * epochs
    warmup = int(round(float(optimization.get("warmup_fraction", 0.05)) * total_steps))
    final_fraction = float(optimization["final_learning_rate_fraction"])
    identity = {"kind": "fresh_parent", "seed": int(seed),
                "population_sha256": paths.array_digest(np.asarray(index)),
                "selection_sha256": paths.array_digest(np.asarray(selection_index)[positives]),
                "raw_parent_sha256": storage.state_digest(policy.model),
                "source_snapshot_sha256": source_sha256, "plan": dict(plan_config),
                "optimization": dict(optimization),
                "prefix": paths.array_digest(policy.prefix_ids.detach().cpu().numpy())}
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=float(optimization["learning_rate"]),
                                  betas=tuple(optimization["betas"]),
                                  weight_decay=float(optimization["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: her2_policy.learning_rate_scale(
            step + 1, total_steps=total_steps, warmup_steps=warmup, final_fraction=final_fraction))
    optimizer.zero_grad(set_to_none=True)
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    state_path = directory / "stage1_state.pt"
    progress = {"epoch": 1, "next_start": 0, "updates": 0, "epoch_loss": 0.0,
                "seen": 0, "history": [], "checkpoints": {}}

    def verify(payload):
        require(payload.get("identity") == identity, "Stage-1 population, parent or protocol changed")
        require(storage.state_dict_digest(payload["state"]) == payload["state_sha256"],
                "Stage-1 model state is corrupt")
        require(trajectory._payload_digest(payload) == payload["payload_sha256"],
                "Stage-1 optimizer, RNG or cursor state is corrupt")

    def save():
        tensors = policy.model.state_dict()
        payload = {"schema_version": "her2-next-flight-stage1/1", "identity": identity,
                   "state": tensors, "state_sha256": storage.state_dict_digest(tensors),
                   "state_schema": trajectory.state_schema(tensors),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": None, "scaler_reason": trajectory.NO_SCALER_REASON,
                   "rng": trajectory.capture_rng_state(), "progress": dict(progress),
                   "stream_digests": {"population": identity["population_sha256"]}}
        payload["payload_sha256"] = trajectory._payload_digest(payload)
        storage.atomic_torch_save(payload, state_path, verify=verify)

    if state_path.is_file():
        payload = storage.load_cpu(state_path)
        verify(payload)
        policy.model.load_state_dict(payload["state"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        trajectory.restore_rng_state(payload["rng"])
        require(storage.state_digest(policy.model) == payload["state_sha256"], "Stage-1 restore changed weights")
        restored = dict(payload, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        rng=trajectory.capture_rng_state())
        require(trajectory._payload_digest(restored) == payload["payload_sha256"],
                "Stage-1 restore changed optimizer or RNG")
        progress = dict(payload["progress"])
        del restored, payload
    checkpoints = set(int(value) for value in plan_config["checkpoints"])
    save_interval = int(plan_config.get("resume_interval_updates", 100))
    while progress["epoch"] <= epochs:
        epoch = int(progress["epoch"])
        policy.model.train()
        order = np.random.default_rng([int(seed), epoch]).permutation(rows)
        for start in range(int(progress["next_start"]), rows, batch):
            chunk = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(chunk), micro):
                piece = chunk[offset:offset + micro]
                loss = policy.loss(index[piece]) * (len(piece) / float(len(chunk)))
                require(bool(torch.isfinite(loss)), "Nonfinite stage-1 training loss")
                loss.backward()
                progress["epoch_loss"] += float(loss.detach()) * len(chunk)
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), float(optimization["gradient_clip"]),
                                           error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            progress["updates"] += 1
            progress["seen"] += len(chunk)
            progress["next_start"] = start + len(chunk)
            if progress["updates"] % save_interval == 0 or progress["next_start"] == rows:
                save()
            if after_update is not None:
                after_update(progress["updates"])
        record = {"epoch": epoch, "rows": progress["seen"], "steps_per_epoch": steps_per_epoch,
                  "train_nll_per_residue": progress["epoch_loss"] / max(progress["seen"], 1),
                  "learning_rate": float(scheduler.get_last_lr()[0])}
        if epoch in checkpoints:
            target = directory / f"epoch_{epoch}.pt"
            digest = storage.save_checkpoint(target, policy, {"epoch": epoch, "seed": int(seed),
                                                               "identity": identity})
            rng = trajectory.capture_rng_state()
            scored = policy.score(np.asarray(selection_index)[positives])["mean_log_probability"]
            trajectory.restore_rng_state(rng)
            nll = float(-np.asarray(scored).mean())
            require(np.isfinite(nll), "Nonfinite high-C selection NLL")
            progress["checkpoints"][epoch] = {"path": str(target), "state_sha256": digest,
                                                "selection_nll_per_residue": nll}
            record.update(checkpoint=str(target), selection_nll_per_residue=nll)
        progress["history"].append(record)
        progress.update(epoch=epoch + 1, next_start=0, epoch_loss=0.0, seen=0)
        save()
        paths.write_json(directory / "stage1_history.json", progress["history"])
    saved = progress["checkpoints"]
    require(saved, "Stage-1 produced no selectable checkpoint")
    for entry in saved.values():
        payload = storage.load_cpu(entry["path"], weights_only=True)
        require(payload.get("identity") == identity
                and storage.state_dict_digest(payload["state"]) == entry["state_sha256"],
                "A selectable stage-1 checkpoint changed")
        del payload
    best = min(saved, key=lambda epoch: (saved[epoch]["selection_nll_per_residue"], int(epoch)))
    return {"fit": {"history": progress["history"], "total_steps": total_steps,
                    "steps_per_epoch": steps_per_epoch, "warmup_steps": warmup,
                    "effective_batch": batch, "microbatch_rows": micro,
                    "learning_rate": float(optimization["learning_rate"]),
                    "resume_interval_updates": save_interval,
                    "schedule": f"warmup {warmup} steps then cosine to {final_fraction:g}"},
            "identity": identity, "checkpoints": saved, "selected_epoch": int(best),
            "selected": saved[best], "selection_metric": "positive-class NLL per residue on C",
            "selection_population": "C only", "protocol": {"plan": plan_config, "optimization": optimization}}
