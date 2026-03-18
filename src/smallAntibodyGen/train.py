import math
from pathlib import Path

import torch
from torch.optim import AdamW


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy only on MLM target positions.

    Args:
        logits:
            Tensor of shape [B, L, V].
        labels:
            Tensor of shape [B, L] with target token IDs at masked positions
            and -100 elsewhere.

    Returns:
        Scalar float accuracy over masked positions only.
    """
    preds = logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum().item() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).float().mean().item()
    return correct


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Run validation over one loader.

    Args:
        model:
            The MLM model.
        loader:
            Validation dataloader.
        device:
            torch.device.

    Returns:
        Tuple (mean_loss, mean_masked_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = model.compute_loss(logits, labels)
        acc = masked_accuracy(logits, labels)

        total_loss += float(loss.item())
        total_acc += acc
        total_batches += 1

    if total_batches == 0:
        return math.nan, math.nan

    return total_loss / total_batches, total_acc / total_batches


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip_norm: float = 1.0,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
):
    """
    Train the MLM for one epoch.

    Args:
        model:
            The MLM model.
        loader:
            Training dataloader.
        optimizer:
            Optimizer, e.g. AdamW.
        device:
            torch.device.
        grad_clip_norm:
            Gradient clipping threshold.
        use_amp:
            Whether to use mixed precision.
        scaler:
            GradScaler if AMP is enabled.

    Returns:
        Tuple (mean_train_loss, mean_train_masked_accuracy).
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, enabled=True):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = model.compute_loss(logits, labels)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = model.compute_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        acc = masked_accuracy(logits.detach(), labels)

        total_loss += float(loss.item())
        total_acc += acc
        total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


def fit(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    grad_clip_norm: float = 1.0,
    output_dir: str = "checkpoints",
    use_amp: bool = False,
):
    """
    Full training loop with checkpointing.

    Args:
        model:
            The MLM model.
        train_loader:
            Training dataloader.
        val_loader:
            Validation dataloader.
        device:
            torch.device.
        num_epochs:
            Number of training epochs.
        lr:
            Learning rate for AdamW.
        weight_decay:
            Weight decay for AdamW.
        grad_clip_norm:
            Gradient clipping threshold.
        output_dir:
            Directory where checkpoints are saved.
        use_amp:
            Whether to use mixed precision.

    Returns:
        None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
            use_amp=use_amp,
            scaler=scaler,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        print(
            f"[epoch {epoch+1}/{num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        latest_ckpt = output_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            latest_ckpt,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = output_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt,
            )