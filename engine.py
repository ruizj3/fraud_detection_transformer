import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are raw logits, targets are 0 or 1
        p = torch.sigmoid(inputs)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


def resolve_device(device_setting: str) -> torch.device:
    if device_setting != "auto":
        return torch.device(device_setting)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x_cat, x_cont, y in loader:
        x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x_cat, x_cont).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_targets = [], []

    for x_cat, x_cont, y in loader:
        x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

        logits = model(x_cat, x_cont).squeeze(1)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)

        all_probs.append(torch.sigmoid(logits).cpu())
        all_targets.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "roc_auc": roc_auc_score(targets, probs) if len(set(targets)) > 1 else float("nan"),
        "pr_auc": average_precision_score(targets, probs) if len(set(targets)) > 1 else float("nan"),
        "f1": f1_score(targets, preds, zero_division=0),
    }
    return metrics


def train(model, train_loader, val_loader, config):
    training_cfg = config["training"]
    device = resolve_device(training_cfg.get("device", "auto"))
    model.to(device)

    criterion = FocalLoss(
        alpha=training_cfg.get("focal_loss_alpha", 0.25),
        gamma=training_cfg.get("focal_loss_gamma", 2.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-5),
    )

    checkpoint_dir = training_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_model.pt")

    best_pr_auc = -float("inf")
    patience = training_cfg.get("early_stopping_patience", 4)
    epochs_without_improvement = 0

    for epoch in range(1, training_cfg.get("epochs", 20) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
            f"| val_roc_auc={val_metrics['roc_auc']:.4f} | val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"| val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"⏹️  Early stopping at epoch {epoch} (no val PR-AUC improvement in {patience} epochs).")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, device
