"""
src/dl/train.py
------------------
Shared training/evaluation loop for both deep models (`HybridSN` and
`SpectralSpatialTransformer`) -- they expose an identical
(batch, 1, C, patch, patch) -> (batch, num_classes) interface, so one loop
drives either architecture via the `model_name` switch.

Protocol (matches standard HSI classification benchmark practice):
    1. PCA-reduce the corrected reflectance cube to a fixed number of
       components (deep spectral-spatial nets are trained on PCA-reduced
       cubes, not the full band count -- see `src/dl/dataset.py`).
    2. Stratified per-class pixel split into train/val/test.
    3. Train with Adam + weight decay, monitoring validation loss for early
       stopping (patience from config) to avoid overfitting the (small,
       synthetic) label set.
    4. Evaluate on the held-out test split with the same OA/AA/Kappa metrics
       used everywhere else in this project, and additionally predict the
       *entire* scene (every pixel, not just labeled/test ones) to render a
       full classification-map figure.
"""
from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np

from src.evaluation.metrics import classification_metrics
from src.preprocessing.dimensionality_reduction import pca_reduce_fixed
from src.visualization import viz


def _log(msg: str) -> None:
    print(f"[pixxel-hsi-cv:dl] {msg}", flush=True)


def build_model(model_name: str, num_classes: int, cfg):
    from src.dl.hybridsn import HybridSN
    from src.dl.spectral_transformer import SpectralSpatialTransformer

    dl_cfg = cfg.deep_learning
    if model_name == "hybridsn":
        return HybridSN(num_classes=num_classes, pca_components=dl_cfg.pca_components,
                         patch_size=dl_cfg.patch_size)
    if model_name == "spectral_transformer":
        t_cfg = dl_cfg.transformer
        return SpectralSpatialTransformer(
            num_classes=num_classes, pca_components=dl_cfg.pca_components,
            patch_size=dl_cfg.patch_size, embed_dim=t_cfg.embed_dim, depth=t_cfg.depth,
            num_heads=t_cfg.num_heads, mlp_ratio=t_cfg.mlp_ratio, dropout=t_cfg.dropout,
            spectral_group_size=t_cfg.spectral_group_size,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def run_training(cfg, reflectance_est: np.ndarray, label_map: np.ndarray, class_names,
                  outdir: str, model_name: str = "hybridsn") -> Dict:
    import torch
    from torch.utils.data import DataLoader

    from src.dl.dataset import HSIPatchDataset, stratified_pixel_split

    dl_cfg = cfg.deep_learning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"device: {device}")

    pca_cube = pca_reduce_fixed(reflectance_est, dl_cfg.pca_components)
    train_idx, val_idx, test_idx = stratified_pixel_split(
        label_map, dl_cfg.train_ratio, dl_cfg.val_ratio, seed=42
    )
    _log(f"train/val/test pixels: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    train_ds = HSIPatchDataset(pca_cube, label_map, train_idx, dl_cfg.patch_size)
    val_ds = HSIPatchDataset(pca_cube, label_map, val_idx, dl_cfg.patch_size)
    test_ds = HSIPatchDataset(pca_cube, label_map, test_idx, dl_cfg.patch_size)

    train_loader = DataLoader(train_ds, batch_size=dl_cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=dl_cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=dl_cfg.batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name, len(class_names), cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=dl_cfg.lr, weight_decay=dl_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(dl_cfg.epochs):
        t0 = time.time()
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            running_n += labels.size(0)

        train_loss = running_loss / max(running_n, 1)
        train_acc = running_correct / max(running_n, 1)

        model.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                logits = model(patches)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_n += labels.size(0)
        val_loss /= max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        _log(f"[{model_name}] epoch {epoch+1}/{dl_cfg.epochs}  "
             f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
             f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= dl_cfg.early_stopping_patience:
                _log(f"[{model_name}] early stopping at epoch {epoch+1} "
                     f"(no val improvement for {dl_cfg.early_stopping_patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Test-set evaluation -------------------------------------------------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for patches, labels in test_loader:
            patches = patches.to(device)
            logits = model(patches)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = classification_metrics(y_true, y_pred, class_names)
    _log(f"[{model_name}] TEST  OA={metrics['overall_accuracy']:.4f}  "
         f"AA={metrics['average_accuracy']:.4f}  Kappa={metrics['cohen_kappa']:.4f}")

    viz.save_training_curves(history, f"{outdir}/figures/{model_name}_training_curves.png",
                              title=f"{model_name} training curves")
    viz.save_confusion_matrix(metrics["confusion_matrix"], class_names,
                               f"{outdir}/figures/{model_name}_confusion_matrix.png",
                               title=f"{model_name} confusion matrix (test set)")

    # --- Full-scene prediction map -------------------------------------------
    all_coords = np.argwhere(np.ones_like(label_map, dtype=bool))
    full_ds = HSIPatchDataset(pca_cube, label_map, all_coords, dl_cfg.patch_size)
    full_loader = DataLoader(full_ds, batch_size=max(dl_cfg.batch_size, 256), shuffle=False)
    full_preds = []
    with torch.no_grad():
        for patches, _ in full_loader:
            logits = model(patches.to(device))
            full_preds.append(logits.argmax(1).cpu().numpy())
    full_pred_map = np.concatenate(full_preds).reshape(label_map.shape)

    palette = [
        (30, 80, 200), (30, 160, 60), (190, 170, 40), (150, 110, 70),
        (190, 90, 40), (170, 140, 190), (120, 120, 120),
    ]
    viz.save_classification_map(full_pred_map, class_names, palette,
                                 f"{outdir}/figures/{model_name}_classification_map.png",
                                 title=f"{model_name} predicted class map (full scene)")

    return metrics
