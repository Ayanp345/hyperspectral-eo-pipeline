from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def band_index_for_wavelength(wavelengths_nm: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths_nm - target_nm)))


def false_color_composite(cube: np.ndarray, wavelengths_nm: np.ndarray,
                           r_nm: float = 660, g_nm: float = 560, b_nm: float = 480) -> np.ndarray:
    """Build a stretched RGB composite from three chosen spectral bands."""
    ri = band_index_for_wavelength(wavelengths_nm, r_nm)
    gi = band_index_for_wavelength(wavelengths_nm, g_nm)
    bi = band_index_for_wavelength(wavelengths_nm, b_nm)
    rgb = np.stack([cube[..., ri], cube[..., gi], cube[..., bi]], axis=-1).astype(np.float64)

    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-9), 0, 1)
    return rgb


def label_map_to_rgb(label_map: np.ndarray, class_colors: Sequence[Sequence[int]]) -> np.ndarray:
    palette = np.array(class_colors, dtype=np.uint8)
    safe_labels = np.clip(label_map, 0, len(palette) - 1)
    rgb = palette[safe_labels]
    rgb = np.where((label_map[..., None] < 0), 0, rgb)  # unclassified (-1) -> black
    return rgb


def save_false_color(cube, wavelengths_nm, out_path, title="False-Color Composite"):
    rgb = false_color_composite(cube, wavelengths_nm)
    fig, ax = plt.subplots(figsize=(6, 6.4))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11, wrap=True)
    ax.axis("off")
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_classification_map(label_map, class_names, class_colors, out_path,
                             title="Classification Map"):
    rgb = label_map_to_rgb(label_map, class_colors)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255.0) for c in class_colors
    ]
    ax.legend(handles, class_names, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_spectral_signatures(wavelengths_nm, spectra: Dict[str, np.ndarray], out_path,
                              title="Reference Spectral Signatures", vlines=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, spectrum in spectra.items():
        ax.plot(wavelengths_nm, spectrum, label=name, linewidth=1.8)
    if vlines:
        for x, label in vlines:
            ax.axvline(x, color="gray", linestyle="--", linewidth=0.8)
            ax.text(x, ax.get_ylim()[1] * 0.97, label, rotation=90, fontsize=7,
                    ha="right", va="top", color="gray")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_variance_plot(explained_var_ratio, out_path, k_used: Optional[int] = None,
                        title="PCA Explained Variance"):
    cumulative = np.cumsum(explained_var_ratio)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker=".", markersize=3)
    if k_used:
        ax.axvline(k_used, color="red", linestyle="--", label=f"selected k={k_used}")
        ax.legend()
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix (test set)"):
    cm = np.array(cm)
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="viridis",
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_heatmap_overlay(base_rgb, score_map, out_path, title="Anomaly / Detection Score",
                          cmap="inferno", alpha=0.55, mask: Optional[np.ndarray] = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(base_rgb)
    axes[0].set_title("False-Color Reference")
    axes[0].axis("off")

    axes[1].imshow(base_rgb)
    im = axes[1].imshow(score_map, cmap=cmap, alpha=alpha)
    if mask is not None:
        axes[1].contour(mask, colors="cyan", linewidths=1.2)
    axes[1].set_title(title)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_abundance_maps(abundance_maps, class_names, out_path, title="Abundance Maps (Unmixing)"):
    k = abundance_maps.shape[-1]
    ncols = 4
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.6))
    axes = np.array(axes).reshape(-1)
    for i in range(k):
        im = axes[i].imshow(abundance_maps[..., i], cmap="magma", vmin=0, vmax=1)
        axes[i].set_title(class_names[i], fontsize=9)
        axes[i].axis("off")
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_training_curves(history: Dict[str, List[float]], out_path, title="Training Curves"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_generic_map(data2d, out_path, title, cmap="viridis", colorbar_label=None):
    """
    Renders a 2D scalar map. NaN values are shown as light gray (used for
    e.g. vegetation-only indices masked out over non-vegetated pixels, where
    the underlying formula is not physically meaningful).
    """
    masked = np.ma.masked_invalid(data2d)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#dddddd")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(masked, cmap=cmap_obj)
    ax.set_title(title)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cb.set_label(colorbar_label)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
