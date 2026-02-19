"""K-means color clustering for robust background separation.

Replaces single-Lab-distance background detection with k-means clustering
to handle multi-colored backgrounds, tinted pages, watercolor art, etc.

Pipeline:
1. Downsample image for speed
2. K-means clustering in Lab color space (k=3-5)
3. Identify background cluster (largest cluster near page borders)
4. Build mask from background cluster(s)
5. Optional: identify gutter/border clusters

This handles pages where the background is not uniform (gradients,
textures, multi-colored borders).
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from dataclasses import dataclass

from .utils import pdebug, HAS_CV2, HAS_NUMPY

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class ColorCluster:
    """A color cluster from k-means."""
    center_lab: Tuple[float, float, float]  # Cluster center in Lab
    center_bgr: Tuple[int, int, int]        # Cluster center in BGR
    pixel_count: int                        # Number of pixels in cluster
    fraction: float                         # Fraction of total pixels
    is_background: bool                     # Whether this is a background cluster
    border_presence: float                  # How much this cluster appears at borders


def cluster_colors(
    img_bgr: NDArray,
    k: int = 4,
    max_pixels: int = 50000,
) -> List[ColorCluster]:
    """Cluster image colors using k-means in Lab color space.

    Args:
        img_bgr: Input image in BGR format
        k: Number of clusters
        max_pixels: Maximum pixels to use (downsamples if larger)

    Returns:
        List of ColorCluster objects sorted by pixel count (descending)
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []

    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # Downsample for speed
    if total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
        new_w = max(10, int(w * scale))
        new_h = max(10, int(h * scale))
        img_small = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr
        new_h, new_w = h, w

    # Convert to Lab
    img_lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2Lab).astype(np.float32)
    pixels_lab = img_lab.reshape(-1, 3)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels_lab,
        k,
        None,
        criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.flatten()

    # Compute border presence for each cluster
    border_size = max(2, int(min(new_h, new_w) * 0.04))
    border_mask = np.zeros((new_h, new_w), dtype=bool)
    border_mask[:border_size, :] = True
    border_mask[-border_size:, :] = True
    border_mask[:, :border_size] = True
    border_mask[:, -border_size:] = True
    border_flat = border_mask.flatten()
    total_border = border_flat.sum()

    clusters = []
    for i in range(k):
        cluster_mask = (labels == i)
        pixel_count = int(cluster_mask.sum())
        fraction = pixel_count / len(labels)

        # Border presence
        border_count = int((cluster_mask & border_flat).sum())
        border_presence = border_count / total_border if total_border > 0 else 0.0

        # Center in Lab
        center_lab = tuple(float(x) for x in centers[i])

        # Center in BGR
        center_lab_img = np.array([[centers[i]]], dtype=np.float32).astype(np.uint8)
        center_bgr_img = cv2.cvtColor(center_lab_img.reshape(1, 1, 3), cv2.COLOR_Lab2BGR)
        center_bgr = tuple(int(x) for x in center_bgr_img[0, 0])

        clusters.append(ColorCluster(
            center_lab=center_lab,
            center_bgr=center_bgr,
            pixel_count=pixel_count,
            fraction=fraction,
            is_background=False,
            border_presence=border_presence,
        ))

    # Sort by pixel count (largest first)
    clusters.sort(key=lambda c: c.pixel_count, reverse=True)

    return clusters


def identify_background_clusters(
    clusters: List[ColorCluster],
    min_border_presence: float = 0.15,
    min_fraction: float = 0.05,
) -> List[ColorCluster]:
    """Identify which clusters represent the page background.

    Background clusters are those that:
    1. Appear significantly at page borders
    2. Have sufficient pixel count
    3. Are relatively bright (L > 140 in OpenCV Lab scale)

    Args:
        clusters: List of ColorCluster objects
        min_border_presence: Minimum fraction of border pixels
        min_fraction: Minimum fraction of total pixels

    Returns:
        Clusters identified as background (with is_background=True)
    """
    bg_clusters = []

    for cluster in clusters:
        L = cluster.center_lab[0]

        # Background must be relatively bright and present at borders
        is_bright = L > 140  # OpenCV Lab L range is 0-255
        is_at_border = cluster.border_presence >= min_border_presence
        is_large_enough = cluster.fraction >= min_fraction

        if is_bright and is_at_border and is_large_enough:
            cluster.is_background = True
            bg_clusters.append(cluster)
        elif cluster.border_presence >= 0.40 and is_large_enough:
            # Very dominant at borders, even if not super bright
            cluster.is_background = True
            bg_clusters.append(cluster)

    # If no background found, use the brightest large cluster
    if not bg_clusters:
        bright_large = [c for c in clusters if c.center_lab[0] > 120 and c.fraction > 0.08]
        if bright_large:
            bright_large[0].is_background = True
            bg_clusters.append(bright_large[0])

    pdebug(f"[KMeans] Background clusters: {len(bg_clusters)}/{len(clusters)}")
    for bc in bg_clusters:
        pdebug(f"  L={bc.center_lab[0]:.0f} border={bc.border_presence:.2f} "
               f"frac={bc.fraction:.2f}")

    return bg_clusters


def make_kmeans_background_mask(
    img_bgr: NDArray,
    k: int = 4,
    delta_expand: float = 15.0,
) -> NDArray:
    """Create background mask using k-means clustering.

    More robust than single-Lab-distance approach:
    - Handles multi-colored backgrounds
    - Handles tinted/textured pages
    - Adapts to any background color

    Args:
        img_bgr: Input image in BGR format
        k: Number of clusters
        delta_expand: Lab distance to expand background mask around cluster centers

    Returns:
        Binary mask (255 = background, 0 = foreground)
    """
    if not HAS_CV2 or not HAS_NUMPY:
        h, w = img_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    h, w = img_bgr.shape[:2]

    # Cluster colors
    clusters = cluster_colors(img_bgr, k=k)
    if not clusters:
        return np.zeros((h, w), dtype=np.uint8)

    # Identify background
    bg_clusters = identify_background_clusters(clusters)
    if not bg_clusters:
        return np.zeros((h, w), dtype=np.uint8)

    # Build mask: pixels close to any background cluster center
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)

    for bc in bg_clusters:
        center = np.array(bc.center_lab, dtype=np.float32)
        dist = np.linalg.norm(img_lab - center, axis=2)
        # Expand delta based on cluster size (larger clusters get more tolerance)
        effective_delta = delta_expand * (1.0 + bc.fraction * 0.5)
        bg_pixels = (dist < effective_delta)
        mask[bg_pixels] = 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    bg_pct = (np.sum(mask == 255) / mask.size) * 100
    pdebug(f"[KMeans] Background mask: {bg_pct:.1f}% of image")

    return mask


def get_dominant_bg_lab(
    img_bgr: NDArray,
    k: int = 4,
) -> NDArray:
    """Get dominant background color in Lab using k-means.

    More robust alternative to border-sampling for background estimation.

    Args:
        img_bgr: Input image in BGR format
        k: Number of clusters

    Returns:
        Array of shape (3,) containing (L, a, b) values
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return np.array([255.0, 128.0, 128.0])

    clusters = cluster_colors(img_bgr, k=k)
    bg_clusters = identify_background_clusters(clusters)

    if bg_clusters:
        # Use the most border-dominant background cluster
        best = max(bg_clusters, key=lambda c: c.border_presence)
        return np.array(best.center_lab, dtype=np.float32)
    elif clusters:
        # Fallback: brightest cluster
        brightest = max(clusters, key=lambda c: c.center_lab[0])
        return np.array(brightest.center_lab, dtype=np.float32)
    else:
        return np.array([255.0, 128.0, 128.0])
