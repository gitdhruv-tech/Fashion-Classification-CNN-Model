"""
dataset_utils.py — Dataset scanning, class mapping, splitting, and array building.

Key invariant: preprocess() from preprocess.py is the ONLY pixel transformation used.
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────

CLOTHING_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
OTHER_CLASS = "Other / Not Clothing"
CANONICAL_11_CLASSES = CLOTHING_CLASSES + [OTHER_CLASS]
CANONICAL_CLASSES = CLOTHING_CLASSES  # backward compatibility alias

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ─── Class name resolution ────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    """Lowercase and strip all separators (-, _, /, space)."""
    return name.lower().replace("-", "").replace("_", "").replace("/", "").replace(" ", "")


# Pre-compute normalised canonical names
_NORM_TO_CANONICAL = {_normalise(c): c for c in CANONICAL_11_CLASSES}

# Explicit aliases for the non-clothing / other category
_OTHER_ALIASES = [
    "other", "notclothing", "nonclothing", "othernotclothing", "notacloth",
    "reject", "unrelated", "random", "negative", "notaclothing", "none",
    "background", "others", "nonclothes"
]
for alias in _OTHER_ALIASES:
    _NORM_TO_CANONICAL[_normalise(alias)] = OTHER_CLASS


def resolve_class_name(folder_name: str):
    """
    Case-insensitive, separator-insensitive match against clothing categories
    and the optional 'Other / Not Clothing' category.
    Returns the canonical class name (str) or None if no match.
    """
    # 1. Direct (case-insensitive) match
    for c in CANONICAL_11_CLASSES:
        if folder_name.strip().lower() == c.lower():
            return c

    # 2. Normalised match (strips all separators)
    norm = _normalise(folder_name)
    return _NORM_TO_CANONICAL.get(norm, None)


def get_class_names_for_dataset(matched_classes: dict) -> list:
    """Returns CANONICAL_11_CLASSES if OTHER_CLASS is present, else CLOTHING_CLASSES."""
    if OTHER_CLASS in matched_classes:
        return list(CANONICAL_11_CLASSES)
    return list(CLOTHING_CLASSES)


# ─── Dataset scanning ─────────────────────────────────────────────────────────

def scan_dataset(folder_path: str) -> dict:
    """
    Scans a folder of class-named sub-directories.

    Returns:
        {
            "valid": bool,
            "matched_classes": dict[str, list[Path]],  # canonical_name → valid image paths
            "unmatched_folders": list[str],
            "corrupt_files": list[str],
            "unsupported_files": int,
            "total_valid": int,
            "warnings": list[str],
            "errors": list[str],
        }
    Does NOT modify any files.
    """
    result = {
        "valid": False,
        "matched_classes": {},
        "unmatched_folders": [],
        "corrupt_files": [],
        "unsupported_files": 0,
        "total_valid": 0,
        "warnings": [],
        "errors": [],
    }

    root = Path(folder_path)

    if not root.exists():
        result["errors"].append(f"Path does not exist: {folder_path}")
        return result

    if not root.is_dir():
        result["errors"].append(f"Path is not a directory: {folder_path}")
        return result

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        result["errors"].append("No sub-folders found. Each sub-folder should represent one clothing class.")
        return result

    matched_classes: dict = {}

    for subdir in sorted(subdirs):
        canonical = resolve_class_name(subdir.name)
        if canonical is None:
            result["unmatched_folders"].append(subdir.name)
            continue

        valid_images: list = []
        for f in sorted(subdir.iterdir()):
            if f.is_dir():
                continue
            if f.suffix.lower() not in SUPPORTED_EXT:
                result["unsupported_files"] += 1
                continue
            # Verify image can be opened (catches corrupt files)
            try:
                with Image.open(f) as img:
                    img.verify()
                valid_images.append(f)
            except Exception:
                result["corrupt_files"].append(str(f))

        if valid_images:
            matched_classes[canonical] = valid_images
        else:
            result["warnings"].append(
                f"Folder '{subdir.name}' matched '{canonical}' but contains no valid images — skipped."
            )

    if not matched_classes:
        result["errors"].append(
            "No valid images found in any matched class folder. "
            "Check that sub-folder names match clothing categories."
        )
        return result

    # Per-class count warnings
    for cls, imgs in matched_classes.items():
        if len(imgs) < 6:
            result["warnings"].append(
                f"Class '{cls}' has only {len(imgs)} image(s) — results will be unreliable (recommend ≥ 20)."
            )
        elif len(imgs) < 20:
            result["warnings"].append(
                f"Class '{cls}' has {len(imgs)} images — more data will improve accuracy (recommend ≥ 20)."
            )

    result["valid"] = True
    result["matched_classes"] = matched_classes
    result["total_valid"] = sum(len(v) for v in matched_classes.values())
    result["has_other"] = OTHER_CLASS in matched_classes
    result["num_classes"] = 11 if result["has_other"] else 10

    if result["has_other"]:
        n_other = len(matched_classes[OTHER_CLASS])
        if n_other < 10:
            result["warnings"].append(
                f"Category '{OTHER_CLASS}' has only {n_other} image(s). "
                "For reliable rejection of unrelated images, provide at least 20–50+ diverse non-clothing photos "
                "(wallpapers, animals, landscapes, vehicles, furniture)."
            )
    return result


# ─── Dataset splitting ────────────────────────────────────────────────────────

def split_dataset(matched_classes: dict, train_ratio: float = 0.70,
                  val_ratio: float = 0.15, seed: int = 42) -> tuple:
    """
    Stratified split per class.

    Returns (train_items, val_items, test_items) where each item is
    (path: Path, label_idx: int).
    Labels 0..9 correspond to the 10 clothing categories.
    Label 10 corresponds to 'Other / Not Clothing' if present.

    Warns if any class has < 6 images.
    Raises ValueError if any class has < 3 images (cannot split).
    """
    rng = random.Random(seed)
    class_names = get_class_names_for_dataset(matched_classes)

    train_items, val_items, test_items = [], [], []

    for class_name, paths in matched_classes.items():
        if class_name not in class_names:
            continue
        label_idx = class_names.index(class_name)
        n = len(paths)

        if n < 3:
            raise ValueError(
                f"Class '{class_name}' has only {n} image(s) — need at least 3 to split."
            )
        if n < 6:
            print(f"[dataset_utils] WARNING: Class '{class_name}' has only {n} images.")

        shuffled = list(paths)
        rng.shuffle(shuffled)

        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # test gets the remainder (at least 1)
        n_train = min(n_train, n - 2)  # guarantee at least 1 val, 1 test
        n_val   = min(n_val,   n - n_train - 1)

        for p in shuffled[:n_train]:
            train_items.append((p, label_idx))
        for p in shuffled[n_train:n_train + n_val]:
            val_items.append((p, label_idx))
        for p in shuffled[n_train + n_val:]:
            test_items.append((p, label_idx))

    return train_items, val_items, test_items


# ─── Item loading ─────────────────────────────────────────────────────────────

def load_and_preprocess_item(path: Path, label: int):
    """
    Opens a PIL image, calls preprocess(img, invert='auto', crop=True),
    normalises to float32 /255, adds channel dim → shape (28, 28, 1).
    Returns (array, label) or None on error.
    """
    from preprocess import preprocess  # local import to avoid circular at module load

    try:
        with Image.open(path) as img:
            img.load()  # force decode before closing
            small = preprocess(img, invert="auto", crop=True)   # (28, 28) uint8
    except Exception as exc:
        print(f"[dataset_utils] WARNING: Could not load {path}: {exc}")
        return None

    arr = (small.astype("float32") / 255.0)[..., np.newaxis]   # (28, 28, 1)
    return arr, label


# ─── Array building ───────────────────────────────────────────────────────────

def build_arrays(items: list) -> tuple:
    """
    Calls load_and_preprocess_item for every (path, label) in items.
    Skips None results with a printed warning.
    Returns:
        X : float32 ndarray, shape (N, 28, 28, 1)
        y : int   ndarray, shape (N,)
    """
    X_list, y_list = [], []

    for path, label in items:
        result = load_and_preprocess_item(path, label)
        if result is None:
            print(f"[dataset_utils] WARNING: Skipping {path}")
            continue
        arr, lbl = result
        X_list.append(arr)
        y_list.append(lbl)

    if not X_list:
        raise ValueError("No valid images could be loaded from the provided items.")

    X = np.stack(X_list, axis=0).astype("float32")  # (N, 28, 28, 1)
    y = np.array(y_list, dtype=np.int64)             # (N,)
    return X, y


# ─── Stats ────────────────────────────────────────────────────────────────────

def dataset_stats(scan_result: dict) -> dict:
    """Returns a summary dict: total, per_class counts, missing_clothing list, has_other."""
    matched = scan_result.get("matched_classes", {})
    per_class = {cls: len(imgs) for cls, imgs in matched.items()}
    missing_clothing = [c for c in CLOTHING_CLASSES if c not in matched]
    has_other = OTHER_CLASS in matched
    return {
        "total": sum(per_class.values()),
        "per_class": per_class,
        "missing_clothing": missing_clothing,
        "has_other": has_other,
        "num_classes": 11 if has_other else 10,
    }
