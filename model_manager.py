"""
model_manager.py — Model registry, active-model tracking, and Keras model loading.

All JSON writes are ATOMIC (write temp → os.replace) so a partial write can never
corrupt the registry or the active-model pointer.
"""
import json
import os
import uuid
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
MODELS_DIR      = Path("models")
VERSIONS_DIR    = MODELS_DIR / "versions"
METADATA_DIR    = MODELS_DIR / "metadata"
REGISTRY_PATH   = MODELS_DIR / "model_registry.json"
ACTIVE_PATH     = MODELS_DIR / "active_model.json"

BASELINE_ID     = "baseline_v0"
BASELINE_PATH   = Path("fashion_cnn.keras")

# ─── Baseline registry entry ──────────────────────────────────────────────────
_BASELINE_ENTRY = {
    "id":               BASELINE_ID,
    "display_name":     "Baseline (Fashion-MNIST)",
    "kind":             "baseline",
    "keras_path":       str(BASELINE_PATH),
    "created_at":       "2026-10-30T00:00:00",
    "dataset_path":     None,
    "train_count":      None,
    "val_count":        None,
    "test_count":       None,
    "class_dist":       None,
    "accuracy":         None,
    "macro_f1":         None,
    "weighted_f1":      None,
    "precision":        None,
    "recall":           None,
    "train_duration_s": None,
    "model_size_mb":    None,
    "num_params":       None,
    "train_config":     None,
    "num_classes":      10,
    "supports_rejection": False,
    "class_names":      [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
    "notes":            "Original model trained on Fashion-MNIST (93.9% test accuracy). 10 clothing classes, no rejection.",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically: write to a temp file, then rename."""
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ─── Public API ───────────────────────────────────────────────────────────────

def initialise_registry() -> None:
    """
    Creates:
      models/versions/   models/metadata/
      models/model_registry.json  (baseline entry if not present)
      models/active_model.json    (points to baseline_v0 if not present)
    Idempotent — safe to call on every app start.
    """
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Seed registry
    if not REGISTRY_PATH.exists():
        _atomic_write_json(REGISTRY_PATH, [_BASELINE_ENTRY])
    else:
        # Ensure baseline entry is present (in case registry was manually edited)
        registry = _read_json(REGISTRY_PATH)
        ids = {entry["id"] for entry in registry}
        if BASELINE_ID not in ids:
            registry.insert(0, _BASELINE_ENTRY)
            _atomic_write_json(REGISTRY_PATH, registry)
        else:
            # Upgrade baseline entry fields if needed
            updated = False
            for entry in registry:
                if entry["id"] == BASELINE_ID and ("num_classes" not in entry or "class_names" not in entry):
                    entry.update({
                        "num_classes": 10,
                        "supports_rejection": False,
                        "class_names": _BASELINE_ENTRY["class_names"],
                    })
                    updated = True
            if updated:
                _atomic_write_json(REGISTRY_PATH, registry)

    # Seed active model pointer
    if not ACTIVE_PATH.exists():
        _atomic_write_json(ACTIVE_PATH, {"active_model_id": BASELINE_ID})

    # Seed baseline metadata file
    baseline_meta = METADATA_DIR / f"{BASELINE_ID}.json"
    if not baseline_meta.exists():
        _atomic_write_json(baseline_meta, _BASELINE_ENTRY)
    else:
        b_data = _read_json(baseline_meta)
        if "num_classes" not in b_data:
            b_data.update({
                "num_classes": 10,
                "supports_rejection": False,
                "class_names": _BASELINE_ENTRY["class_names"],
            })
            _atomic_write_json(baseline_meta, b_data)


def get_registry() -> list:
    """Returns the full registry as a list of dicts."""
    if not REGISTRY_PATH.exists():
        return [_BASELINE_ENTRY]
    return _read_json(REGISTRY_PATH)


def get_active_model_id() -> str:
    """Returns the currently active model ID string."""
    if not ACTIVE_PATH.exists():
        return BASELINE_ID
    data = _read_json(ACTIVE_PATH)
    return data.get("active_model_id", BASELINE_ID)


def set_active_model(model_id: str) -> None:
    """Atomically updates active_model.json."""
    _atomic_write_json(ACTIVE_PATH, {"active_model_id": model_id})


def load_active_keras_model():
    """Loads and returns the Keras model for the currently active model ID."""
    return load_model_by_id(get_active_model_id())


def load_model_by_id(model_id: str):
    """
    Loads and returns the Keras model for the given model ID.
    For baseline_v0 → loads fashion_cnn.keras directly.
    For others     → loads models/versions/{model_id}.keras.
    """
    from tensorflow import keras  # deferred import for fast module loading

    if model_id == BASELINE_ID:
        return keras.models.load_model(str(BASELINE_PATH))

    rec = get_model_record(model_id)
    if rec is None:
        raise ValueError(f"No registry entry found for model_id={model_id!r}")

    keras_path = Path(rec["keras_path"])
    if not keras_path.is_absolute():
        # Paths in registry are stored relative to project root
        keras_path = keras_path.resolve()
    return keras.models.load_model(str(keras_path))


def register_candidate(record: dict) -> None:
    """Appends a new candidate record to the registry and writes individual metadata (atomic write)."""
    registry = get_registry()
    registry.append(record)
    _atomic_write_json(REGISTRY_PATH, registry)

    meta_path = METADATA_DIR / f"{record['id']}.json"
    _atomic_write_json(meta_path, record)


def get_model_record(model_id: str):
    """Returns the registry dict for the given model_id, or None if not found."""
    for entry in get_registry():
        if entry["id"] == model_id:
            return entry
    return None


def list_models() -> list:
    """Returns all registry entries sorted by created_at (newest first)."""
    registry = get_registry()
    return sorted(registry, key=lambda e: e.get("created_at", ""), reverse=True)
