"""
training.py — CNN model building, fine-tuning, training loop, and candidate saving.

Safety rules enforced here:
  - fashion_cnn.keras is NEVER written to (only read, into a separate variable).
  - X_test is never accepted as a parameter.
  - All candidate models go to models/versions/{model_id}.keras.
"""
import time
from pathlib import Path

import numpy as np

SEED = 42
BASELINE_PATH = Path("fashion_cnn.keras")


# ─── Architecture ─────────────────────────────────────────────────────────────

def build_model(num_classes: int = 10):
    """
    CNN architecture from the training notebook, parameterized by num_classes (10 or 11).
    Compiled with Adam(lr=1e-3), sparse_categorical_crossentropy.
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    # Block 2
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    # Head
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── Candidate model factory ──────────────────────────────────────────────────

def get_candidate_model(strategy: str = "finetune", lr: float = 1e-4, num_classes: int = 10):
    """
    Returns a fresh Keras model ready for training.

    strategy='finetune':
      - num_classes == 10: loads BASELINE_PATH into a NEW variable and recompiles with Adam(lr).
      - num_classes == 11: builds fresh 11-class model, transfers all feature & embedding weights
        from BASELINE_PATH (all layers except the final output dense), and initializes
        a fresh 11-neuron softmax head. Recompiles with Adam(lr).
    strategy='scratch': builds from random init with num_classes and compiles with Adam(lr).
    """
    from tensorflow import keras

    if strategy == "finetune":
        if num_classes == 10:
            # Load into a new model object — does NOT hold a file-write handle
            model = keras.models.load_model(str(BASELINE_PATH))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        else:
            # 11-class model with transferred feature representations
            base = keras.models.load_model(str(BASELINE_PATH))
            model = build_model(num_classes=num_classes)
            # Transfer all convolutional, batchnorm, and dense feature layers before the head
            for src_layer, dst_layer in zip(base.layers[:-1], model.layers[1:-1]):
                w = src_layer.get_weights()
                if w:
                    dst_layer.set_weights(w)

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
    else:
        model = build_model(num_classes=num_classes)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


# ─── Progress callback ────────────────────────────────────────────────────────

def _make_progress_callback(progress_fn, total_epochs: int):
    """
    Returns a Keras Callback that calls progress_fn(epoch_num, total, logs)
    at the end of every epoch.  progress_fn may be None.
    """
    from tensorflow import keras

    class _ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_fn is not None:
                progress_fn(epoch + 1, total_epochs, logs or {})

    return _ProgressCallback()


# ─── Standard callbacks ───────────────────────────────────────────────────────

def make_callbacks(patience_stop: int = 4, patience_lr: int = 2) -> list:
    """Returns [EarlyStopping, ReduceLROnPlateau]."""
    from tensorflow import keras

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_stop,
        restore_best_weights=True,
        verbose=0,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience_lr,
        verbose=0,
    )
    return [early_stop, reduce_lr]


# ─── Training entry point ─────────────────────────────────────────────────────

def train_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    epochs:           int   = 25,
    batch_size:       int   = 32,
    lr:               float = 1e-4,
    strategy:         str   = "finetune",
    num_classes:      int   = 10,
    progress_callback=None,         # callable(epoch_num, total_epochs, logs) | None
) -> tuple:
    """
    Trains a candidate model and returns (trained_model, history_dict, duration_s).

    history_dict keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss' (lists).
    Raises RuntimeError on failure.
    NEVER writes to fashion_cnn.keras.
    NEVER receives test data.
    """
    try:
        model = get_candidate_model(strategy=strategy, lr=lr, num_classes=num_classes)

        callbacks = make_callbacks()
        if progress_callback is not None:
            callbacks.append(_make_progress_callback(progress_callback, epochs))

        t_start = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        duration = time.time() - t_start

        history_dict = {
            "accuracy":     history.history.get("accuracy", []),
            "val_accuracy": history.history.get("val_accuracy", []),
            "loss":         history.history.get("loss", []),
            "val_loss":     history.history.get("val_loss", []),
        }

        return model, history_dict, duration

    except Exception as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc


# ─── Saving ───────────────────────────────────────────────────────────────────

def save_candidate(model, model_id: str) -> Path:
    """
    Saves the model to models/versions/{model_id}.keras.
    Creates the directory if needed.  Returns the saved path.
    """
    save_dir = Path("models") / "versions"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{model_id}.keras"
    model.save(str(save_path))
    return save_path
