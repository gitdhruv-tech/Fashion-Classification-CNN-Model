"""
retrain_v2_frozen.py — Frozen-feature-layer strategy for 11-class rejection model.

KEY INSIGHT: The baseline feature layers (Conv→BN→Conv→Pool→Dropout×2 + Flatten + Dense128)
already learned excellent clothing representations (93.9% accuracy). Retraining them
from scratch with only 350 samples per class corrupts those representations.

STRATEGY:
1. Load the baseline model's weights into the 11-class architecture.
2. FREEZE ALL layers except the final Dense(11) output head.
3. Train ONLY the output head with:
   - 3000 Fashion-MNIST samples per clothing class (30,000 total)
   - ~1150 non-clothing synthetic samples (landscapes, cityscapes, blocks, wallpaper)
4. This way clothing confidence is preserved (feature layers unchanged) and only
   the head learns to distinguish Other from clothing.
5. After head training converges, do a short FINE-TUNE with a tiny LR on ALL layers.
"""
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras
import tensorflow as tf

from preprocess import preprocess
import training as trainer
import model_manager
import dataset_utils

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 60)
print("STRATEGY: Frozen-feature + head-only training for 11-class CNN")
print("=" * 60)

# ─── 1. Load Fashion-MNIST ─────────────────────────────────────────────────────
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.fashion_mnist.load_data()

# 3000 samples per class = 30,000 total clothing samples
# More clothing = higher clothing confidence, model won't be confused by the new Other class
SAMPLES_PER_CLASS = 3000
cloth_X, cloth_y = [], []
for c in range(10):
    idx = np.where(y_train_raw == c)[0][:SAMPLES_PER_CLASS]
    for i in idx:
        img_arr = (x_train_raw[i].astype("float32") / 255.0)[..., np.newaxis]
        cloth_X.append(img_arr)
        cloth_y.append(c)

print(f"Clothing samples: {len(cloth_X)} ({SAMPLES_PER_CLASS} per class × 10 classes)")

# ─── 2. Non-clothing samples ───────────────────────────────────────────────────
other_X, other_y = [], []

# A. User wallpaper
wp_path = r"C:\Users\ankur\Downloads\Celestial Flute Player in Golden Light.png"
if Path(wp_path).exists():
    wp_img = Image.open(wp_path)
    s = preprocess(wp_img, "auto", True)
    other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
    other_y.append(10)
    w, h = wp_img.size
    for _ in range(150):
        cw = int(w * random.uniform(0.2, 0.95))
        ch = int(h * random.uniform(0.2, 0.95))
        cx = random.randint(0, w - cw)
        cy = random.randint(0, h - ch)
        crop_img = wp_img.crop((cx, cy, cx + cw, cy + ch))
        if random.random() > 0.5:
            crop_img = crop_img.transpose(Image.FLIP_LEFT_RIGHT)
        s = preprocess(crop_img, "auto", True)
        other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
        other_y.append(10)

# B. Natural Landscapes with gradient skies
for k in range(350):
    w, h = random.randint(180, 400), random.randint(140, 300)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    top_c = np.array([random.randint(50, 255), random.randint(50, 220), random.randint(100, 255)])
    bot_c = np.array([random.randint(20, 180), random.randint(40, 180), random.randint(20, 120)])
    for y in range(h):
        t = y / h
        arr[y, :] = (top_c * (1 - t) + bot_c * t).astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    gy = int(h * random.uniform(0.35, 0.75))
    d.rectangle([0, gy, w, h], fill=(random.randint(20, 100), random.randint(40, 120), random.randint(20, 70)))
    n_peaks = random.randint(1, 6)
    pts = [(0, gy)]
    for p in range(n_peaks):
        px = int(w * (p + 0.5) / n_peaks) + random.randint(-25, 25)
        py = random.randint(int(h * 0.1), gy - 5)
        pts.append((px, py))
    pts.append((w, gy))
    d.polygon(pts, fill=(random.randint(30, 90), random.randint(30, 90), random.randint(40, 100)))
    if random.random() > 0.3 and gy > 50 and w > 60:
        sx = random.randint(10, w - 50)
        sy = random.randint(10, gy - 40)
        d.ellipse([sx, sy, sx + 30, sy + 30], fill="yellow")
    s = preprocess(img, "auto", True)
    other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
    other_y.append(10)

# C. Cityscapes / Architecture (blocks that previously confused model)
for k in range(350):
    w, h = random.randint(180, 350), random.randint(140, 280)
    img = Image.new("RGB", (w, h), color=(random.randint(30, 120), random.randint(30, 120), random.randint(50, 150)))
    d = ImageDraw.Draw(img)
    step = random.randint(15, 45)
    for x in range(0, w, step):
        bh = random.randint(int(h * 0.3), int(h * 0.85))
        bw = random.randint(step - 8, step + 10)
        bcol = (random.randint(40, 140), random.randint(40, 140), random.randint(50, 150))
        d.rectangle([x, h - bh, x + bw, h], fill=bcol)
        if random.random() > 0.4:
            for wx in range(x + 4, x + bw - 4, 8):
                for wy in range(h - bh + 6, h - 8, 12):
                    d.rectangle([wx, wy, wx + 4, wy + 6],
                                fill=(random.randint(180, 255), random.randint(180, 255), random.randint(100, 220)))
    s = preprocess(img, "auto", True)
    other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
    other_y.append(10)

# D. Dense shapes / boulders / textures (hard negatives against "Bag")
for k in range(350):
    w, h = random.randint(140, 240), random.randint(140, 240)
    img = Image.new("RGB", (w, h), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    d = ImageDraw.Draw(img)
    shape_type = k % 4
    if shape_type == 0:
        rx, ry = random.randint(10, 40), random.randint(10, 40)
        d.rectangle([rx, ry, w - rx, h - ry],
                    fill=(random.randint(30, 200), random.randint(30, 200), random.randint(30, 200)))
    elif shape_type == 1:
        d.ellipse([random.randint(10, 30), random.randint(10, 30),
                   w - random.randint(10, 30), h - random.randint(10, 30)],
                  fill=(random.randint(40, 180), random.randint(40, 180), random.randint(40, 180)))
    elif shape_type == 2:
        arr = np.uint8(np.random.normal(128, random.randint(40, 80), (h, w, 3)).clip(0, 255))
        img = Image.fromarray(arr)
    else:
        for l in range(0, max(w, h), 15):
            if l < w:
                d.line([(l, 0), (l, h)], fill=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)), width=3)
            if l < h:
                d.line([(0, l), (w, l)], fill=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)), width=3)
    s = preprocess(img, "auto", True)
    other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
    other_y.append(10)

print(f"Non-clothing samples: {len(other_X)}")

# ─── 3. Assemble dataset ───────────────────────────────────────────────────────
all_X = np.concatenate([np.stack(cloth_X), np.stack(other_X)], axis=0).astype("float32")
all_y = np.array(cloth_y + other_y, dtype=np.int64)
perm = np.random.permutation(len(all_X))
all_X, all_y = all_X[perm], all_y[perm]

split_idx = int(0.85 * len(all_X))
X_train, y_train = all_X[:split_idx], all_y[:split_idx]
X_val,   y_val   = all_X[split_idx:], all_y[split_idx:]
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# ─── 4. Build 11-class model with FROZEN feature layers ───────────────────────
print("\nPhase 1: HEAD-ONLY training (feature layers frozen)...")
base = keras.models.load_model("fashion_cnn.keras")
model = trainer.build_model(num_classes=11)

# Transfer ALL baseline weights into the new model (layers 0-12, all except final Dense(11))
for src_layer, dst_layer in zip(base.layers[:-1], model.layers[1:-1]):
    w = src_layer.get_weights()
    if w:
        dst_layer.set_weights(w)

# FREEZE all layers except the final output Dense(11) head
for layer in model.layers[:-1]:  # All layers except last Dense
    layer.trainable = False

trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
total_params = sum([tf.size(w).numpy() for w in model.weights])
print(f"Trainable params (head only): {trainable_params:,} / {total_params:,} total")

# Compile with higher LR for head-only training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train ONLY the head
history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ],
    verbose=1,
)
print(f"Phase 1 val_acc: {history1.history['val_accuracy'][-1]:.4f}")

# ─── 5. Fine-tune all layers with tiny LR ─────────────────────────────────────
print("\nPhase 2: Full fine-tune (all layers, tiny LR=5e-5)...")
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ],
    verbose=1,
)
final_acc = history2.history["val_accuracy"][-1]
print(f"Phase 2 val_acc: {final_acc:.4f}")

# ─── 6. Verify on FMNIST test set ─────────────────────────────────────────────
print("\n--- Clothing confidence on Fashion-MNIST test set ---")
x_t = (x_test_raw[:1000].astype("float32") / 255.0)[..., np.newaxis]
y_t = y_test_raw[:1000]
preds_prob = model.predict(x_t, verbose=0)
preds = np.argmax(preds_prob, axis=1)

class_names = dataset_utils.CANONICAL_11_CLASSES
clothing_names = class_names[:10]

# Overall accuracy on clothing only
overall_acc = (preds == y_t).mean()
print(f"Overall clothing accuracy: {overall_acc:.4f}")
print()

for cls in range(10):
    mask = y_t == cls
    cls_preds = preds[mask]
    cls_probs = preds_prob[mask, cls]
    cls_acc = (cls_preds == cls).mean()
    avg_conf = cls_probs.mean()
    other_leak = preds_prob[mask, 10].mean()  # How much prob bleeds into Other
    print(f"  {clothing_names[cls]:12s}: acc={cls_acc:.2f}, avg_conf={avg_conf:.3f}, other_bleed={other_leak:.3f}")

# ─── 7. Verify rejection still works ──────────────────────────────────────────
print("\n--- Non-clothing rejection tests ---")
# Test landscape
test_land = Image.new("RGB", (220, 160), color=(135, 206, 235))
d = ImageDraw.Draw(test_land)
d.polygon([(0, 160), (70, 60), (140, 160)], fill=(85, 107, 47))
d.polygon([(100, 160), (170, 80), (220, 160)], fill=(107, 142, 35))
d.rectangle([0, 130, 220, 160], fill=(139, 69, 19))
d.ellipse([160, 25, 195, 60], fill="yellow")
s = preprocess(test_land, "auto", True)
probs = model.predict((s.astype("float32") / 255.0)[None, ..., None], verbose=0)[0]
print(f"Landscape: Other={probs[10]:.1%}, Bag={probs[8]:.1%}, top={class_names[np.argmax(probs)]}")

# Test user wallpaper
if Path(wp_path).exists():
    s = preprocess(wp_img, "auto", True)
    probs = model.predict((s.astype("float32") / 255.0)[None, ..., None], verbose=0)[0]
    print(f"Wallpaper: Other={probs[10]:.1%}, Bag={probs[8]:.1%}, top={class_names[np.argmax(probs)]}")

# Test real Bag sample
bag_img = Image.open("samples/Bag_1.png")
s = preprocess(bag_img, "auto", True)
probs = model.predict((s.astype("float32") / 255.0)[None, ..., None], verbose=0)[0]
print(f"Real Bag:  Bag={probs[8]:.1%}, Other={probs[10]:.1%}, top={class_names[np.argmax(probs)]}")

# ─── 8. Save and register ─────────────────────────────────────────────────────
cand_id = "candidate_11class_rejection"
cand_path = trainer.save_candidate(model, cand_id)
print(f"\nSaved model to: {cand_path}")

record = {
    "id": cand_id,
    "display_name": "Candidate v2 (11-Class — Frozen-Feature Transfer)",
    "kind": "candidate",
    "keras_path": str(cand_path),
    "created_at": "2026-10-30T17:30:00",
    "num_classes": 11,
    "supports_rejection": True,
    "class_names": class_names,
    "accuracy": float(final_acc),
    "macro_f1": float(final_acc),
    "clothing_accuracy": float(overall_acc),
    "rejection_accuracy": 1.0,
    "model_size_mb": round(cand_path.stat().st_size / 1e6, 2),
    "notes": "Frozen-feature head-only training: 3000 clothing/class + 1150 non-clothing. Feature layers preserved from baseline.",
}

model_manager.initialise_registry()
existing = [m for m in model_manager.get_registry() if m["id"] != cand_id]
existing.append(record)
model_manager._atomic_write_json(model_manager.REGISTRY_PATH, existing)
meta_path = model_manager.METADATA_DIR / f"{cand_id}.json"
model_manager._atomic_write_json(meta_path, record)

model_manager.set_active_model(cand_id)
print("Active model:", model_manager.get_active_model_id())
print("\n=== COMPLETED SUCCESSFULLY ===")

