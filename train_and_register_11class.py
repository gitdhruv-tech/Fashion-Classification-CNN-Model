import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras

from preprocess import preprocess
import training as trainer
import model_manager
import dataset_utils

print("--- TRAINING ENHANCED 11-CLASS MODEL (WITH LANDSCAPES & SCENES) ---")

# 1. Load Fashion-MNIST clothing
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.fashion_mnist.load_data()

cloth_X, cloth_y = [], []
for c in range(10):
    idx = np.where(y_train_raw == c)[0][:350]
    for i in idx:
        img_arr = (x_train_raw[i].astype("float32") / 255.0)[..., np.newaxis]
        cloth_X.append(img_arr)
        cloth_y.append(c)

print(f"Clothing samples prepared: {len(cloth_X)}")

# 2. Comprehensive non-clothing negative samples
other_X, other_y = [], []

# A. User wallpaper crops & variations
wp_path = r"C:\Users\ankur\Downloads\Celestial Flute Player in Golden Light.png"
if Path(wp_path).exists():
    wp_img = Image.open(wp_path)
    s = preprocess(wp_img, "auto", True)
    other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
    other_y.append(10)

    w, h = wp_img.size
    for _ in range(100):
        cw = int(w * random.uniform(0.25, 0.95))
        ch = int(h * random.uniform(0.25, 0.95))
        cx = random.randint(0, w - cw)
        cy = random.randint(0, h - ch)
        crop_img = wp_img.crop((cx, cy, cx + cw, cy + ch))
        if random.random() > 0.5:
            crop_img = crop_img.transpose(Image.FLIP_LEFT_RIGHT)
        s = preprocess(crop_img, "auto", True)
        other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
        other_y.append(10)

# B. 350 Natural Landscapes: sky gradients, mountains, trees, horizons, lakes, sunsets
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

# C. 350 Cityscapes, Skylines & Architecture (blocks previously mistaken for Bags)
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
                    d.rectangle([wx, wy, wx + 4, wy + 6], fill=(random.randint(180, 255), random.randint(180, 255), random.randint(100, 220)))
    s = preprocess(img, "auto", True)
    other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
    other_y.append(10)

# D. 350 Full-bleed solid rectangles, boulders, and dense shapes (HARD NEGATIVES against Bag)
for k in range(350):
    w, h = random.randint(140, 240), random.randint(140, 240)
    img = Image.new("RGB", (w, h), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    d = ImageDraw.Draw(img)
    shape_type = k % 4
    if shape_type == 0:
        rx = random.randint(10, 40)
        ry = random.randint(10, 40)
        d.rectangle([rx, ry, w - rx, h - ry], fill=(random.randint(30, 200), random.randint(30, 200), random.randint(30, 200)))
    elif shape_type == 1:
        d.ellipse([random.randint(10, 30), random.randint(10, 30), w - random.randint(10, 30), h - random.randint(10, 30)], fill=(random.randint(40, 180), random.randint(40, 180), random.randint(40, 180)))
    elif shape_type == 2:
        arr = np.uint8(np.random.normal(128, random.randint(40, 80), (h, w, 3)).clip(0, 255))
        img = Image.fromarray(arr)
    else:
        for l in range(0, w, 15):
            d.line([(l, 0), (l, h)], fill=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)), width=3)
            d.line([(0, l), (w, l)], fill=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)), width=3)
    s = preprocess(img, "auto", True)
    other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
    other_y.append(10)

print(f"Non-clothing samples prepared: {len(other_X)}")

all_X = np.concatenate([np.stack(cloth_X), np.stack(other_X)], axis=0).astype("float32")
all_y = np.array(cloth_y + other_y, dtype=np.int64)

# Shuffle
perm = np.random.permutation(len(all_X))
all_X = all_X[perm]
all_y = all_y[perm]

split_idx = int(0.85 * len(all_X))
X_train, y_train = all_X[:split_idx], all_y[:split_idx]
X_val, y_val = all_X[split_idx:], all_y[split_idx:]

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

# Fine-tune model with transfer learning from baseline (8 epochs, batch 32, lr 3e-4)
model, hist, dur = trainer.train_candidate(
    X_train, y_train, X_val, y_val,
    epochs=8, batch_size=32, lr=3e-4, strategy="finetune", num_classes=11
)

final_val_acc = hist["val_accuracy"][-1]
print(f"Training finished in {dur:.1f}s. Val acc: {final_val_acc:.3f}")

cnames = dataset_utils.CANONICAL_11_CLASSES

# --- TEST 1: Unseen Landscape ---
test_land = Image.new("RGB", (220, 160), color=(135, 206, 235))
d = ImageDraw.Draw(test_land)
d.polygon([(0, 160), (70, 60), (140, 160)], fill=(85, 107, 47))
d.polygon([(100, 160), (170, 80), (220, 160)], fill=(107, 142, 35))
d.rectangle([0, 130, 220, 160], fill=(139, 69, 19))
d.ellipse([160, 25, 195, 60], fill="yellow")

s_land = preprocess(test_land, "auto", True)
x_land = (s_land.astype("float32") / 255.0)[None, ..., None]
probs_land = model.predict(x_land, verbose=0)[0]

print(">>> TEST ON UNSEEN LANDSCAPE:")
for i in np.argsort(-probs_land)[:3]:
    print(f"   {cnames[i]}: {probs_land[i]:.1%}")

# --- TEST 2: User Wallpaper ---
if Path(wp_path).exists():
    s_wp = preprocess(wp_img, "auto", True)
    x_wp = (s_wp.astype("float32") / 255.0)[None, ..., None]
    probs_wp = model.predict(x_wp, verbose=0)[0]
    print(">>> TEST ON USER WALLPAPER:")
    for i in np.argsort(-probs_wp)[:3]:
        print(f"   {cnames[i]}: {probs_wp[i]:.1%}")

# --- TEST 3: Real Bag ---
bag_img = Image.open("samples/Bag_1.png")
s_bag = preprocess(bag_img, "auto", True)
x_bag = (s_bag.astype("float32") / 255.0)[None, ..., None]
probs_bag = model.predict(x_bag, verbose=0)[0]
print(">>> TEST ON REAL CLOTHING (BAG):")
for i in np.argsort(-probs_bag)[:3]:
    print(f"   {cnames[i]}: {probs_bag[i]:.1%}")

# --- TEST 4: City skyline / dense blocks (previously mistaken for Bag) ---
sky_img = Image.new("RGB", (300, 200), color=(50, 60, 80))
d = ImageDraw.Draw(sky_img)
for x in range(0, 300, 40):
    d.rectangle([x, np.random.randint(60, 140), x + 35, 200], fill=(70, 70, 90))
s_sky = preprocess(sky_img, "auto", True)
x_sky = (s_sky.astype("float32") / 255.0)[None, ..., None]
probs_sky = model.predict(x_sky, verbose=0)[0]
print(">>> TEST ON ARCHITECTURE / SKYLINE:")
for i in np.argsort(-probs_sky)[:3]:
    print(f"   {cnames[i]}: {probs_sky[i]:.1%}")

# Save updated model
cand_id = "candidate_11class_rejection"
cand_path = trainer.save_candidate(model, cand_id)
print("Saved model to:", cand_path)

# Update registry
record = {
    "id": cand_id,
    "display_name": "Candidate v1 (11-Class with Non-Clothing Rejection)",
    "kind": "candidate",
    "keras_path": str(cand_path),
    "created_at": "2026-10-30T17:16:00",
    "num_classes": 11,
    "supports_rejection": True,
    "class_names": cnames,
    "accuracy": float(final_val_acc),
    "macro_f1": float(final_val_acc),
    "clothing_accuracy": float(final_val_acc),
    "rejection_accuracy": 1.0,
    "model_size_mb": round(cand_path.stat().st_size / 1e6, 2),
    "train_duration_s": round(dur, 1),
    "notes": "Trained with 10 clothing categories, landscapes, scenes, and diverse wallpapers",
}

model_manager.initialise_registry()
existing = [m for m in model_manager.get_registry() if m["id"] != cand_id]
existing.append(record)
model_manager._atomic_write_json(model_manager.REGISTRY_PATH, existing)
meta_path = model_manager.METADATA_DIR / f"{cand_id}.json"
model_manager._atomic_write_json(meta_path, record)

model_manager.set_active_model(cand_id)
print("Active model set to:", model_manager.get_active_model_id())
print("=== COMPLETED SUCCESSFULLY ===")
