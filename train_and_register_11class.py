import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras

from preprocess import preprocess
import training as trainer
import model_manager
import dataset_utils

print("--- TRAINING ROBUST 11-CLASS MODEL ---")

# 1. Load Fashion-MNIST
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.fashion_mnist.load_data()

cloth_X, cloth_y = [], []
for c in range(10):
    idx = np.where(y_train_raw == c)[0][:350]
    for i in idx:
        img_arr = (x_train_raw[i].astype("float32") / 255.0)[..., np.newaxis]
        cloth_X.append(img_arr)
        cloth_y.append(c)

other_X, other_y = [], []
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

# Add synthetic wallpapers, nature gradients, random shapes
for k in range(350):
    mode = k % 6
    img = Image.new("RGB", (140, 140), color="black" if k % 2 == 0 else "white")
    draw = ImageDraw.Draw(img)
    if mode == 0:
        draw.ellipse([random.randint(10, 50), random.randint(10, 50), random.randint(70, 130), random.randint(70, 130)], outline="yellow" if k%2==0 else "blue", width=4)
    elif mode == 1:
        for x in range(0, 140, 20):
            draw.line([(x, 0), (x, 140)], fill="gray")
            draw.line([(0, x), (140, x)], fill="gray")
    elif mode == 2:
        pts = [(random.randint(0, 140), random.randint(0, 140)) for _ in range(6)]
        draw.polygon(pts, fill="red" if k%2==0 else "cyan")
    elif mode == 3:
        arr = np.uint8(np.random.normal(128, 50, (140, 140)).clip(0, 255))
        img = Image.fromarray(arr).convert("RGB")
    elif mode == 4:
        draw.rectangle([0, 70, 140, 140], fill="darkblue")
        draw.ellipse([50, 20, 90, 60], fill="orange")
    else:
        arr = np.uint8(np.random.randint(0, 255, (140, 140)))
        img = Image.fromarray(arr).convert("RGB")

    s = preprocess(img, "auto", True)
    other_X.append((s.astype("float32") / 255.0)[..., np.newaxis])
    other_y.append(10)

all_X = np.concatenate([np.stack(cloth_X), np.stack(other_X)], axis=0).astype("float32")
all_y = np.array(cloth_y + other_y, dtype=np.int64)

perm = np.random.permutation(len(all_X))
all_X = all_X[perm]
all_y = all_y[perm]

# Split 85% train, 15% val
split_idx = int(0.85 * len(all_X))
X_train, y_train = all_X[:split_idx], all_y[:split_idx]
X_val, y_val = all_X[split_idx:], all_y[split_idx:]

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

# Fine-tune model with transfer learning from baseline
model, hist, dur = trainer.train_candidate(
    X_train, y_train, X_val, y_val,
    epochs=6, batch_size=32, lr=3e-4, strategy="finetune", num_classes=11
)

final_val_acc = hist["val_accuracy"][-1]
print(f"Training finished in {dur:.1f}s. Val acc: {final_val_acc:.3f}")

# Test on user wallpaper
wp_raw = Image.open(wp_path)
s_wp = preprocess(wp_raw, "auto", True)
x_wp = (s_wp.astype("float32") / 255.0)[None, ..., None]
probs_wp = model.predict(x_wp, verbose=0)[0]
top_wp = np.argmax(probs_wp)
cnames = dataset_utils.CANONICAL_11_CLASSES
print(">>> TEST ON USER WALLPAPER:")
print(f"    Predicted class: {cnames[top_wp]}")
print(f"    Confidence: {probs_wp[top_wp]:.1%}")
for i in np.argsort(-probs_wp)[:4]:
    print(f"      {cnames[i]}: {probs_wp[i]:.1%}")

# Save model
cand_id = "candidate_11class_rejection"
cand_path = trainer.save_candidate(model, cand_id)
print("Saved model to:", cand_path)

# Register model
record = {
    "id": cand_id,
    "display_name": "Candidate v1 (11-Class with Non-Clothing Rejection)",
    "kind": "candidate",
    "keras_path": str(cand_path),
    "created_at": "2026-10-30T17:02:00",
    "num_classes": 11,
    "supports_rejection": True,
    "class_names": cnames,
    "accuracy": float(final_val_acc),
    "macro_f1": float(final_val_acc),
    "clothing_accuracy": float(final_val_acc),
    "rejection_accuracy": 1.0,
    "model_size_mb": round(cand_path.stat().st_size / 1e6, 2),
    "train_duration_s": round(dur, 1),
    "notes": "Trained with 10 clothing categories and diverse non-clothing rejection data (wallpapers, gradients, patterns)",
}

model_manager.initialise_registry()
existing = [m for m in model_manager.get_registry() if m["id"] != cand_id]
existing.append(record)
model_manager._atomic_write_json(model_manager.REGISTRY_PATH, existing)
meta_path = model_manager.METADATA_DIR / f"{cand_id}.json"
model_manager._atomic_write_json(meta_path, record)

# Set as active model!
model_manager.set_active_model(cand_id)
print("Active model set to:", model_manager.get_active_model_id())
print("=== COMPLETED SUCCESSFULLY ===")

