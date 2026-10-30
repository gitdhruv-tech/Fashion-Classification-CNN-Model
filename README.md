# Fashion Item Classifier — CNN with Non-Clothing Rejection

A Streamlit web application that classifies clothing images using a Convolutional Neural Network trained on Fashion-MNIST. Supports custom dataset retraining, model comparison, and optional non-clothing image rejection.

---

## Features

- **Image Classification** — Classifies clothing into 10 categories with confidence scores and uncertainty margin
- **Non-Clothing Rejection** — An optional 11-class model detects and rejects unrelated images (landscapes, wallpapers, animals, vehicles, etc.) instead of forcing them into a clothing category
- **Custom Dataset Retraining** — Train a new model on your own labelled clothing images via a UI, with live epoch progress
- **Model Comparison** — Side-by-side inference and evaluation between the baseline and any trained candidate model
- **Model Version Management** — All trained models are stored and can be activated or rolled back at any time; the original baseline is never overwritten

---

## Project Structure

```
Fashion-Classification-CNN-Model/
├── app.py                        # Streamlit application (all pages)
├── preprocess.py                 # Image preprocessing pipeline (grayscale, crop, resize to 28×28)
├── training.py                   # Model architecture, training loop, callbacks
├── dataset_utils.py              # Dataset folder scanner, class mapper, image loader
├── evaluation.py                 # Metrics: accuracy, precision, recall, F1, confusion matrix
├── model_manager.py              # Model registry, active model, version storage
├── requirements.txt              # Python dependencies
├── fashion_cnn.keras             # Pre-trained baseline (Fashion-MNIST, 93.9% test accuracy)
├── models/
│   ├── model_registry.json       # Registry of all model versions
│   ├── active_model.json         # Currently active model pointer
│   ├── metadata/                 # Per-model metadata JSON files
│   └── versions/                 # Trained candidate .keras files
├── samples/                      # Built-in sample images for demo
└── cnn training and testing/     # Original Colab training notebooks
```

---

## Quick Start

**Requirements:** Python 3.10 / 3.11 / 3.12

```bash
# 1. Clone the repo
git clone https://github.com/gitdhruv-tech/Fashion-Classification-CNN-Model.git
cd Fashion-Classification-CNN-Model

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser. The first launch is slow (TensorFlow loads). Predictions take a fraction of a second after that.

---

## The 10 Clothing Classes

| Label | Folder name (for custom dataset) |
|---|---|
| T-shirt/top | `T-shirt_top` |
| Trouser | `Trouser` |
| Pullover | `Pullover` |
| Dress | `Dress` |
| Coat | `Coat` |
| Sandal | `Sandal` |
| Shirt | `Shirt` |
| Sneaker | `Sneaker` |
| Bag | `Bag` |
| Ankle boot | `Ankle_boot` |

---

## Custom Dataset Retraining

### Dataset Folder Structure

```
YourDataset/
├── T-shirt_top/      ← maps to "T-shirt/top"
├── Trouser/
├── Pullover/
├── Dress/
├── Coat/
├── Sandal/
├── Shirt/
├── Sneaker/
├── Bag/
├── Ankle_boot/       ← maps to "Ankle boot"
└── Other/            ← optional: maps to "Other / Not Clothing"
```

- Folder naming is case-insensitive and separator-insensitive (`T-shirt_top`, `tshirt_top`, etc. all work)
- Supported image formats: **JPG, JPEG, PNG, WEBP**
- Minimum ~10 images per class recommended; 50+ per class for better accuracy
- If an `Other` folder is present, an **11-class model** with non-clothing rejection is trained automatically

### Training Workflow

1. Open the app and navigate to **🏋️ Model Training** in the sidebar
2. Enter your dataset folder path and click **Validate Dataset** — the app scans and shows per-class image counts and any issues
3. Configure training parameters (epochs, batch size, learning rate)
4. Click **Start Training** — a live progress bar shows each epoch's metrics
5. After training, click **Evaluate & Save** to run full evaluation
6. Navigate to **📊 Model Comparison** to compare accuracy, F1, and confusion matrices
7. Navigate to **🗂️ Model Management** to activate the new model or roll back to any previous version

---

## Non-Clothing Rejection

When an 11-class model is active, the prediction pipeline uses multiple gates to reject non-clothing images:

1. **Direct rejection** — Model outputs `Other / Not Clothing` as the top class
2. **Sensitivity threshold** *(sidebar slider, default 20%)* — If `P(Other) ≥ threshold`, the image is rejected even if a clothing class had a slight plurality
3. **Scene border variance gate** — Scenes and landscapes have busy pixel content all the way to the image border; isolated garments do not. High border variance combined with elevated `Other` probability triggers rejection
4. **Ambiguity gate** — If the gap between the top clothing class and `Other` is narrow and clothing confidence is low, the image is conservatively rejected

The **10-class baseline** cannot reject non-clothing images — it will always assign one of the 10 clothing labels. Switching to the 11-class candidate enables rejection.

---

## Model Architecture

The CNN follows the architecture from the training notebook:

```
Input (28×28×1)
  → Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool(2×2) → Dropout(0.25)
  → Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool(2×2) → Dropout(0.25)
  → Flatten → Dense(128) → Dropout(0.4)
  → Dense(10 or 11, softmax)
```

**Baseline:** Trained on full Fashion-MNIST (60,000 images) — 93.9% test accuracy.

**Candidate (11-class):** Uses a two-phase frozen-feature transfer learning strategy:
- **Phase 1** — All convolutional and embedding layers are frozen. Only the final Dense head is trained on clothing + non-clothing data. Feature representations from the baseline are fully preserved.
- **Phase 2** — Full fine-tune with a very small learning rate (`5e-5`) to make small calibration adjustments without forgetting clothing features.

This approach keeps per-class clothing confidence near baseline levels while adding the ability to reject non-clothing images.

---

## Sidebar Settings

| Setting | Default | Description |
|---|---|---|
| Auto-tag confidence threshold | 90% | Below this, prediction is flagged for human review |
| Uncertainty margin threshold | 20% | Minimum gap between top and 2nd class; below this triggers review |
| Non-clothing rejection sensitivity | 20% | If P(Other) ≥ this, image is rejected as non-clothing |
| Invert colours | Auto | Auto-detects whether to invert (model expects dark background) |
| Crop and centre | ✅ | Crops to the item bounding box before resizing |

---

## Requirements

```
tensorflow>=2.10
streamlit>=1.20
numpy
pillow
scikit-learn
matplotlib
pandas
```

Full pinned versions in [`requirements.txt`](requirements.txt).

---

## Notes

- The baseline model `fashion_cnn.keras` is **never overwritten**. All candidate models are saved under `models/versions/`.
- Model metadata and the active model pointer are stored in `models/` as plain JSON — easy to inspect or edit manually.
- The `Shirt` class is inherently difficult (visually similar to T-shirt and Coat) and has lower confidence than other classes — this is consistent with the original Fashion-MNIST benchmark results.
