# Fashion Item Classifier - local demo

## Folder layout
```
fashion_app/
  app.py
  preprocess.py
  requirements.txt
  fashion_cnn.keras     <- downloaded from Colab (see notebook, Section 12)
  samples/              <- unzip samples.zip here (optional, for the "Pick a sample" mode)
```

## Run (Windows / Mac / Linux)
1. Install Python 3.10, 3.11 or 3.12.
2. Open a terminal in the `fashion_app` folder, then:
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
3. Your browser opens at http://localhost:8501

The first start is slow (TensorFlow loads). After that each prediction takes a fraction of a second.


## Custom Dataset Retraining & Non-Clothing Rejection

### Prepare Your Dataset

Create a folder with one sub-folder per category. The system supports either:
- **10 clothing categories** (standard fashion classification), or
- **10 clothing categories + 1 optional non-clothing category** (`Other / Not Clothing`) to detect and reject unrelated images (wallpapers, landscapes, animals, vehicles, etc.).

```
D:\FashionDataset\
  T-shirt_top\        ← maps to "T-shirt/top"
  Trouser\
  Pullover\
  Dress\
  Coat\
  Sandal\
  Shirt\
  Sneaker\
  Bag\
  Ankle_boot\         ← maps to "Ankle boot"
  Other\              ← optional 11th class: maps to "Other / Not Clothing"
```

Folder naming is flexible (case-insensitive and separator-insensitive). Accepted folder names for the non-clothing category include:
`Other`, `Not_Clothing`, `Non_Clothing`, `Other_Not_Clothing`, `Unrelated`, `Random`, `Negative`.

Each sub-folder should contain 10–50+ JPG/PNG/WEBP images. For the `Other` category, include a diverse set of unrelated images (wallpapers, logos, landscapes, food, vehicles, animals, household objects).

### Workflow Steps

1. Launch the app: `streamlit run app.py`
2. In the sidebar, select **Model Training**.
3. Enter your dataset path (e.g. `D:\FashionDataset`) and click **Validate Dataset**.
   - If an `Other` folder is detected, the system automatically sets up an **11-class model** with non-clothing rejection.
4. Configure epochs / batch size / learning rate, then click **Start Training**. Watch the live epoch progress bar and validation metrics.
5. After training, click **Evaluate Both Models & Save Candidate**.
6. Open **Model Comparison** to review:
   - Shared clothing test subset accuracy, precision, recall, and macro F1.
   - Non-clothing rejection rate (percentage of unrelated images correctly rejected).
   - Baseline false clothing classification rate (unrelated images mistakenly forced into clothing categories).
   - Side-by-side confusion matrices and training curves.
7. Open **Model Management** and click **Set as Active** to activate the new model.
8. To roll back at any time, open **Model Management**, select any previous version (or `Baseline (Fashion-MNIST)`), and click **Set as Active**.

### Prediction & Rejection Logic

When an 11-class model is active:
- Images predicted as **Other / Not Clothing** trigger a rejection banner and are excluded from clothing categorization.
- Predictions use both **Confidence Threshold** and **Uncertainty Margin** (difference between top and second class probability) to flag ambiguous inputs for human review.

### Storage

- Trained models: `models/versions/` (e.g., `candidate_20261030_165000.keras`)
- Metadata: `models/metadata/`
- Registry: `models/model_registry.json`
- Active model: `models/active_model.json`

> **Note:** Baseline model `fashion_cnn.keras` is never overwritten. All historical model versions are preserved for safe rollback.
