"""
Fashion Item Classifier — extended with model training & comparison
Run:  streamlit run app.py
"""
import glob
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import model_manager
import dataset_utils
import training as trainer
import evaluation as evaluator
from preprocess import CLASS_NAMES, preprocess
from dataset_utils import (
    CLOTHING_CLASSES,
    OTHER_CLASS,
    CANONICAL_11_CLASSES,
    get_class_names_for_dataset,
)

# ─── Initialise registry on every cold start ────────────────────────────────
model_manager.initialise_registry()

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fashion Item Classifier", page_icon="👕", layout="wide")

MODEL_PATH = "fashion_cnn.keras"
SAMPLE_DIR = "samples"
UPPER_BODY = {"T-shirt/top", "Shirt", "Pullover", "Coat"}


# ─── Model loading (cached per model_id so cache busts on switch) ────────────
@st.cache_resource(show_spinner="Loading model…")
def _load_model(model_id: str):
    return model_manager.load_model_by_id(model_id)


def _get_model_class_names(model_id: str) -> list:
    """Returns class names for a model based on registry or model architecture."""
    rec = model_manager.get_model_record(model_id)
    if rec and rec.get("class_names"):
        return list(rec["class_names"])
    try:
        m = _load_model(model_id)
        if m.output_shape[-1] == 11:
            return list(CANONICAL_11_CLASSES)
    except Exception:
        pass
    return list(CLOTHING_CLASSES)


# ─── Sidebar navigation ──────────────────────────────────────────────────────
PAGES = ["📷 Image Prediction", "🏋️ Model Training", "📊 Model Comparison", "📁 Model Management"]
page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")

# ─── Sidebar settings (always visible) ──────────────────────────────────────
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Auto-tag confidence threshold", 0.50, 0.99, 0.90, 0.01,
    help="Below this or when uncertainty margin is low, the item is sent to human review.",
)
min_margin = st.sidebar.slider(
    "Uncertainty margin threshold", 0.05, 0.50, 0.20, 0.05,
    help="Minimum difference between top and second class probability to consider prediction distinct.",
)
rejection_sensitivity = st.sidebar.slider(
    "Non-clothing rejection sensitivity", 0.05, 0.50, 0.20, 0.05,
    help="If non-clothing probability reaches this threshold, the image is rejected as non-clothing, even if a clothing category had a slight plurality.",
)
invert_choice = st.sidebar.selectbox(
    "Invert colours?", ["Auto", "Yes", "No"],
    help="The model expects a light item on a DARK background.",
)
crop = st.sidebar.checkbox("Crop and centre the item", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tip:** works best on a single item on a plain background.")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _predict_single(model, img: Image.Image) -> tuple:
    """Returns (probs, small, border_std) for one PIL image using the existing pipeline."""
    arr = np.asarray(ImageOps.grayscale(img.convert("RGB")), dtype=np.uint8)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    border_std = float(border.std())

    small = preprocess(img, invert_choice.lower(), crop)          # (28, 28) uint8
    x = (small.astype("float32") / 255.0)[None, ..., None]        # (1, 28, 28, 1)
    probs = model.predict(x, verbose=0)[0]
    return probs, small, border_std


def _render_prediction_col(col, probs: np.ndarray, model_display_name: str, class_names: list = None, border_std: float = 0.0):
    """Renders prediction result inside a Streamlit column with non-clothing rejection and domain gating."""
    if class_names is None:
        class_names = CLOTHING_CLASSES

    order = np.argsort(-probs)
    top, second = order[0], order[1]
    conf = float(probs[top])
    margin = float(probs[top] - probs[second])
    top_name = class_names[top]

    other_idx = class_names.index(OTHER_CLASS) if OTHER_CLASS in class_names else -1
    other_prob = float(probs[other_idx]) if other_idx >= 0 else 0.0

    # ── Non-clothing rejection check (Direct or Gate-Triggered) ───────────────
    is_direct_rejection = (top_name == OTHER_CLASS)
    is_sensitivity_rejection = (other_idx >= 0 and other_prob >= rejection_sensitivity and top_name != OTHER_CLASS)
    is_domain_scene_rejection = (other_idx >= 0 and border_std > 20.0 and other_prob >= 0.10 and conf < 0.85)
    is_ambiguity_rejection = (other_idx >= 0 and top_name != OTHER_CLASS and conf < 0.70 and (conf - other_prob) < 0.25 and other_prob >= 0.08)

    if is_direct_rejection or is_sensitivity_rejection or is_domain_scene_rejection or is_ambiguity_rejection:
        col.error("🚫 **Non-Clothing Image Detected & Rejected**")
        if is_direct_rejection:
            col.markdown(
                f"The image was classified as **{OTHER_CLASS}** with **{conf:.1%}** confidence "
                f"(margin: {margin:.1%}).\n\n"
                "This image does not belong to any of the 10 supported clothing categories and has been excluded."
            )
            col.metric("Rejection Confidence", f"{conf:.1%}", help=f"Model: {model_display_name}")
        elif is_sensitivity_rejection:
            col.markdown(
                f"**Domain Gating Rejection:** Although closest clothing silhouette was **{top_name}** ({conf:.1%}), "
                f"the non-clothing probability reached **{other_prob:.1%}** (exceeding your {rejection_sensitivity:.0%} sensitivity threshold).\n\n"
                "Because clothing confidence was not decisive, the system rejected this as an unrelated scene / non-clothing image."
            )
            col.metric("Non-Clothing Probability", f"{other_prob:.1%}", delta=f"Threshold: {rejection_sensitivity:.0%}")
        elif is_domain_scene_rejection:
            col.markdown(
                f"**Scene Background Rejection:** Continuous scene background detected (border variance: {border_std:.1f}) "
                f"with non-clothing probability **{other_prob:.1%}** and clothing confidence {conf:.1%}.\n\n"
                "Natural landscapes, wallpapers, or unisolated room photos are rejected to avoid false clothing tags."
            )
            col.metric("Scene Border Variance", f"{border_std:.1f}", delta=f"Non-Clothing: {other_prob:.1%}")
        else:
            col.markdown(
                f"**Ambiguous Prediction Rejection:** The model had low confidence ({conf:.1%} for {top_name}) "
                f"and could not decisively distinguish this image from non-clothing ({other_prob:.1%} Other, gap: {conf - other_prob:.1%}).\n\n"
                "Rather than hazarding a false clothing prediction, the image is conservatively rejected as non-clothing."
            )
            col.metric("Uncertainty Gap", f"{conf - other_prob:.1%}", delta=f"Top Clothing: {conf:.1%}")

        col.markdown("**Top 3 predictions**")
        for i in order[:3]:
            col.write(f"{class_names[i]}: {probs[i]:.1%}")

        col.markdown("**All class probabilities**")
        col.bar_chart(pd.DataFrame({"probability": probs}, index=class_names))
        return OTHER_CLASS

    # ── Clothing prediction ──────────────────────────────────────────────────
    col.subheader(f"Prediction: {top_name}")
    col.metric("Confidence", f"{conf:.1%}", delta=f"Margin: {margin:.1%}", help=f"Model: {model_display_name}")

    if border_std > 25.0:
        col.info(f"ℹ️ Background variance is elevated ({border_std:.1f}). Fashion CNNs perform best on plain/isolated backgrounds.")

    if conf >= threshold and margin >= min_margin:
        col.success(f"Auto-tagged as **{top_name}** (confidence ≥ {threshold:.0%}, margin ≥ {min_margin:.0%}).")
    elif conf < threshold:
        col.warning(f"Confidence is below {threshold:.0%}. **Send to human review.**")
    else:
        col.warning(f"Narrow margin ({margin:.1%} < {min_margin:.0%}). Prediction is ambiguous. **Send to human review.**")

    if (top_name in UPPER_BODY and class_names[second] in UPPER_BODY
            and probs[second] > 0.15):
        col.info(
            f"Close call between **{top_name}** and **{class_names[second]}**. "
            "These are the classes the model confuses most."
        )

    col.markdown("**Top 3 predictions**")
    for i in order[:3]:
        col.write(f"{class_names[i]}: {probs[i]:.1%}")

    col.markdown("**All class probabilities**")
    col.bar_chart(pd.DataFrame({"probability": probs}, index=class_names))
    return top_name


def _image_source_widget(key_prefix: str = ""):
    """Shared image-source widget. Returns a PIL Image or None."""
    source = st.radio(
        "Image source",
        ["Upload an image", "Pick a sample"],
        horizontal=True,
        key=f"{key_prefix}source",
    )
    img = None
    if source == "Upload an image":
        f = st.file_uploader(
            "Upload a clothing image",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"{key_prefix}uploader",
        )
        if f:
            img = Image.open(f)
    else:
        files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.png")))
        if not files:
            st.info("No 'samples' folder found. Run the export cell in the notebook to create it.")
        else:
            choice = st.selectbox(
                "Sample image",
                [os.path.basename(p) for p in files],
                key=f"{key_prefix}sample",
            )
            img = Image.open(os.path.join(SAMPLE_DIR, choice))
    return img


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Image Prediction
# ═════════════════════════════════════════════════════════════════════════════
if page == "📷 Image Prediction":
    st.title("👕 Fashion Item Classifier")
    st.caption("CNN trained on Fashion-MNIST  |  93.9% test accuracy  |  macro F1 0.939")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Put it in the same folder as app.py.")
        st.stop()

    active_id = model_manager.get_active_model_id()
    all_models = model_manager.list_models()
    candidates = [m for m in all_models if m["kind"] == "candidate"]

    if candidates:
        pred_mode = st.radio(
            "Prediction Mode",
            ["Active Model", "Side-by-Side Comparison"],
            horizontal=True,
            help="Choose 'Active Model' for standard inference, or 'Side-by-Side' to compare two models simultaneously.",
        )
    else:
        pred_mode = "Active Model"

    # ── Single Active Model Mode ──────────────────────────────────────────────
    if pred_mode == "Active Model":
        active_rec = model_manager.get_model_record(active_id)
        display_name = active_rec["display_name"] if active_rec else active_id
        class_names = _get_model_class_names(active_id)
        is_rej = (len(class_names) == 11)

        if is_rej:
            st.success(f"**Active model:** {display_name} 🛡️ *(11 classes — Non-clothing rejection enabled)*")
        else:
            st.info(f"**Active model:** {display_name} 👕 *(10 clothing classes)*")
            st.caption("ℹ️ Notice: The current model only has 10 clothing outputs. It cannot reject non-clothing images; unrelated photographs (wallpapers, animals, vehicles) will be forced into a clothing category. Switch to the 11-class model in 'Model Management' for non-clothing rejection.")

        model = _load_model(active_id)

        img = _image_source_widget("singleA_")

        if img is not None:
            probs, small, border_std = _predict_single(model, img)

            c1, c2 = st.columns(2)
            c1.image(img.convert("RGB"), caption="Your image", use_container_width=True)
            c2.image(
                Image.fromarray(small).resize((224, 224), Image.NEAREST),
                caption="What the model sees (28×28)",
                use_container_width=True,
            )

            _render_prediction_col(st, probs, display_name, class_names, border_std=border_std)

    # ── Side-by-Side Comparison Mode ──────────────────────────────────────────
    else:
        st.subheader("Side-by-Side Model Comparison")
        all_model_ids    = [m["id"]           for m in all_models]
        all_model_names  = [m["display_name"] for m in all_models]

        default_a = next((i for i, m in enumerate(all_models) if m["kind"] == "baseline"), 0)
        default_b = 0 if default_a != 0 else (1 if len(all_models) > 1 else 0)

        col_sel_a, col_sel_b = st.columns(2)
        sel_a = col_sel_a.selectbox("Model A", all_model_ids,
                                    index=default_a,
                                    format_func=lambda mid: all_model_names[all_model_ids.index(mid)],
                                    key="pred_sel_a")
        sel_b = col_sel_b.selectbox("Model B", all_model_ids,
                                    index=default_b,
                                    format_func=lambda mid: all_model_names[all_model_ids.index(mid)],
                                    key="pred_sel_b")

        model_a = _load_model(sel_a)
        model_b = _load_model(sel_b)
        name_a  = all_model_names[all_model_ids.index(sel_a)]
        name_b  = all_model_names[all_model_ids.index(sel_b)]
        class_names_a = _get_model_class_names(sel_a)
        class_names_b = _get_model_class_names(sel_b)

        img = _image_source_widget("sideB_")

        if img is not None:
            # Preprocess ONCE — same tensor for both
            probs_a, small, border_std = _predict_single(model_a, img)
            # Reuse the same preprocessed tensor for model B (re-predict from same small)
            x = (small.astype("float32") / 255.0)[None, ..., None]
            probs_b = model_b.predict(x, verbose=0)[0]

            def _is_effective_rejection(p, cnames):
                if OTHER_CLASS not in cnames:
                    return False, cnames[int(np.argmax(p))]
                oidx = cnames.index(OTHER_CLASS)
                tidx = int(np.argmax(p))
                tname = cnames[tidx]
                oprob = float(p[oidx])
                cconf = float(p[tidx])
                if tname == OTHER_CLASS:
                    return True, OTHER_CLASS
                if oprob >= rejection_sensitivity:
                    return True, OTHER_CLASS
                if border_std > 20.0 and oprob >= 0.10 and cconf < 0.85:
                    return True, OTHER_CLASS
                if cconf < 0.70 and (cconf - oprob) < 0.25 and oprob >= 0.08:
                    return True, OTHER_CLASS
                return False, tname

            rej_a, eff_a = _is_effective_rejection(probs_a, class_names_a)
            rej_b, eff_b = _is_effective_rejection(probs_b, class_names_b)

            # Agreement & rejection contrast banner
            if eff_a == eff_b:
                if rej_a:
                    st.error(f"🛡️ Both models agree: Non-clothing image rejected (**{OTHER_CLASS}**)")
                else:
                    st.success(f"✅ Both models agree: **{eff_a}**")
            elif rej_a and not rej_b:
                st.warning(
                    f"🛡️ **Non-Clothing Rejection in Action:** Model A ({name_a}) recognized and rejected this image as **{OTHER_CLASS}** (non-clothing signal: {probs_a[class_names_a.index(OTHER_CLASS)]:.1%}), "
                    f"whereas Model B ({name_b}) falsely forced it into clothing category **{eff_b}** ({probs_b.max():.1%})."
                )
            elif rej_b and not rej_a:
                st.warning(
                    f"🛡️ **Non-Clothing Rejection in Action:** Model B ({name_b}) recognized and rejected this image as **{OTHER_CLASS}** (non-clothing signal: {probs_b[class_names_b.index(OTHER_CLASS)]:.1%}), "
                    f"whereas Model A ({name_a}) falsely forced it into clothing category **{eff_a}** ({probs_a.max():.1%})."
                )
            else:
                st.warning(f"⚠️ Models disagree — Model A says **{eff_a}**, Model B says **{eff_b}**")

            img_col, _ = st.columns([1, 3])
            img_col.image(img.convert("RGB"), caption="Input image", use_container_width=True)

            col_a, col_b = st.columns(2)
            col_a.markdown(f"### {name_a}")
            col_b.markdown(f"### {name_b}")
            _render_prediction_col(col_a, probs_a, name_a, class_names_a, border_std=border_std)
            _render_prediction_col(col_b, probs_b, name_b, class_names_b, border_std=border_std)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Model Training
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏋️ Model Training":
    st.header("🏋️ Model Training")
    st.info("Train a new candidate model on your own labelled clothing images.")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    st.subheader("1. Dataset")
    folder_path = st.text_input(
        "Dataset folder path",
        placeholder=r"D:\FashionDataset",
        help="Local path on this computer. Each sub-folder = one clothing category.",
    )

    col1, col2 = st.columns([1, 4])
    validate_btn = col1.button("🔍 Validate Dataset")

    if validate_btn and folder_path:
        with st.spinner("Scanning dataset…"):
            scan = dataset_utils.scan_dataset(folder_path)

        if scan["errors"]:
            for e in scan["errors"]:
                st.error(e)
        else:
            st.success(
                f"✅ Dataset valid — {scan['total_valid']} images "
                f"in {len(scan['matched_classes'])} categories"
            )
            if scan.get("has_other"):
                st.success(
                    f"🛡️ **Non-clothing category detected ('{OTHER_CLASS}')!** "
                    "The new candidate model will be trained with 11 output classes and will support non-clothing rejection."
                )
            else:
                st.info("ℹ️ 10 clothing categories detected. Candidate model will have 10 output classes.")

            rows = [
                {"Class": c, "Images": len(imgs), "Role": "Negative Example (Reject)" if c == OTHER_CLASS else "Clothing Category"}
                for c, imgs in scan["matched_classes"].items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            train_items, val_items, test_items = dataset_utils.split_dataset(
                scan["matched_classes"]
            )
            dataset_classes = dataset_utils.get_class_names_for_dataset(scan["matched_classes"])
            st.info(
                f"Split preview ({len(dataset_classes)} classes): **{len(train_items)}** train / "
                f"**{len(val_items)}** val / **{len(test_items)}** test"
            )

            for w in scan["warnings"]:
                st.warning(w)
            if scan["corrupt_files"]:
                st.warning(f"Skipped {len(scan['corrupt_files'])} corrupt files.")
            if scan["unmatched_folders"]:
                st.warning(f"Unknown folders (ignored): {scan['unmatched_folders']}")

            st.session_state["scan_result"]     = scan
            st.session_state["dataset_path"]    = folder_path
            st.session_state["dataset_classes"] = dataset_classes

    # ── 2. Training config ────────────────────────────────────────────────────
    if "scan_result" in st.session_state:
        st.subheader("2. Training Configuration")
        dataset_classes = st.session_state.get("dataset_classes", CLOTHING_CLASSES)
        c1, c2, c3 = st.columns(3)
        epochs     = c1.number_input("Epochs",         5,   100,  25)
        batch_size = c2.number_input("Batch size",     8,   128,  32)
        lr         = c3.number_input("Learning rate",  1e-6, 1e-2, 1e-4, format="%.6f")
        strategy   = st.radio(
            "Training strategy",
            ["Fine-tune baseline", "Train from scratch"],
            horizontal=True,
            help=(
                "Fine-tune reuses feature layers from baseline (recommended). "
                "For 11-class models, feature layers are transferred and a fresh 11-neuron head is initialized."
            ),
        )

        # ── 3. Train ──────────────────────────────────────────────────────────
        st.subheader("3. Train")
        train_btn = st.button(
            "🚀 Start Training",
            type="primary",
            disabled=st.session_state.get("training_running", False),
        )

        if train_btn:
            st.session_state["training_running"] = True
            scan = st.session_state["scan_result"]
            dataset_classes = st.session_state.get("dataset_classes", CLOTHING_CLASSES)
            num_classes = len(dataset_classes)

            with st.spinner("Loading and preprocessing images…"):
                t_items, v_items, te_items = dataset_utils.split_dataset(scan["matched_classes"])
                X_train, y_train = dataset_utils.build_arrays(t_items)
                X_val,   y_val   = dataset_utils.build_arrays(v_items)
                X_test,  y_test  = dataset_utils.build_arrays(te_items)

            st.write(
                f"Arrays ready — train: `{X_train.shape}`, "
                f"val: `{X_val.shape}`, test: `{X_test.shape}` | Classes: `{num_classes}`"
            )

            progress_bar = st.progress(0, text="Starting training…")
            epoch_status = st.empty()

            history_log = []

            def on_epoch_end(epoch_num, total, logs):
                pct = epoch_num / total
                progress_bar.progress(pct, text=f"Epoch {epoch_num}/{total}")
                epoch_status.markdown(
                    f"**train_acc:** {logs.get('accuracy', 0):.3f} &nbsp;|&nbsp;"
                    f"**val_acc:** {logs.get('val_accuracy', 0):.3f} &nbsp;|&nbsp;"
                    f"**train_loss:** {logs.get('loss', 0):.4f} &nbsp;|&nbsp;"
                    f"**val_loss:** {logs.get('val_loss', 0):.4f}"
                )
                history_log.append(logs)

            strat_key = "finetune" if "Fine-tune" in strategy else "scratch"

            try:
                with st.spinner("Training in progress… (do not close this tab)"):
                    model, hist, duration = trainer.train_candidate(
                        X_train, y_train, X_val, y_val,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        lr=float(lr),
                        strategy=strat_key,
                        num_classes=num_classes,
                        progress_callback=on_epoch_end,
                    )

                progress_bar.progress(1.0, text="✅ Training complete!")
                st.success(
                    f"Training finished in {duration:.1f}s — "
                    f"{len(hist['accuracy'])} epochs completed"
                )

                st.session_state["trained_model"]   = model
                st.session_state["train_history"]   = hist
                st.session_state["X_test"]          = X_test
                st.session_state["y_test"]          = y_test
                st.session_state["train_duration"]  = duration
                st.session_state["split_counts"]    = {
                    "train": len(t_items), "val": len(v_items), "test": len(te_items),
                }
                st.session_state["class_dist"] = {
                    c: len(imgs) for c, imgs in scan["matched_classes"].items()
                }
                st.session_state["train_config"] = {
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "lr": float(lr),
                    "strategy": strat_key,
                    "num_classes": num_classes,
                }

            except Exception as ex:
                st.error(f"Training failed: {ex}")
            finally:
                st.session_state["training_running"] = False

        # ── 4. Evaluate & Save ────────────────────────────────────────────────
        if "trained_model" in st.session_state:
            st.subheader("4. Evaluate & Save")
            eval_btn = st.button("📊 Evaluate Both Models & Save Candidate")

            if eval_btn:
                X_test = st.session_state["X_test"]
                y_test = st.session_state["y_test"]
                model  = st.session_state["trained_model"]
                dataset_classes = st.session_state.get("dataset_classes", CLOTHING_CLASSES)
                num_classes = len(dataset_classes)
                supports_rejection = (num_classes == 11 and OTHER_CLASS in dataset_classes)

                with st.spinner("Evaluating models on unseen test split…"):
                    baseline_model = model_manager.load_model_by_id("baseline_v0")
                    model_id = f"candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    eval_res = evaluator.evaluate_dual_with_rejection(
                        baseline_model=baseline_model,
                        candidate_model=model,
                        X_test=X_test,
                        y_test=y_test,
                        class_names=dataset_classes,
                        model_id=model_id,
                    )

                baseline_eval = eval_res["baseline_clothing_eval"]
                cand_eval     = eval_res["candidate_clothing_eval"]
                cand_full     = eval_res["candidate_full_eval"]
                rejection_stats = eval_res["rejection_stats"]

                saved_path = trainer.save_candidate(model, model_id)
                size_mb    = saved_path.stat().st_size / 1e6

                try:
                    num_params = model.count_params()
                except Exception:
                    num_params = None

                rejection_rate = rejection_stats["candidate_rejection_rate"] if rejection_stats else None

                record = {
                    "id":                 model_id,
                    "display_name":       f"Candidate ({'11-class Rejection' if supports_rejection else '10-class'} — {datetime.now().strftime('%Y-%m-%d %H:%M')})",
                    "kind":               "candidate",
                    "keras_path":         str(saved_path),
                    "created_at":         datetime.now().isoformat(),
                    "dataset_path":       st.session_state.get("dataset_path"),
                    "train_count":        st.session_state["split_counts"]["train"],
                    "val_count":          st.session_state["split_counts"]["val"],
                    "test_count":         st.session_state["split_counts"]["test"],
                    "class_dist":         st.session_state.get("class_dist"),
                    "num_classes":        num_classes,
                    "supports_rejection": supports_rejection,
                    "class_names":        dataset_classes,
                    "accuracy":           round(cand_full.accuracy,    4),
                    "macro_f1":           round(cand_full.macro_f1,    4),
                    "clothing_accuracy":  round(cand_eval.accuracy,    4),
                    "clothing_macro_f1":  round(cand_eval.macro_f1,    4),
                    "rejection_accuracy": round(rejection_rate, 4) if rejection_rate is not None else None,
                    "weighted_f1":        round(cand_full.weighted_f1, 4),
                    "precision":          round(cand_full.precision,   4),
                    "recall":             round(cand_full.recall,      4),
                    "train_duration_s":   round(st.session_state["train_duration"], 1),
                    "model_size_mb":      round(size_mb, 2),
                    "num_params":         num_params,
                    "train_config":       st.session_state.get("train_config"),
                    "notes":              "Includes Other/Not Clothing rejection" if supports_rejection else "10 clothing classes",
                }
                model_manager.register_candidate(record)

                st.session_state["eval_res"]         = eval_res
                st.session_state["baseline_eval"]    = baseline_eval
                st.session_state["cand_eval"]        = cand_eval
                st.session_state["cand_full"]        = cand_full
                st.session_state["rejection_stats"]  = rejection_stats
                st.session_state["last_model_id"]    = model_id

                st.success(f"✅ Candidate saved as `{model_id}` ({size_mb:.1f} MB)")
                st.info("Open **📊 Model Comparison** in the sidebar to review detailed performance.")

                m1, m2 = st.columns(2)
                m1.metric("Baseline clothing test accuracy", f"{baseline_eval.accuracy:.1%}")
                m2.metric(
                    "Candidate clothing test accuracy",
                    f"{cand_eval.accuracy:.1%}",
                    delta=f"{(cand_eval.accuracy - baseline_eval.accuracy):+.1%}",
                )
                if rejection_stats:
                    st.metric("Candidate non-clothing rejection accuracy", f"{rejection_stats['candidate_rejection_rate']:.1%}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.header("📊 Model Comparison")

    if "baseline_eval" not in st.session_state or "cand_eval" not in st.session_state:
        st.info("Run **Model Training** and click **'Evaluate & Save'** first.")
    else:
        baseline_eval: evaluator.EvalResult = st.session_state["baseline_eval"]
        cand_eval:     evaluator.EvalResult = st.session_state["cand_eval"]
        cand_full:     evaluator.EvalResult = st.session_state.get("cand_full", cand_eval)
        rejection_stats = st.session_state.get("rejection_stats")
        hist = st.session_state.get("train_history", {})

        comparison = evaluator.compare_models(baseline_eval, cand_eval)

        # ── Non-clothing rejection highlight card ────────────────────────────
        if rejection_stats:
            st.subheader("🛡️ Non-Clothing Rejection Analysis")
            st.markdown(
                "Evaluation on unseen **Other / Not Clothing** test images. "
                "Compares how the 11-class candidate and the 10-class baseline handle non-clothing inputs."
            )
            r_col1, r_col2 = st.columns(2)
            r_col1.metric(
                "Candidate Rejection Rate",
                f"{rejection_stats['candidate_rejection_rate']:.1%}",
                help=f"{rejection_stats['candidate_rejected_count']} of {rejection_stats['total_other_test']} non-clothing test images correctly rejected.",
            )
            r_col2.metric(
                "Baseline False Clothing Assignment",
                f"{rejection_stats['baseline_false_clothing_rate']:.1%}",
                delta="100% false clothing predictions (no rejection)",
                delta_color="inverse",
                help="Baseline has only 10 output classes, so 100% of non-clothing images are falsely forced into clothing classes.",
            )

            if rejection_stats.get("baseline_false_classes"):
                st.markdown("**False Clothing Categories Assigned by Baseline on Non-Clothing Images:**")
                false_df = pd.DataFrame(
                    [{"False Category": k, "Count": v} for k, v in rejection_stats["baseline_false_classes"].items()]
                )
                st.dataframe(false_df, use_container_width=True)

            st.markdown("---")

        # ── Clothing Metrics table ───────────────────────────────────────────
        st.subheader("Clothing Test Subset Comparison (Shared 10 Classes)")
        st.caption("Fair comparison on identical unseen clothing test images.")
        table_rows = evaluator.format_comparison_table(baseline_eval, cand_eval)
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # ── Metric cards ─────────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            "Clothing Accuracy",
            f"{cand_eval.accuracy:.4f}",
            delta=f"{comparison['accuracy_delta']:+.4f}",
        )
        mc2.metric(
            "Clothing Macro F1",
            f"{cand_eval.macro_f1:.4f}",
            delta=f"{comparison['macro_f1_delta']:+.4f}",
        )
        mc3.metric(
            "Clothing Weighted F1",
            f"{cand_eval.weighted_f1:.4f}",
            delta=f"{comparison['weighted_f1_delta']:+.4f}",
        )
        last_id = st.session_state.get("last_model_id", "")
        cand_rec = model_manager.get_model_record(last_id)
        cand_size = cand_rec["model_size_mb"] if cand_rec else None
        mc4.metric("Candidate Size (MB)", f"{cand_size:.1f}" if cand_size else "—")

        # ── Bar chart: accuracy & macro F1 ───────────────────────────────────
        st.subheader("Accuracy & Macro F1 Comparison")
        chart_df = pd.DataFrame(
            {
                "Baseline":  [baseline_eval.accuracy, baseline_eval.macro_f1],
                "Candidate": [cand_eval.accuracy,     cand_eval.macro_f1],
            },
            index=["Clothing Accuracy", "Clothing Macro F1"],
        )
        st.bar_chart(chart_df)

        # ── Training curves (candidate only) ─────────────────────────────────
        if hist.get("accuracy"):
            st.subheader("Training Curves (Candidate)")
            epochs_range = list(range(1, len(hist["accuracy"]) + 1))

            acc_df = pd.DataFrame(
                {
                    "train_accuracy": hist["accuracy"],
                    "val_accuracy":   hist["val_accuracy"],
                },
                index=epochs_range,
            )
            loss_df = pd.DataFrame(
                {
                    "train_loss": hist["loss"],
                    "val_loss":   hist["val_loss"],
                },
                index=epochs_range,
            )
            st.line_chart(acc_df)
            st.line_chart(loss_df)

        # ── Confusion matrices ────────────────────────────────────────────────
        st.subheader("Confusion Matrices")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Baseline confusion matrix (10 classes on clothing)
        ax_base = axes[0]
        cm_base = baseline_eval.conf_matrix
        ax_base.imshow(cm_base, cmap="Blues")
        b_names = baseline_eval.class_names
        ax_base.set_xticks(range(len(b_names)))
        ax_base.set_yticks(range(len(b_names)))
        ax_base.set_xticklabels(b_names, rotation=45, ha="right", fontsize=8)
        ax_base.set_yticklabels(b_names, fontsize=8)
        ax_base.set_title(f"Baseline Confusion Matrix ({len(b_names)} Classes)")
        ax_base.set_xlabel("Predicted")
        ax_base.set_ylabel("True")
        for i in range(len(b_names)):
            for j in range(len(b_names)):
                ax_base.text(
                    j, i, str(cm_base[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_base[i, j] > cm_base.max() / 2 else "black",
                )

        # Candidate confusion matrix (all classes, 10 or 11)
        ax_cand = axes[1]
        cm_cand = cand_full.conf_matrix
        c_names = cand_full.class_names
        ax_cand.imshow(cm_cand, cmap="Blues")
        ax_cand.set_xticks(range(len(c_names)))
        ax_cand.set_yticks(range(len(c_names)))
        ax_cand.set_xticklabels(c_names, rotation=45, ha="right", fontsize=8)
        ax_cand.set_yticklabels(c_names, fontsize=8)
        ax_cand.set_title(f"Candidate Confusion Matrix ({len(c_names)} Classes)")
        ax_cand.set_xlabel("Predicted")
        ax_cand.set_ylabel("True")
        for i in range(len(c_names)):
            for j in range(len(c_names)):
                ax_cand.text(
                    j, i, str(cm_cand[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_cand[i, j] > cm_cand.max() / 2 else "black",
                )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Recommendation ────────────────────────────────────────────────────
        st.subheader("Recommendation")
        st.markdown(comparison["recommendation"])
        if rejection_stats and rejection_stats["candidate_rejection_rate"] >= 0.70:
            st.success("🛡️ **Recommendation Note**: The candidate model demonstrates effective non-clothing rejection. Promoting it will protect against false classifications on non-clothing imagery.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Model Management
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📁 Model Management":
    st.header("📁 Model Management")

    models    = model_manager.list_models()
    active_id = model_manager.get_active_model_id()

    rows = [
        {
            "ID":        m["id"],
            "Name":      m["display_name"],
            "Kind":      m["kind"],
            "Classes":   m.get("num_classes", 10),
            "Rejection": "🛡️ Yes" if m.get("supports_rejection") else "No",
            "Accuracy":  m["accuracy"],
            "Macro F1":  m["macro_f1"],
            "Size MB":   m["model_size_mb"],
            "Created":   m["created_at"][:16] if m.get("created_at") else "",
            "Active":    "✅" if m["id"] == active_id else "",
        }
        for m in models
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Active model selector ─────────────────────────────────────────────────
    st.subheader("Switch Active Model")
    model_ids   = [m["id"]           for m in models]
    model_names = [f"{m['display_name']} [{m['id']}]" for m in models]
    current_idx = model_ids.index(active_id) if active_id in model_ids else 0

    selected_idx = st.radio(
        "Select model",
        range(len(model_ids)),
        format_func=lambda i: model_names[i],
        index=current_idx,
    )
    selected_id = model_ids[selected_idx]

    if selected_id != active_id:
        if st.button(
            f"⚡ Set '{model_names[selected_idx]}' as Active Model",
            type="primary",
        ):
            model_manager.set_active_model(selected_id)
            st.success(f"Active model changed to: {model_names[selected_idx]}")
            st.info("The Prediction page will now use this model. Reload the page if needed.")
            st.cache_resource.clear()   # force model reload on next visit
    else:
        st.info(f"'{model_names[selected_idx]}' is already the active model.")

    # ── Model details expander ────────────────────────────────────────────────
    selected_rec = model_manager.get_model_record(selected_id)
    if selected_rec:
        with st.expander("Model Details"):
            st.json(selected_rec)
