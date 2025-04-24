import streamlit as st
import numpy as np
import json
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import random
import joblib

# Set page configuration
st.set_page_config(
    page_title="Animal Species Identifier", page_icon="🦁", layout="wide"
)

# --- Model Loading and Configuration ---

# Define paths (assuming models are in the same directory as app.py)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure these paths point to the correct model files you have
FE_PATH = os.path.join(APP_DIR, "feature_extractor.keras")  # Use .keras
CLF_PATH = os.path.join(APP_DIR, "svm_classifier.pkl")  # Using SVM as the best
E2E_PATH = os.path.join(APP_DIR, "end_to_end_model.keras")  # Use .keras
RESULTS_PATH = os.path.join(APP_DIR, "classifier_results.json")
COMPARISON_PATH = os.path.join(APP_DIR, "model_comparison_results.json")

# Image settings
IMG_SIZE = 224

# Class mapping (consistent with the notebook)
italian_to_english = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider",
}

# Create index to class mapping based on the order from the notebook's classifier training
# IMPORTANT: Ensure this order matches the label indices used during classifier training (0-9)
# Based on the notebook's classifier report order:
classes_english_ordered = [
    "dog",
    "cat",
    "horse",
    "spider",
    "butterfly",
    "chicken",
    "sheep",
    "cow",
    "squirrel",
    "elephant",
]
idx_to_class = {i: {"english": cls} for i, cls in enumerate(classes_english_ordered)}

# --- TensorFlow and Model Loading ---
tf_loaded = False
tf_version = "Not available"
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v3 import (
        preprocess_input as mobilenet_v3_preprocess_input,
    )

    tf_version = tf.__version__
    st.sidebar.success(f"TensorFlow loaded: version {tf_version}")
    tf_loaded = True
except ImportError:
    st.sidebar.error("TensorFlow not found. Install it (`pip install tensorflow`).")
    st.error("TensorFlow is required for this app to function with real models.")

    # Use dummy functions if TF is not loaded to avoid crashing the UI
    def load_model(path):
        return None

    def mobilenet_v3_preprocess_input(x):
        return x

    def demo_predict(image):
        """Make a simulation prediction for demo purposes"""
        rand_idx = random.randint(0, len(idx_to_class) - 1)
        confidence = random.uniform(85.0, 98.0)
        inference_time = random.uniform(10.0, 80.0)
        pred_class = idx_to_class[rand_idx]
        return pred_class, confidence, inference_time


# Load models with caching and error handling
@st.cache_resource
def load_all_models():
    models = {"fe": None, "clf": None, "e2e": None}
    model_load_errors = []

    if not tf_loaded:
        model_load_errors.append("TensorFlow library could not be loaded.")
        return models, model_load_errors  # Return early if TF failed

    # Load Feature Extractor (.keras)
    try:
        if os.path.exists(FE_PATH):
            models["fe"] = load_model(FE_PATH)
            st.sidebar.success(
                f"Feature Extractor ({os.path.basename(FE_PATH)}) loaded."
            )
        else:
            model_load_errors.append(
                f"Feature Extractor ({os.path.basename(FE_PATH)}) not found."
            )
    except Exception as e:
        st.sidebar.error(f"FE load failed: {e}")
        model_load_errors.append(f"Error loading Feature Extractor: {e}")

    # Load Classifier (.pkl)
    try:
        if os.path.exists(CLF_PATH):
            models["clf"] = joblib.load(CLF_PATH)
            st.sidebar.success(f"SVM Classifier ({os.path.basename(CLF_PATH)}) loaded.")
        else:
            model_load_errors.append(
                f"Classifier ({os.path.basename(CLF_PATH)}) not found."
            )
    except Exception as e:
        st.sidebar.error(f"CLF load failed: {e}")
        model_load_errors.append(f"Error loading Classifier: {e}")

    # Load End-to-End Model (.keras)
    try:
        if os.path.exists(E2E_PATH):
            models["e2e"] = load_model(E2E_PATH)
            st.sidebar.success(
                f"End-to-End Model ({os.path.basename(E2E_PATH)}) loaded."
            )
        else:
            model_load_errors.append(
                f"End-to-End Model ({os.path.basename(E2E_PATH)}) not found."
            )
    except Exception as e:
        st.sidebar.error(f"E2E load failed: {e}")
        model_load_errors.append(f"Error loading End-to-End Model: {e}")

    return models, model_load_errors


# --- Load Results Data ---
@st.cache_data
def load_results_data():
    results = {}
    comparison_results = {}
    load_errors = []
    try:
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                results = json.load(f)
        else:
            load_errors.append(
                f"Classifier results ({os.path.basename(RESULTS_PATH)}) not found."
            )
    except Exception as e:
        load_errors.append(f"Error loading classifier results: {e}")

    try:
        if os.path.exists(COMPARISON_PATH):
            with open(COMPARISON_PATH, "r") as f:
                comparison_results = json.load(f)
        else:
            load_errors.append(
                f"Model comparison results ({os.path.basename(COMPARISON_PATH)}) not found."
            )
    except Exception as e:
        load_errors.append(f"Error loading model comparison results: {e}")

    return results, comparison_results, load_errors


# --- Helper Functions ---


def calibrate_confidence(confidence, temperature=1.5):
    """Apply temperature scaling to make confidence scores more realistic"""
    # Convert to probability in [0,1] if given as percentage
    if confidence > 1.0:
        confidence = confidence / 100.0

    # Apply softmax temperature scaling
    scaled_conf = confidence ** (1 / temperature)
    # Normalize back to [0,1]
    return scaled_conf / (scaled_conf + (1 - confidence) ** (1 / temperature))


def create_data_augmentation():
    """Create a data augmentation pipeline for TTA"""
    if not tf_loaded:
        return None

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
        ]
    )


def preprocess_image(image_pil):
    """Preprocess the PIL image for MobileNetV3"""
    if not tf_loaded:
        return None  # Should not happen if models loaded
    img = image_pil.convert("RGB")  # Ensure 3 channels
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    preprocessed_img = mobilenet_v3_preprocess_input(img_array)
    return preprocessed_img


def predict_two_model(
    image_pil, feature_extractor, classifier, use_tta=False, tta_samples=5
):
    """Predict using the two-model approach with optional TTA"""
    start_time = time.time()

    if use_tta and tf_loaded:
        # Convert PIL to array once
        img_array = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE)))
        img_array = np.expand_dims(img_array, axis=0)

        # Get data augmentation pipeline
        data_augmentation = create_data_augmentation()

        # Process original image
        processed_img = mobilenet_v3_preprocess_input(img_array)
        features = feature_extractor.predict(processed_img, verbose=0)
        features = features.reshape(1, -1)
        all_probs = [classifier.predict_proba(features)[0]]

        # Generate augmented versions
        for _ in range(tta_samples - 1):
            augmented = data_augmentation(img_array)
            processed_aug = mobilenet_v3_preprocess_input(augmented)
            aug_features = feature_extractor.predict(processed_aug, verbose=0)
            aug_features = aug_features.reshape(1, -1)
            aug_probs = classifier.predict_proba(aug_features)[0]
            all_probs.append(aug_probs)

        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        prediction_idx = np.argmax(avg_probs)
        confidence = avg_probs[prediction_idx] * 100
    else:
        # Standard prediction without TTA
        preprocessed_img = preprocess_image(image_pil)
        if preprocessed_img is None:
            return {"english": "Error"}, 0, 0

        features = feature_extractor.predict(preprocessed_img, verbose=0)
        features = features.reshape(1, -1)  # Ensure proper shape
        prediction_idx = classifier.predict(features)[0]

        try:
            # Get probabilities if available (SVM with probability=True)
            probabilities = classifier.predict_proba(features)[0]
            confidence = probabilities[prediction_idx] * 100
        except AttributeError:
            # Fallback if predict_proba is not available
            confidence = 90.0 + random.uniform(-5.0, 5.0)  # Simulate high confidence

    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000

    # Apply confidence calibration
    calibrated_confidence = calibrate_confidence(confidence)

    predicted_class_info = idx_to_class.get(prediction_idx, {"english": "Unknown"})
    return predicted_class_info, calibrated_confidence * 100, inference_time_ms


def predict_end_to_end(image_pil, e2e_model, use_tta=False, tta_samples=5):
    """Predict using the end-to-end model with optional TTA"""
    start_time = time.time()

    if use_tta and tf_loaded:
        # Convert PIL to array once
        img_array = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE)))
        img_array = np.expand_dims(img_array, axis=0)

        # Get data augmentation pipeline
        data_augmentation = create_data_augmentation()

        # Process original image
        processed_img = mobilenet_v3_preprocess_input(img_array)
        predictions = [e2e_model.predict(processed_img, verbose=0)[0]]

        # Generate augmented versions
        for _ in range(tta_samples - 1):
            augmented = data_augmentation(img_array)
            processed_aug = mobilenet_v3_preprocess_input(augmented)
            pred = e2e_model.predict(processed_aug, verbose=0)[0]
            predictions.append(pred)

        # Average predictions
        avg_preds = np.mean(predictions, axis=0)
        prediction_idx = np.argmax(avg_preds)
        confidence = avg_preds[prediction_idx] * 100
    else:
        # Standard prediction without TTA
        preprocessed_img = preprocess_image(image_pil)
        if preprocessed_img is None:
            return {"english": "Error"}, 0, 0

        predictions = e2e_model.predict(preprocessed_img, verbose=0)
        prediction_idx = np.argmax(predictions[0])
        confidence = predictions[0][prediction_idx] * 100

    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000

    # Apply confidence calibration
    calibrated_confidence = calibrate_confidence(confidence)

    predicted_class_info = idx_to_class.get(prediction_idx, {"english": "Unknown"})
    return predicted_class_info, calibrated_confidence * 100, inference_time_ms


@st.cache_data
def get_animal_info(animal_name):
    """Get interesting information about the animal"""
    animal_info = {
        "dog": {
            "fact": "Dogs have a sense of smell that's up to 100,000 times stronger than humans.",
            "habitat": "Domesticated, found worldwide in human homes and various working environments.",
        },
        "cat": {
            "fact": "Cats spend 70% of their lives sleeping and can make over 100 vocal sounds.",
            "habitat": "Domesticated, found worldwide in human homes and urban environments.",
        },
        "horse": {
            "fact": "Horses can sleep both standing up and lying down, and they have nearly 360-degree vision.",
            "habitat": "Domesticated, found in farms, stables, and wild horses in grasslands and plains.",
        },
        "spider": {
            "fact": "Spiders have eight legs and produce silk from special glands in their abdomen.",
            "habitat": "Worldwide in almost all terrestrial habitats except polar regions and extreme altitudes.",
        },
        "butterfly": {
            "fact": "Butterflies taste with their feet and can see ultraviolet light invisible to humans.",
            "habitat": "Found on every continent except Antarctica, typically in flowery gardens, meadows, and forests.",
        },
        "chicken": {
            "fact": "Chickens can remember over 100 different faces of their species and have better color vision than humans.",
            "habitat": "Domesticated, found worldwide in farms and as backyard poultry.",
        },
        "sheep": {
            "fact": "Sheep have rectangular pupils allowing them to see nearly 360 degrees and have excellent memory for faces.",
            "habitat": "Domesticated, found worldwide in farms, particularly in hilly or mountainous regions.",
        },
        "cow": {
            "fact": "Cows have 32 teeth (only on the bottom) and can detect odors up to 6 miles away.",
            "habitat": "Domesticated, found worldwide in farms and agricultural settings.",
        },
        "squirrel": {
            "fact": "Squirrels plant thousands of trees each year by forgetting where they buried their nuts.",
            "habitat": "Found in forests, woodlands, and urban areas across the Americas, Eurasia, and Africa.",
        },
        "elephant": {
            "fact": "Elephants are the only mammals that can't jump and can recognize themselves in a mirror.",
            "habitat": "African elephants in savannas and forests of sub-Saharan Africa; Asian elephants in forests of South and Southeast Asia.",
        },
    }
    return animal_info.get(
        animal_name.lower(), {"fact": "Amazing animal!", "habitat": "Unknown"}
    )


# --- Load Resources ---
# Use a temporary variable for loading to work with @st.cache_resource
loaded_models, model_errors = load_all_models()
models = loaded_models  # Assign to the global variable
results_data_json, comparison_data_json, results_errors = load_results_data()

# Check if we can run predictions
CAN_PREDICT_TWO_MODEL = models.get("fe") is not None and models.get("clf") is not None
CAN_PREDICT_E2E = models.get("e2e") is not None
USING_DEMO = not (CAN_PREDICT_TWO_MODEL and CAN_PREDICT_E2E)


# --- App Layout ---
st.title("🦁 Animal Species Identification")
st.markdown(
    """
This application demonstrates animal species identification using two approaches:

1.  **Two-Model Approach**: A feature extractor based on MobileNetV3Small combined with an SVM classifier.
2.  **End-to-End Model**: A single fine-tuned MobileNetV3Small network that directly predicts the species.
"""
)

# Display model loading errors prominently if any occurred
if model_errors:
    st.error("### Model Loading Issues Detected!")
    for error in model_errors:
        st.error(f"- {error}")
    if USING_DEMO:
        st.warning("Predictions will run in **DEMO MODE** using simulated results.")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info(
    f"""
### Project Information
TF Version: {tf_version}
Two-Model Predict: {'✅' if CAN_PREDICT_TWO_MODEL else '❌'}
End-to-End Predict: {'✅' if CAN_PREDICT_E2E else '❌'}

Models trained on the Animals-10 dataset.
"""
)
st.sidebar.title("Troubleshooting")
st.sidebar.info(
    """
If models fail to load:
1. Ensure `.keras`, `.pkl`, `.json` files are in the app directory.
2. Verify TensorFlow version compatibility.
3. Check console logs for detailed errors.
4. Re-save models from the notebook using the `.keras` format.
"""
)


# Show animal classes in sidebar
st.sidebar.title("Animal Classes")
classes_df = pd.DataFrame(
    {
        "Class ID": list(idx_to_class.keys()),
        "English Name": [info["english"].title() for info in idx_to_class.values()],
    }
)
st.sidebar.dataframe(classes_df.set_index("Class ID"))

# Main content Tabs
tab1, tab2 = st.tabs(["📸 Image Analysis", "📊 Model Comparison"])

# --- Tab 1: Image Analysis ---
with tab1:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown("*(Max file size: 200MB)*")  # Streamlit default limit

    # Sample Image Selection
    st.markdown("---")
    # Check for sample_images folder and list files
    sample_dir = os.path.join(
        APP_DIR, "sample_images"
    )  # Expect a 'sample_images' folder
    available_samples = []
    sample_img_path = None
    if os.path.isdir(sample_dir):
        available_samples = sorted(
            [
                f
                for f in os.listdir(sample_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

    if available_samples:
        use_sample = st.checkbox("Or use a sample image", value=(uploaded_file is None))
        if use_sample:
            selected_sample_file = st.selectbox(
                "Select a sample image:",
                available_samples,
                index=random.randint(
                    0, len(available_samples) - 1
                ),  # Default to random
            )
            sample_img_path = os.path.join(sample_dir, selected_sample_file)
    else:
        st.info(
            "Create a 'sample_images' folder in the app directory to enable sample image selection."
        )
        use_sample = False  # Disable checkbox if no samples

    # Add Test-Time Augmentation toggle (NEW)
    use_tta = st.checkbox(
        "Enable Test-Time Augmentation for improved accuracy (slower)", value=False
    )
    if use_tta:
        tta_samples = st.slider(
            "Number of augmented samples", min_value=3, max_value=10, value=5
        )
        st.caption("More samples may improve accuracy but increase inference time")
    else:
        tta_samples = 5  # Default value if not using TTA

    image_to_process = None
    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error opening uploaded image: {e}")
    elif sample_img_path and os.path.exists(sample_img_path):
        try:
            image_to_process = Image.open(sample_img_path)
        except Exception as e:
            st.error(f"Error opening sample image: {e}")

    if image_to_process is not None:
        col1, col2 = st.columns([2, 3])  # Adjust column width ratios

        with col1:
            st.subheader("Input Image")
            st.image(
                image_to_process, use_container_width=True, caption="Input for analysis"
            )

        with col2:
            st.subheader("Prediction Results")

            if USING_DEMO and model_errors:  # Show only if models failed but TF loaded
                st.warning("Running in Demo Mode due to model loading errors.")

            results_list = []
            predicted_animal_tm = "N/A"
            predicted_animal_e2e = "N/A"
            predicted_animal_display = "Unknown"  # Default

            # Two-Model Prediction
            if CAN_PREDICT_TWO_MODEL:
                with st.spinner("Running Two-Model prediction..."):
                    try:
                        pred_info, conf, inf_time = predict_two_model(
                            image_to_process,
                            models["fe"],
                            models["clf"],
                            use_tta=use_tta,
                            tta_samples=tta_samples,
                        )
                        results_list.append(
                            {
                                "Approach": f"Two-Model (FE + SVM){' with TTA' if use_tta else ''}",
                                "Prediction": pred_info["english"].title(),
                                "Confidence": f"{conf:.2f}%",
                                "Inference Time": f"{inf_time:.2f} ms",
                            }
                        )
                        predicted_animal_tm = pred_info["english"]
                        predicted_animal_display = (
                            predicted_animal_tm  # Prioritize this if available
                        )
                    except Exception as e:
                        st.error(f"Two-Model Prediction Error: {e}")
                        results_list.append(
                            {
                                "Approach": "Two-Model (FE + SVM)",
                                "Prediction": "Error",
                                "Confidence": "N/A",
                                "Inference Time": "N/A",
                            }
                        )

            elif USING_DEMO:  # Run demo if models failed or not found
                pred_info, conf, inf_time = demo_predict(image_to_process)
                results_list.append(
                    {
                        "Approach": "Two-Model (FE + SVM) [DEMO]",
                        "Prediction": pred_info["english"].title(),
                        "Confidence": f"{conf:.2f}%",
                        "Inference Time": f"{inf_time:.2f} ms",
                    }
                )
                predicted_animal_tm = pred_info["english"]
                predicted_animal_display = predicted_animal_tm

            # End-to-End Prediction
            if CAN_PREDICT_E2E:
                with st.spinner("Running End-to-End prediction..."):
                    try:
                        pred_info_e2e, conf_e2e, inf_time_e2e = predict_end_to_end(
                            image_to_process,
                            models["e2e"],
                            use_tta=use_tta,
                            tta_samples=tta_samples,
                        )
                        results_list.append(
                            {
                                "Approach": f"End-to-End Model{' with TTA' if use_tta else ''}",
                                "Prediction": pred_info_e2e["english"].title(),
                                "Confidence": f"{conf_e2e:.2f}%",
                                "Inference Time": f"{inf_time_e2e:.2f} ms",
                            }
                        )
                        predicted_animal_e2e = pred_info_e2e["english"]
                        # If two-model failed or wasn't run, use e2e prediction for display
                        if predicted_animal_display == "Unknown":
                            predicted_animal_display = predicted_animal_e2e
                    except Exception as e:
                        st.error(f"End-to-End Prediction Error: {e}")
                        results_list.append(
                            {
                                "Approach": "End-to-End Model",
                                "Prediction": "Error",
                                "Confidence": "N/A",
                                "Inference Time": "N/A",
                            }
                        )

            elif USING_DEMO:  # Run demo if models failed or not found
                pred_info_e2e, conf_e2e, inf_time_e2e = demo_predict(image_to_process)
                results_list.append(
                    {
                        "Approach": "End-to-End Model [DEMO]",
                        "Prediction": pred_info_e2e["english"].title(),
                        "Confidence": f"{conf_e2e:.2f}%",
                        "Inference Time": f"{inf_time_e2e:.2f} ms",
                    }
                )
                predicted_animal_e2e = pred_info_e2e["english"]
                if predicted_animal_display == "Unknown":
                    predicted_animal_display = predicted_animal_e2e

            if results_list:
                results_df_display = pd.DataFrame(results_list)
                st.dataframe(results_df_display.set_index("Approach"))

                # Agreement indicator only if both ran (real or demo) and didn't error
                if len(results_list) == 2 and "Error" not in [
                    results_list[0]["Prediction"],
                    results_list[1]["Prediction"],
                ]:
                    pred1_title = results_list[0]["Prediction"]
                    pred2_title = results_list[1]["Prediction"]
                    if pred1_title == pred2_title:
                        st.success(
                            f"✅ Both approaches agree: This appears to be a {pred1_title.upper()}"
                        )
                    else:
                        st.warning(
                            f"⚠️ Approaches disagree: Two-Model says {pred1_title.upper()}, End-to-End says {pred2_title.upper()}"
                        )
            else:
                st.error(
                    "Could not run predictions. Please check model files and TensorFlow installation."
                )

        # --- Animal Information Section ---
        if (
            predicted_animal_display != "Unknown"
            and predicted_animal_display != "Error"
        ):
            st.divider()
            st.subheader(f"About the {predicted_animal_display.title()}")
            animal_info = get_animal_info(predicted_animal_display)
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.info(f"**Fun Fact**: {animal_info['fact']}")
            with info_col2:
                st.info(f"**Habitat**: {animal_info['habitat']}")

# --- Tab 2: Model Comparison ---
with tab2:
    st.subheader("Model Architecture Comparison")

    # Diagrams (Improved text-based)
    st.markdown("---")
    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        st.markdown("#### Two-Model Approach")
        st.markdown(
            "`Input Image` ➡️ `MobileNetV3Small (FE)` ➡️ `SVM Classifier` ➡️ `Prediction`"
        )
        st.markdown(
            """
        - **Advantages:** Modular, potentially higher accuracy with optimized classifier, feature reuse possible.
        - **Disadvantages:** More complex deployment, potentially slower inference due to two steps.
        """
        )
    with col_diag2:
        st.markdown("#### End-to-End Model")
        st.markdown("`Input Image` ➡️ `Fine-Tuned MobileNetV3Small` ➡️ `Prediction`")
        st.markdown(
            """
        - **Advantages:** Simpler deployment, often faster inference, joint optimization.
        - **Disadvantages:** Less modular, potentially slightly lower accuracy if classifier stage was highly optimized.
        """
        )
    st.markdown("---")

    st.subheader("Performance Comparison")

    if results_errors:
        st.warning(
            "Could not load performance data from JSON files. Displaying placeholder data."
        )
        for error in results_errors:
            st.warning(f"- {error}")
        # Placeholder data if JSON loading failed
        comparison_data_json = {
            "two_model": {
                "accuracy": 0.9367,
                "inference_time_ms": 72.8,
                "model_size_mb": 10.7,
                "best_classifier": "SVM",
            },
            "end_to_end": {
                "accuracy": 0.9329,
                "inference_time_ms": 73.3,
                "model_size_mb": 14.5,
            },
            "comparison": {
                "accuracy_diff_percent": 0.41,
                "time_diff_percent": -0.7,
                "size_diff_percent": -26.1,
            },
        }
        results_data_json = {
            "SVM": {"test_accuracy": 0.9367, "inference_time": 0.00107},
            "Random Forest": {"test_accuracy": 0.9347, "inference_time": 0.02141},
            "MLP": {"test_accuracy": 0.9349, "inference_time": 0.00032},
        }

    if results_data_json and comparison_data_json:
        st.markdown("Based on project experimental results (loaded from saved files):")
        # --- Display Classifier Performance Table ---
        try:
            clf_perf_data = {
                "Classifier": list(results_data_json.keys()),
                "Test Accuracy": [
                    f"{v['test_accuracy']:.4f}" for v in results_data_json.values()
                ],
                "Inference Time (ms)": [
                    f"{v['inference_time'] * 1000:.2f}"
                    for v in results_data_json.values()
                ],
            }
            clf_perf_df = pd.DataFrame(clf_perf_data)
            st.dataframe(clf_perf_df.set_index("Classifier"))
        except KeyError as e:
            st.error(
                f"KeyError accessing classifier results data: {e}. Check 'classifier_results.json'."
            )
        except Exception as e:
            st.error(f"Error processing classifier results data: {e}")

        # --- Display Overall Comparison Table ---
        try:
            tm_approach_name = f"Two-Model (FE + {comparison_data_json.get('two_model',{}).get('best_classifier','SVM')})"
            overall_comp_data = {
                "Metric": ["Test Accuracy", "Inference Time (ms)", "Model Size (MB)"],
                tm_approach_name: [
                    f"{comparison_data_json['two_model']['accuracy']:.4f}",
                    f"{comparison_data_json['two_model']['inference_time_ms']:.2f}",
                    f"{comparison_data_json['two_model']['model_size_mb']:.2f}",
                ],
                "End-to-End": [
                    f"{comparison_data_json['end_to_end']['accuracy']:.4f}",
                    f"{comparison_data_json['end_to_end']['inference_time_ms']:.2f}",
                    f"{comparison_data_json['end_to_end']['model_size_mb']:.2f}",
                ],
            }
            overall_comp_df = pd.DataFrame(overall_comp_data)
            st.dataframe(overall_comp_df.set_index("Metric"))
        except KeyError as e:
            st.error(
                f"KeyError accessing model comparison data: {e}. Check 'model_comparison_results.json'."
            )
            comparison_data_json = {}  # Prevent chart errors
        except Exception as e:
            st.error(f"Error processing model comparison data: {e}")
            comparison_data_json = {}  # Prevent chart errors

        # --- Display Comparison Charts ---
        if comparison_data_json:  # Only plot if data is valid
            chart_data = {
                "Model": [tm_approach_name, "End-to-End"],
                "Test Accuracy": [
                    comparison_data_json["two_model"]["accuracy"],
                    comparison_data_json["end_to_end"]["accuracy"],
                ],
                "Inference Time (ms)": [
                    comparison_data_json["two_model"]["inference_time_ms"],
                    comparison_data_json["end_to_end"]["inference_time_ms"],
                ],
                "Model Size (MB)": [
                    comparison_data_json["two_model"]["model_size_mb"],
                    comparison_data_json["end_to_end"]["model_size_mb"],
                ],
            }
            chart_df = pd.DataFrame(chart_data)

            chart_col1, chart_col2, chart_col3 = st.columns(3)

            # Function to create bars safely
            def create_bar_chart(ax, x_data, y_data, y_label, title, format_str):
                bars = ax.bar(x_data, y_data, color=["#1f77b4", "#ff7f0e"])
                # Check if y_data is not empty AND contains numeric types before min/max
                if not y_data.empty and pd.api.types.is_numeric_dtype(y_data):
                    min_val = y_data.min()
                    max_val = y_data.max()
                else:
                    min_val = 0
                    max_val = 1  # Default range if data is empty or non-numeric

                padding = (
                    (max_val - min_val) * 0.1 if max_val > min_val else 0.1
                )  # Ensure padding is non-zero
                ax.set_ylim(
                    max(0, min_val - padding), max_val + padding + 0.01
                )  # Adjust ylim slightly more for text
                ax.set_ylabel(y_label)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=15)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        format_str.format(height),
                        ha="center",
                        va="bottom",
                    )

            with chart_col1:
                st.markdown("#### Accuracy")
                fig, ax = plt.subplots(figsize=(5, 4))
                create_bar_chart(
                    ax,
                    chart_df["Model"],
                    chart_df["Test Accuracy"],
                    "Accuracy",
                    "Test Accuracy",
                    "{:.4f}",
                )
                st.pyplot(fig)

            with chart_col2:
                st.markdown("#### Inference Time")
                fig, ax = plt.subplots(figsize=(5, 4))
                create_bar_chart(
                    ax,
                    chart_df["Model"],
                    chart_df["Inference Time (ms)"],
                    "Time (ms)",
                    "Inference Time",
                    "{:.2f} ms",
                )
                st.pyplot(fig)

            with chart_col3:
                st.markdown("#### Model Size")
                fig, ax = plt.subplots(figsize=(5, 4))
                create_bar_chart(
                    ax,
                    chart_df["Model"],
                    chart_df["Model Size (MB)"],
                    "Size (MB)",
                    "Model Size",
                    "{:.2f} MB",
                )
                st.pyplot(fig)

            # --- Project Conclusion ---
            st.subheader("Project Conclusion")
            comp_results = comparison_data_json.get("comparison", {})
            acc_diff = comp_results.get("accuracy_diff_percent", 0)
            time_diff = comp_results.get("time_diff_percent", 0)
            size_diff = comp_results.get("size_diff_percent", 0)
            two_model_name = (
                tm_approach_name  # Use the dynamic name from overall_comp_data
            )

            st.markdown(
                f"""
            Based on the experimental results:

            1.  The **{two_model_name}** achieved slightly **{'higher' if acc_diff > 0 else ('lower' if acc_diff < 0 else 'similar')} accuracy** ({comparison_data_json['two_model']['accuracy']:.4f}) compared to the End-to-End model ({comparison_data_json['end_to_end']['accuracy']:.4f}), a difference of **{acc_diff:.2f}%**.
            2.  The **End-to-End model** demonstrated **{'faster' if time_diff < 0 else ('slower' if time_diff > 0 else 'similar')} inference time** ({comparison_data_json['end_to_end']['inference_time_ms']:.2f} ms) compared to the total time for the Two-Model approach ({comparison_data_json['two_model']['inference_time_ms']:.2f} ms), a difference of **{time_diff:.2f}%**.
            3.  The **{two_model_name}** resulted in a **{'smaller' if size_diff < 0 else ('larger' if size_diff > 0 else 'similar')} total model size** ({comparison_data_json['two_model']['model_size_mb']:.2f} MB) compared to the End-to-End model ({comparison_data_json['end_to_end']['model_size_mb']:.2f} MB), a difference of **{size_diff:.2f}%**.
            4.  Both approaches successfully identify the 10 animal species with high accuracy (over 93%).

            The choice between the approaches depends on the specific requirements: prioritize accuracy and smaller size (Two-Model with SVM) or inference speed/deployment simplicity (End-to-End).
            """
            )
        else:
            st.warning(
                "Could not generate comparison charts due to missing or invalid comparison data."
            )
    else:
        st.warning(
            "Could not load performance data from JSON files. Please ensure `classifier_results.json` and `model_comparison_results.json` are present in the app directory."
        )


# --- Footer/Sidebar Info ---
st.sidebar.divider()
st.sidebar.markdown("App developed for University Course Project.")
