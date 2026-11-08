"""Streamlit Interactive Research Dashboard: Medical SSL Representation Learning.

Enables interactive exploration of PathMNIST histopathology patches, real-time
multi-model comparative inference, label efficiency trajectories, and latent representation geometry.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config, get_device
from src.models import ResNet18Encoder, SupervisedResNet18, LinearProbeClassifier, FineTuneClassifier

# Page configuration
st.set_page_config(
    page_title="Medical SSL Representation Learning",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1A365D; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4A5568; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F7FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; }
    .badge-correct { background-color: #C6F6D5; color: #22543D; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-incorrect { background-color: #FED7D7; color: #742A2A; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_torch_device():
    return get_device("auto")


@st.cache_data
def load_project_config():
    return load_config(os.path.join(PROJECT_ROOT, "configs", "config.yaml"))


@st.cache_data
def load_test_dataset():
    data_path = os.path.join(PROJECT_ROOT, "data", "pathmnist.npz")
    data = np.load(data_path)
    test_images = data["test_images"]
    test_labels = data["test_labels"].squeeze()
    return test_images, test_labels


@st.cache_resource
def load_model(method_name: str, proportion: float):
    device = get_torch_device()
    prop_str = f"prop{int(proportion*100):03d}"
    checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    if method_name == "Supervised Scratch":
        ckpt_name = f"supervised_scratch_{prop_str}_seed42_best.pth"
        model = SupervisedResNet18(num_classes=9, pretrained=False).to(device)
    elif method_name == "ImageNet Transfer":
        ckpt_name = f"imagenet_transfer_{prop_str}_seed42_best.pth"
        model = SupervisedResNet18(num_classes=9, pretrained=False).to(device)
    elif method_name == "SimCLR Linear Probe":
        ckpt_name = f"simclr_linear_probe_{prop_str}_seed42_best.pth"
        encoder = ResNet18Encoder(pretrained=False)
        model = LinearProbeClassifier(encoder=encoder, num_classes=9).to(device)
    elif method_name == "SimCLR Fine-Tuning":
        ckpt_name = f"simclr_fine-tuning_{prop_str}_seed42_best.pth"
        encoder = ResNet18Encoder(pretrained=False)
        model = FineTuneClassifier(encoder=encoder, num_classes=9).to(device)
    else:
        raise ValueError(f"Unknown method {method_name}")

    ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        return None

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(img_np: np.ndarray, config: dict, device: torch.device) -> torch.Tensor:
    # img_np: (28, 28, 3) uint8 -> tensor (1, 3, 28, 28) normalized
    norm = config["augmentations"]["normalization"]
    mean = torch.tensor(norm["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(norm["std"]).view(1, 3, 1, 1).to(device)

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)
    tensor = (tensor - mean) / std
    return tensor


# Load data and configuration
config = load_project_config()
class_names = [config["dataset"]["classes"][i] for i in range(config["dataset"]["num_classes"])]
test_images, test_labels = load_test_dataset()

# Sidebar
st.sidebar.image("https://img.shields.io/badge/MedMNIST-PathMNIST-brightgreen", width=180)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View:",
    [
        "🔬 Interactive Patch Classifier",
        "📊 Master Benchmark Leaderboard",
        "🧬 Latent Representation Geometry",
        "📖 Clinical Context & Methodology",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Device Acceleration:** `" + str(get_torch_device()) + "`\n\n"
    "**Dataset:** MedMNIST v2 (PathMNIST)\n\n"
    "**External Test Cohort:** CRC-VAL-HE-7K ($N=7,180$)\n\n"
    "**Resolution:** $28 \\times 28 \\times 3$"
)

# Header
st.markdown('<div class="main-header">🔬 Medical Self-Supervised Representation Learning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Evaluating SimCLR Contrastive Pretraining vs. Supervised Baselines & ImageNet Transfer Under Limited Histopathology Annotations</div>',
    unsafe_allow_html=True,
)


# ==============================================================================
# Page 1: Interactive Patch Classifier
# ==============================================================================
if page == "🔬 Interactive Patch Classifier":
    st.subheader("Interactive Multi-Model Inference & Side-by-Side Comparison")
    st.markdown(
        "Select a tissue patch from the untouched **CRC-VAL-HE-7K external test cohort** and compare real-time "
        "inference across all 4 paradigms at your chosen label availability regime."
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        selected_class = st.selectbox(
            "Filter Test Samples by Ground-Truth Tissue Type:",
            ["All Classes"] + [f"{i}: {name}" for i, name in enumerate(class_names)],
            index=3,  # default to debris or lymphocytes
        )

    # Filter indices based on class selection
    if selected_class == "All Classes":
        valid_indices = np.arange(len(test_labels))
    else:
        target_class_id = int(selected_class.split(":")[0])
        valid_indices = np.where(test_labels == target_class_id)[0]

    with col_ctrl2:
        sample_choice = st.slider(
            f"Select Sample Index (out of {len(valid_indices)}):",
            min_value=0,
            max_value=len(valid_indices) - 1,
            value=min(12, len(valid_indices) - 1),
        )
        sample_idx = valid_indices[sample_choice]

    with col_ctrl3:
        proportion = st.selectbox(
            "Training Label Availability Regime:",
            [0.01, 0.05, 0.10, 1.00],
            format_func=lambda p: f"{int(p*100)}% Labels ({int(p*89996):,} annotated samples)",
            index=0,  # default to 1% regime to showcase label efficiency
        )

    st.markdown("---")

    # Display selected patch
    patch = test_images[sample_idx]
    true_label = test_labels[sample_idx]
    true_class_name = class_names[true_label]

    col_img, col_info = st.columns([1, 3])
    with col_img:
        st.markdown(f"**Sample #{sample_idx} (28x28 H&E Patch)**")
        # Upscale for visual inspection
        st.image(patch, width=170, caption=f"Ground Truth: {true_class_name}")

    with col_info:
        st.markdown("#### Tissue Ground-Truth Metadata")
        st.markdown(f"- **Class ID:** `{true_label}`")
        st.markdown(f"- **Class Name:** **{true_class_name}**")
        st.markdown(f"- **Evaluating Regime:** **{int(proportion*100)}% Labeled Data** (~{int(proportion*89996)//9} per class)")
        st.info(f"Analyzing discriminative feature responses from external cohort patient test slides...")

    st.markdown("### Model Predictions Comparison")

    methods = [
        "Supervised Scratch",
        "ImageNet Transfer",
        "SimCLR Linear Probe",
        "SimCLR Fine-Tuning",
    ]

    cols = st.columns(4)
    device = get_torch_device()
    input_tensor = preprocess_image(patch, config, device)

    for i, method in enumerate(methods):
        with cols[i]:
            model = load_model(method, proportion)
            if model is None:
                st.warning(f"Checkpoint for {method} at {int(proportion*100)}% not found.")
                continue

            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred_class_id = int(np.argmax(probs))
                pred_prob = float(probs[pred_class_id])

            is_correct = (pred_class_id == true_label)
            pred_class_name = class_names[pred_class_id]

            st.markdown(f"#### {method}")
            if is_correct:
                st.markdown(f'<span class="badge-correct">✓ Correct ({pred_prob*100:.1f}%)</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge-incorrect">✗ Incorrect ({pred_prob*100:.1f}%)</span>', unsafe_allow_html=True)

            st.write(f"**Predicted:** {pred_class_name}")

            # Plot mini horizontal probability bar chart
            fig, ax = plt.subplots(figsize=(4, 2.8))
            y_pos = np.arange(len(class_names))
            colors = ["#319795" if c == true_label else ("#DD6B20" if c == pred_class_id else "#CBD5E0") for c in range(len(class_names))]
            ax.barh(y_pos, probs, color=colors, height=0.65)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([c[:12] for c in class_names], fontsize=7)
            ax.set_xlim(0, 1.0)
            ax.invert_yaxis()
            ax.set_xlabel("Confidence", fontsize=7)
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ==============================================================================
# Page 2: Master Benchmark Leaderboard
# ==============================================================================
elif page == "📊 Master Benchmark Leaderboard":
    st.subheader("Official Master Benchmark Performance")
    st.markdown(
        "All metrics reported on the **quarantined external test set CRC-VAL-HE-7K ($N = 7,180$)** under controlled Seed 42."
    )

    summary_csv_path = os.path.join(PROJECT_ROOT, "results", "metrics", "results_summary.csv")
    if os.path.exists(summary_csv_path):
        df_summary = pd.read_csv(summary_csv_path)

        display_cols = [
            "method",
            "proportion",
            "n_samples",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_macro_roc_auc",
            "training_time_sec",
        ]
        df_disp = df_summary[display_cols].copy()
        df_disp["proportion"] = df_disp["proportion"].apply(lambda p: f"{int(p*100)}%")
        df_disp.rename(
            columns={
                "method": "Paradigm",
                "proportion": "Labels (%)",
                "n_samples": "N Annotated",
                "test_accuracy": "Accuracy",
                "test_balanced_accuracy": "Balanced Acc",
                "test_macro_f1": "Macro F1",
                "test_macro_roc_auc": "Macro ROC-AUC",
                "training_time_sec": "Train Time (s)",
            },
            inplace=True,
        )

        st.dataframe(
            df_disp.sort_values(by=["Labels (%)", "Macro F1"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Label Efficiency Scaling Trajectories")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        f1_plot = os.path.join(PROJECT_ROOT, "results", "plots", "label_efficiency_macro_f1.png")
        if os.path.exists(f1_plot):
            st.image(f1_plot, caption="Macro F1 Scaling across Proportions (Log Scale)")
    with col_p2:
        acc_plot = os.path.join(PROJECT_ROOT, "results", "plots", "label_efficiency_accuracy.png")
        if os.path.exists(acc_plot):
            st.image(acc_plot, caption="Test Accuracy Scaling across Proportions (Log Scale)")

    st.markdown("### Paradigm Grouped Comparison")
    bar_plot = os.path.join(PROJECT_ROOT, "results", "plots", "model_comparison_bar.png")
    if os.path.exists(bar_plot):
        st.image(bar_plot, caption="Macro F1 Performance Grouped by Labeled Proportion")


# ==============================================================================
# Page 3: Latent Representation Geometry
# ==============================================================================
elif page == "🧬 Latent Representation Geometry":
    st.subheader("t-SNE Representation Projections ($h \\in \\mathbb{R}^{512}$)")
    st.markdown(
        "Low-dimensional manifold projections showing how each training objective clusters or separates "
        "morphologically distinct histopathology categories on the external test set."
    )

    tsne_plot = os.path.join(PROJECT_ROOT, "results", "plots", "tsne_representations.png")
    if os.path.exists(tsne_plot):
        st.image(tsne_plot, use_container_width=True, caption="2D t-SNE Embeddings of 1,500 Test Samples across Paradigms")

    st.markdown("### Key Representation Insights:")
    st.markdown(
        """
        - **Supervised Scratch:** Forms tight, discrete semantic clusters when 100% labeled annotations are available.
        - **ImageNet Transfer:** Provides broadly distributed, linearly separable representations that generalize effectively to low-data histopathology regimes.
        - **SimCLR Linear Probe:** Preserves continuous histological manifolds where morphologically adjacent tissues (e.g. stroma and smooth muscle) lie in contiguous feature space.
        - **SimCLR Fine-Tuning:** Successfully warps unsupervised contrastive geometry toward disease classification boundaries.
        """
    )


# ==============================================================================
# Page 4: Clinical Context & Methodology
# ==============================================================================
elif page == "📖 Clinical Context & Methodology":
    st.subheader("Histopathological Taxonomy & Clinical Reality")
    st.markdown(
        """
        ### The 9 Colorectal Tissue Classes (PathMNIST / CRC-VAL-HE-7K):
        1. **Adipose (ADI):** Mature adipocytes with clear lipid vacuoles.
        2. **Background (BACK):** Optical slide voids and clear space devoid of tissue or stain.
        3. **Debris (DEB):** Necrotic fragments, acute inflammation, and staining artifacts.
        4. **Lymphocytes (LYM):** Small, round, basophilic immune cells indicating tumor-infiltrating lymphocytes.
        5. **Mucus (MUC):** Extracellular basophilic mucinous pools.
        6. **Smooth Muscle (MUS):** Dense bundles of elongated spindle cells of the colonic wall.
        7. **Normal Colon Mucosa (NORM):** Organized, non-neoplastic colonic crypts and goblet cells.
        8. **Cancer-Associated Stroma (STR):** Desmoplastic extracellular collagen and activated myofibroblasts.
        9. **Colorectal Adenocarcinoma Epithelium (TUM):** Malignant epithelial glands exhibiting atypia and nuclear hyperchromasia.
        
        ---
        ### Important Experimental Scope & Reality Disclosures:
        - **Standardized $28 \\times 28$ Image Resolution:** All experiments utilize standardized $28 \\times 28$ patches from MedMNIST v2. Subcellular characteristics (mitotic figures, nuclear pleomorphism) cannot be resolved; models rely on gross architecture, stain texture, and cellular density.
        - **Single-Seed Pilot Benchmark:** All 16 experimental runs were conducted on controlled Seed 42 for internal cross-model parity.
        - **Diagnostic Disclaimer:** This repository is an academic machine learning research project and **not** a clinical diagnostic tool.
        """
    )

    pdf_path = os.path.join(PROJECT_ROOT, "results", "reports", "technical_project_report.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="📄 Download 8-Page Technical Project Report PDF (1.45 MB)",
            data=pdf_bytes,
            file_name="medical_ssl_technical_report.pdf",
            mime="application/pdf",
        )

