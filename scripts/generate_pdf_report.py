#!/usr/bin/env python3
"""
Generate a professional Technical Project Report PDF:
'Self-Supervised Representation Learning for Histopathological Image Classification Under Limited Label Availability'

Uses matplotlib's vector PDF backend (matplotlib.backends.backend_pdf.PdfPages)
with embedded high-resolution figures, styled tables, typography, and page numbers.
Targeted for AI/ML Engineering Portfolios and Graduate Admissions Technical Documentation.
"""

import os
import sys

# Ensure clean cache directory for matplotlib inside sandbox
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

PDF_OUTPUT_PATH = os.path.join(REPORTS_DIR, "technical_project_report.pdf")

# Visual Palette (Clean Engineering & Portfolio Style)
NAVY = "#1A365D"
DARK_SLATE = "#2D3748"
MUTED_SLATE = "#4A5568"
LIGHT_BG = "#F7FAFC"
CARD_BORDER = "#CBD5E0"
ACCENT_BLUE = "#2B6CB0"
ACCENT_TEAL = "#319795"
ACCENT_GREEN = "#38A169"
ACCENT_AMBER = "#DD6B20"
ALERT_RED = "#C53030"
LIGHT_BLUE = "#EBF8FF"
LIGHT_AMBER = "#FFFAF0"

PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11.0
DPI = 300

def draw_header_footer(ax, page_num, total_pages, title="Technical Project Report: Self-Supervised Medical Representation Learning"):
    """Adds formal running header and footer with page number."""
    # Running header
    ax.text(0.06, 0.965, title, fontsize=8, color=MUTED_SLATE, fontstyle="italic", va="center")
    ax.plot([0.06, 0.94], [0.955, 0.955], color="#CBD5E0", lw=0.75)
    
    # Running footer
    ax.plot([0.06, 0.94], [0.045, 0.045], color="#CBD5E0", lw=0.75)
    ax.text(0.06, 0.035, "AI/ML Engineering Portfolio & Graduate Admissions Technical Document", fontsize=7.5, color=MUTED_SLATE, va="center")
    ax.text(0.94, 0.035, f"Page {page_num} of {total_pages}", fontsize=7.5, color=MUTED_SLATE, ha="right", va="center")


def build_page_1(pdf, total_pages=8):
    """Page 1: Project Overview, Problem Statement, Architecture Overview"""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    draw_header_footer(ax, 1, total_pages)
    
    # Title Block
    title_patch = FancyBboxPatch((0.06, 0.81), 0.88, 0.125, boxstyle="round,pad=0.015", 
                                 edgecolor=NAVY, facecolor=LIGHT_BLUE, lw=1.5)
    ax.add_patch(title_patch)
    
    ax.text(0.5, 0.895, "Self-Supervised Representation Learning for Histopathological\nImage Classification Under Limited Label Availability",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color=NAVY, linespacing=1.2)
    
    metadata_text = (
        "Technical Project Report  |  AI/ML Portfolio & Graduate Application Document  |  Date: November 2026\n"
        "Benchmark: MedMNIST v2 PathMNIST (NCT-CRC-HE-100K Training Split & CRC-VAL-HE-7K External Test Set)\n"
        "Framework & Hardware: PyTorch 2.6.0  |  Apple Silicon MPS Unified Memory Acceleration (float32)"
    )
    ax.text(0.5, 0.835, metadata_text, ha="center", va="center", fontsize=7.8, color=DARK_SLATE, linespacing=1.25)
    
    # Project Overview Card
    overview_patch = FancyBboxPatch((0.06, 0.44), 0.88, 0.35, boxstyle="round,pad=0.012",
                                    edgecolor=CARD_BORDER, facecolor="#FFFFFF", lw=1.0)
    ax.add_patch(overview_patch)
    
    ax.text(0.08, 0.765, "1. PROJECT OVERVIEW & PROBLEM STATEMENT", fontsize=10.5, fontweight="bold", color=ACCENT_BLUE)
    
    p1 = (
        "In deep learning for digital histopathology, obtaining high-quality ground-truth annotations requires extensive manual labor "
        "by trained medical specialists. In real-world engineering deployments—such as adapting a model to a new clinical laboratory, "
        "hospital site, or rare diagnostic category—labeled examples are often strictly limited. Standard industry practice relies on "
        "transfer learning from ImageNet (natural photographs), despite the pronounced domain gap with cellular tissue morphology."
    )
    ax.text(0.08, 0.74, p1, fontsize=8.2, color=DARK_SLATE, va="top", wrap=True, linespacing=1.35)
    
    p2 = (
        "This project designs, implements, and benchmarks an end-to-end medical representation learning system in PyTorch. "
        "We evaluate whether domain-specific, self-supervised contrastive pretraining using SimCLR on unlabeled colorectal histology "
        "patches yields better downstream representations than (a) supervised learning trained from scratch with random initialization, "
        "and (b) standard ImageNet transfer learning, across four controlled label fractions: 1% (N=899), 5% (N=4,499), 10% (N=8,999), "
        "and 100% (N=89,996). All models utilize an identical ResNet-18 backbone evaluated on CRC-VAL-HE-7K (N=7,180)."
    )
    ax.text(0.08, 0.65, p2, fontsize=8.2, color=DARK_SLATE, va="top", wrap=True, linespacing=1.35)
    
    ax.text(0.08, 0.575, "Summary of Key Engineering Observations:", fontsize=8.8, fontweight="bold", color=NAVY)
    
    observations = [
        "• 1% Label Regime Performance: ImageNet Transfer achieved the highest Macro F1 (0.7449), followed by SimCLR Fine-Tuning "
        "(0.7408), both outperforming Supervised Scratch (0.6815). SimCLR Fine-Tuning achieved highest discriminative confidence "
        "(ROC-AUC: 0.9662) and outpaced Scratch by +5.05% Test Accuracy. A frozen linear probe achieved 0.6943 Macro F1.",
        
        "• Transfer Learning Strength in Low-to-Medium Regimes: ImageNet Transfer achieves the highest Macro F1 at 1% (0.7449), "
        "5% (0.8375), and 10% (0.8495), showing that general low-level visual filters transfer effectively to microscopic textures.",
        
        "• High-Data Regime Crossover at 100% Labels: When all 89,996 labeled samples are available, Supervised Scratch achieves the "
        "highest overall performance across all paradigms (90.18% Accuracy, 0.8721 Macro F1, 0.9916 ROC-AUC).",
        
        "• Representation Geometry: Supervised Scratch exhibits the tightest cluster compaction (Silhouette: 0.3197), while SimCLR "
        "discovers natural morphological clusters entirely without labels (Silhouette: 0.1162 vs -0.0412 for random initialization)."
    ]
    
    y_pos = 0.555
    for obs in observations:
        ax.text(0.09, y_pos, obs, fontsize=7.8, color=DARK_SLATE, va="top", wrap=True, linespacing=1.25)
        y_pos -= 0.027
        
    # System Architecture Card (Bottom)
    arch_patch = FancyBboxPatch((0.06, 0.07), 0.88, 0.34, boxstyle="round,pad=0.012",
                                edgecolor=CARD_BORDER, facecolor="#F8FAFC", lw=1.0)
    ax.add_patch(arch_patch)
    
    ax.text(0.08, 0.385, "2. SYSTEM ARCHITECTURE & PIPELINE TAXONOMY", fontsize=9.5, fontweight="bold", color=NAVY)
    
    arch_desc = (
        "Pipeline Taxonomy & Architectural Flow:\n\n"
        "  • Unlabeled Pretraining Corpus: PathMNIST training split (N = 89,996 colorectal histology patches, 28x28x3 pixels).\n"
        "  • Convolutional Backbone: ResNet-18 f(.) mapping input patch x -> penultimate embedding h in R^512 (11.2M parameters).\n"
        "  • Contrastive Projection Head: 2-layer MLP g(.) mapping h -> z in R^128 with L2 normalization (discarded post-pretraining).\n"
        "  • Domain-Specific Augmentation Policy: Spatial affine transforms, 180-deg rotation, and color jitter used as an approximation\n"
        "    of color variation (not validated stain preservation). Random grayscale is strictly omitted to preserve dye distinction.\n"
        "  • Downstream Adaptation Framework: Evaluates Supervised Scratch, ImageNet Transfer, SimCLR Linear Probe, and SimCLR Fine-Tuning\n"
        "    across 1%, 5%, 10%, and 100% labeled training subsets under deterministic Seed 42 subset parity.\n"
        "  • Evaluation Cohort: External, unseen CRC-VAL-HE-7K cohort (N = 7,180) evaluated exactly once per condition."
    )
    ax.text(0.08, 0.365, arch_desc, fontsize=7.8, color=DARK_SLATE, va="top", family="monospace", linespacing=1.35)
    
    pdf.savefig(fig)
    plt.close(fig)


def build_page_2(pdf, total_pages=8):
    """Page 2: Dataset Description, Resolution Trade-Offs & Methodology"""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    draw_header_footer(ax, 2, total_pages)
    
    ax.text(0.06, 0.925, "3. Dataset Description & Methodology", 
            fontsize=13, fontweight="bold", color=NAVY)
    
    # Dataset Card
    data_patch = FancyBboxPatch((0.06, 0.55), 0.88, 0.35, boxstyle="round,pad=0.012",
                                edgecolor=CARD_BORDER, facecolor="#FFFFFF", lw=1.0)
    ax.add_patch(data_patch)
    
    ax.text(0.08, 0.875, "DATASET SPECIFICATIONS & 28x28 RESOLUTION TRADE-OFFS", fontsize=9.5, fontweight="bold", color=ACCENT_BLUE)
    
    data_text = (
        "Dataset Protocol & Split Protocol:\n"
        "  • Benchmark: PathMNIST (MedMNIST v2), derived from human colorectal cancer cohorts (NCT-CRC-HE-100K & CRC-VAL-HE-7K).\n"
        "  • Pretraining / Training Split: N = 89,996 patches (all labels stripped during SimCLR pretraining).\n"
        "  • Validation Split: N = 10,004 patches (used exclusively for checkpoint selection, validation loss, and early stopping).\n"
        "  • External Test Cohort: CRC-VAL-HE-7K (N = 7,180 patches from independent patients and scanners, evaluated once per model).\n\n"
        "Standardized 28x28 Image Resolution & Engineering Trade-Offs:\n"
        "  • PathMNIST provides standardized 28x28x3 pixel patches. This spatial downsampling captures overall tissue textures, basophilic\n"
        "    versus eosinophilic dye intensity, and macroscopic crypt/gland architecture, but attenuates fine subcellular cytology (nuclear\n"
        "    atypia, mitotic spindles, chromatin margin details) that diagnostic pathologists inspect at 20x or 40x magnification.\n"
        "  • In an engineering context, this resolution enables full-scale contrastive pretraining on ~90,000 images within a practical\n"
        "    computational footprint (24.5 min on Apple Silicon MPS), providing an ideal environment for controlled algorithmic benchmarking.\n\n"
        "Data-Split Isolation Checks:\n"
        "  • The external test set (CRC-VAL-HE-7K) was never used for training, model selection, early stopping, or hyperparameter tuning.\n"
        "  • Downstream subsets (1%, 5%, 10%) were generated exclusively from the training split via StratifiedShuffleSplit (Seed 42)."
    )
    ax.text(0.08, 0.855, data_text, fontsize=7.6, color=DARK_SLATE, va="top", family="monospace", linespacing=1.28)
    
    # Methodology Card
    method_patch = FancyBboxPatch((0.06, 0.07), 0.88, 0.45, boxstyle="round,pad=0.012",
                                  edgecolor=CARD_BORDER, facecolor="#F8FAFC", lw=1.0)
    ax.add_patch(method_patch)
    
    ax.text(0.08, 0.495, "METHODOLOGY: SUPERVISED, TRANSFER & CONTRASTIVE LEARNING", fontsize=9.5, fontweight="bold", color=NAVY)
    
    method_text = (
        "1. Supervised Learning from Scratch:\n"
        "   Randomly initialized ResNet-18 (He normal). All convolutional kernels and classification head are optimized end-to-end\n"
        "   with cross-entropy loss and learning rate lr = 1e-3. Serves as the primary non-pretrained baseline.\n\n"
        "2. ImageNet Transfer Learning:\n"
        "   Backbone initialized with torchvision's official ResNet18_Weights.IMAGENET1K_V1 (1.28M natural images). Fine-tuned end-to-end\n"
        "   with differential learning rates: lr_backbone = 1e-4 and lr_head = 1e-3.\n\n"
        "3. SimCLR Contrastive Formulation & NT-Xent Loss:\n"
        "   SimCLR maps input x -> h -> z in R^128. For a mini-batch of N_b images, 2N_b augmented views are generated. For a positive\n"
        "   pair (i, j) derived from the same source image, the NT-Xent loss treats the remaining 2(N_b - 1) views as negatives:\n\n"
        "       l_(i, j) = - log [ exp( sim(z_i, z_j) / tau ) / ( sum_(k=1)^(2N_b) 1_[k != i] * exp( sim(z_i, z_k) / tau ) ) ]\n\n"
        "   where sim(u, v) = (u^T v) / (||u||_2 ||v||_2) is cosine similarity, and tau = 0.5 is temperature.\n"
        "   Numerical Stability Offset: Self-similarities along the diagonal are masked with S_ii <- -10^4, preventing GPU underflow.\n\n"
        "4. Downstream Adaptation Modes:\n"
        "   • Linear Probe: Backbone strictly frozen; only the linear layer (512 -> 9) is trained (lr = 1e-3). Evaluates frozen feature representations.\n"
        "   • Full Fine-Tuning: Entire network trained end-to-end with differential learning rates (lr_backbone = 1e-4, lr_head = 1e-3)."
    )
    ax.text(0.08, 0.475, method_text, fontsize=7.5, color=DARK_SLATE, va="top", family="monospace", linespacing=1.28)
    
    pdf.savefig(fig)
    plt.close(fig)


def build_page_3(pdf, total_pages=8):
    """Page 3: SimCLR Pretraining Convergence & Hardware Profiling"""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    draw_header_footer(ax, 3, total_pages)
    
    ax.text(0.06, 0.925, "4. SimCLR Pretraining Convergence & Computational Profiling", 
            fontsize=13, fontweight="bold", color=NAVY)
    
    # Embed SimCLR loss curve
    loss_img_path = os.path.join(PLOTS_DIR, "simclr_pretrain_loss.png")
    if os.path.exists(loss_img_path):
        loss_img = mpimg.imread(loss_img_path)
        img_ax = fig.add_axes([0.08, 0.52, 0.84, 0.38])
        img_ax.imshow(loss_img)
        img_ax.axis("off")
        
    # Profiling Card
    hw_patch = FancyBboxPatch((0.06, 0.07), 0.88, 0.42, boxstyle="round,pad=0.012",
                              edgecolor=CARD_BORDER, facecolor="#FFFFFF", lw=1.0)
    ax.add_patch(hw_patch)
    
    ax.text(0.08, 0.465, "PRETRAINING OPTIMIZATION & HARDWARE RUNTIME PROFILING", fontsize=9.5, fontweight="bold", color=NAVY)
    
    pretrain_analysis = (
        "Pretraining Execution & Convergence Profile (PathMNIST Unlabeled Corpus, N = 89,996):\n"
        "  • Optimization Pipeline: AdamW optimizer, decoupled weight decay = 1e-4, batch size = 256 (512 augmented views per step).\n"
        "  • Learning Rate Schedule: Linear warmup for 2 epochs up to base lr = 5e-4, followed by cosine annealing decay to min_lr = 1e-6.\n"
        "  • Convergence Curve: NT-Xent loss dropped monotonically from 3.5252 (Epoch 1) -> 1.1218 (Epoch 5) -> 0.7812 (Epoch 10) "
        "to a final best of 0.6466 (Epoch 15), confirming rapid representation learning without mode collapse or gradient instability.\n"
        "  • Checkpoint Persistence: Saved to checkpoints/simclr_encoder_best.pth (43.1 MB, 11,176,512 ResNet-18 parameters).\n\n"
        "Hardware Profiling & Runtime Accounting (Apple Silicon MPS Acceleration):\n"
        "  • Environment: Apple Silicon M-series unified memory architecture, PyTorch 2.6.0, float32 precision.\n"
        "  • SimCLR Pretraining Wall Time: 24.5 minutes (98.0s per epoch across 351 batches of batch size 256).\n"
        "  • Downstream Adaptation Wall Times Across All 16 Experimental Conditions:\n"
        "      - 1% Regime (N = 899 samples, 14 batches/epoch): 36.9s (Linear Probe) to 67.2s (Scratch). Average epoch = 3.5s.\n"
        "      - 5% Regime (N = 4,499 samples, 70 batches/epoch): 57.6s (Linear Probe) to 133.0s (Scratch). Average epoch = 8.8s.\n"
        "      - 10% Regime (N = 8,999 samples, 140 batches/epoch): 83.0s (Linear Probe) to 235.1s (Fine-Tuning). Average epoch = 15.6s.\n"
        "      - 100% Regime (N = 89,996 samples, 1,406 batches/epoch): 559.3s (Linear Probe) to 2,185.6s (Scratch, 36.4 minutes).\n"
        "  • Total Experimental Compute: ~2.6 hours of dedicated execution on Apple Silicon MPS for pretraining and all 16 conditions.\n"
        "  • Practical Feasibility: These runtime profiles confirm that low-data regimes (1%, 5%, 10%) are computationally accessible "
        "for rapid representation learning development on standard personal workstations."
    )
    ax.text(0.08, 0.445, pretrain_analysis, fontsize=7.6, color=DARK_SLATE, va="top", family="monospace", linespacing=1.28)
    
    pdf.savefig(fig)
    plt.close(fig)


def build_page_4(pdf, total_pages=8):
    """Page 4: Master Results Table & Label Efficiency Macro F1 Plot"""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    draw_header_footer(ax, 4, total_pages)
    
    ax.text(0.06, 0.925, "5. Results and Model Comparison: Master Benchmark", 
            fontsize=13, fontweight="bold", color=NAVY)
    
    # Table Header & Data
    table_data = [
        ["Method", "Data %", "N Samples", "Test Acc", "Bal Acc", "Macro Prec", "Macro Rec", "Macro F1", "ROC-AUC", "Wall Time"],
        # 1%
        ["ImageNet Transfer", "1%", "899", "0.8024", "0.7456", "0.7477", "0.7456", "0.7449", "0.9640", "53.7s"],
        ["SimCLR Fine-Tuning", "1%", "899", "0.7880", "0.7484", "0.7413", "0.7484", "0.7408", "0.9662", "52.2s"],
        ["SimCLR Linear Probe", "1%", "899", "0.7464", "0.6995", "0.6986", "0.6995", "0.6943", "0.9526", "36.9s"],
        ["Supervised Scratch", "1%", "899", "0.7375", "0.6944", "0.6978", "0.6944", "0.6815", "0.9540", "67.2s"],
        # 5%
        ["ImageNet Transfer", "5%", "4,499", "0.8728", "0.8434", "0.8349", "0.8434", "0.8375", "0.9867", "131.4s"],
        ["Supervised Scratch", "5%", "4,499", "0.8201", "0.7818", "0.7783", "0.7818", "0.7677", "0.9800", "133.0s"],
        ["SimCLR Fine-Tuning", "5%", "4,499", "0.8145", "0.7745", "0.7652", "0.7745", "0.7672", "0.9745", "132.8s"],
        ["SimCLR Linear Probe", "5%", "4,499", "0.7439", "0.7053", "0.6951", "0.7053", "0.6955", "0.9544", "57.6s"],
        # 10%
        ["ImageNet Transfer", "10%", "8,999", "0.8838", "0.8569", "0.8483", "0.8569", "0.8495", "0.9891", "231.1s"],
        ["Supervised Scratch", "10%", "8,999", "0.8467", "0.8096", "0.8047", "0.8096", "0.8008", "0.9824", "232.7s"],
        ["SimCLR Fine-Tuning", "10%", "8,999", "0.8295", "0.7914", "0.7828", "0.7914", "0.7846", "0.9771", "235.1s"],
        ["SimCLR Linear Probe", "10%", "8,999", "0.7462", "0.7100", "0.6979", "0.7100", "0.6997", "0.9555", "83.0s"],
        # 100%
        ["Supervised Scratch", "100%", "89,996", "0.9018", "0.8763", "0.8708", "0.8763", "0.8721", "0.9916", "2185.6s"],
        ["ImageNet Transfer", "100%", "89,996", "0.8923", "0.8628", "0.8582", "0.8628", "0.8587", "0.9901", "2023.5s"],
        ["SimCLR Fine-Tuning", "100%", "89,996", "0.8681", "0.8351", "0.8263", "0.8351", "0.8277", "0.9863", "2038.7s"],
        ["SimCLR Linear Probe", "100%", "89,996", "0.7526", "0.7134", "0.7045", "0.7134", "0.7055", "0.9571", "559.3s"],
    ]
    
    col_x = [0.06, 0.23, 0.30, 0.38, 0.45, 0.52, 0.59, 0.66, 0.74, 0.82, 0.94]
    row_y = 0.895
    row_h = 0.021
    
    ax.text(0.06, row_y + 0.015, "TABLE 1: Comprehensive Performance Evaluation on CRC-VAL-HE-7K External Test Cohort (N = 7,180)",
            fontsize=8.5, fontweight="bold", color=NAVY)
    
    for r_idx, row in enumerate(table_data):
        curr_y = row_y - (r_idx * row_h)
        if r_idx == 0:
            bg_color = NAVY
            text_color = "#FFFFFF"
            font_wt = "bold"
        elif r_idx in [1, 5, 9, 13]:
            bg_color = "#EDF2F7"
            text_color = DARK_SLATE
            font_wt = "bold" if "Scratch" in row[0] or "ImageNet" in row[0] else "normal"
        else:
            bg_color = "#FFFFFF" if (r_idx % 2 == 1) else "#F7FAFC"
            text_color = DARK_SLATE
            font_wt = "normal"
            
        rect = Rectangle((0.06, curr_y - 0.005), 0.88, row_h, facecolor=bg_color, edgecolor="#CBD5E0", lw=0.5)
        ax.add_patch(rect)
        
        for c_idx, cell in enumerate(row):
            ha_align = "left" if c_idx == 0 else "center"
            cx = col_x[c_idx] + 0.01 if c_idx == 0 else (col_x[c_idx] + col_x[c_idx+1]) / 2
            ax.text(cx, curr_y + 0.005, cell, fontsize=7.0 if r_idx > 0 else 7.2,
                    fontweight=font_wt, color=text_color, ha=ha_align, va="center")
            
    # Embed Label Efficiency Macro F1 Plot
    f1_img_path = os.path.join(PLOTS_DIR, "label_efficiency_macro_f1.png")
    if os.path.exists(f1_img_path):
        f1_img = mpimg.imread(f1_img_path)
        img_ax = fig.add_axes([0.08, 0.07, 0.84, 0.44])
        img_ax.imshow(f1_img)
        img_ax.axis("off")
        
    pdf.savefig(fig)
    plt.close(fig)


def build_page_5(pdf, total_pages=8):
    """Page 5: Model Comparison & Accuracy Trajectory Dynamics"""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    draw_header_footer(ax, 5, total_pages)
    
    ax.text(0.06, 0.925, "6. Model Comparison & Trajectory Dynamics", 
            fontsize=13, fontweight="bold", color=NAVY)
    
    # Embed Bar comparison plot
    bar_img_path = os.path.join(PLOTS_DIR, "model_comparison_bar.png")
    if os.path.exists(bar_img_path):
        bar_img = mpimg.imread(bar_img_path)
        img_ax1 = fig.add_axes([0.08, 0.54, 0.84, 0.36])
        img_ax1.imshow(bar_img)
        img_ax1.axis("off")
        
    # Embed Accuracy trajectory plot
    acc_img_path = os.path.join(PLOTS_DIR, "label_efficiency_accuracy.png")
    if os.path.exists(acc_img_path):
        acc_img = mpimg.imread(acc_img_path)
        img_ax2 = fig.add_axes([0.08, 0.07, 0.84, 0.43])
        img_ax2.imshow(acc_img)
        img_ax2.axis("off")
        
    pdf.savefig(fig)
    plt.close(fig)


def build_page_6(pdf, total_pages=8):
    """Page 6: Granular Per-Class Performance & Morphological Analysis"""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    draw_header_footer(ax, 6, total_pages)
    
    ax.text(0.06, 0.925, "7. Granular Per-Class Performance & Morphological Dynamics", 
            fontsize=13, fontweight="bold", color=NAVY)
    
    # Table Header & Data
    class_table = [
        ["ID", "Tissue Class Name", "Test N", "Scratch F1", "Linear Probe", "SimCLR FT", "ImageNet FT", "Gain (FT-Scratch)", "Primary Biological Characteristics"],
        ["0", "Adipose Tissue (ADI)", "718", "0.9162", "0.9547", "0.9496", "0.9538", "+0.0334", "Empty lipid vacuoles with thin peripheral rim membranes."],
        ["1", "Background (BACK)", "780", "0.9260", "0.9758", "0.9674", "0.9860", "+0.0414", "Clear glass slide substrate; absence of biological material."],
        ["2", "Debris (DEB)", "718", "0.4355", "0.5777", "0.5505", "0.4566", "+0.1150", "Amorphous necrotic tissue, dust, precipitate. Large SSL gain."],
        ["3", "Lymphocytes (LYM)", "828", "0.8201", "0.6904", "0.8416", "0.8869", "+0.0215", "Small round hyperchromatic basophilic nuclei with scant cytoplasm."],
        ["4", "Mucus (MUC)", "772", "0.7456", "0.6500", "0.7501", "0.8581", "+0.0045", "Extracellular basophilic pale-blue mucin pools."],
        ["5", "Smooth Muscle (MUS)", "800", "0.5414", "0.6056", "0.6761", "0.6452", "+0.1347", "Elongated eosinophilic spindle cells arranged in bundles."],
        ["6", "Normal Mucosa (NORM)", "780", "0.6566", "0.6889", "0.7359", "0.7013", "+0.0793", "Organized colonic crypts with mucin-secreting goblet cells."],
        ["7", "Cancer-Assoc. Stroma", "822", "0.3707", "0.3333", "0.4116", "0.4286", "+0.0409", "Desmoplastic collagenous stroma. Universally hardest class."],
        ["8", "Adenocarcinoma (TUM)", "962", "0.7219", "0.7727", "0.7842", "0.7878", "+0.0623", "Irregular malignant glandular architecture, marked nuclear atypia."]
    ]
    
    col_x = [0.06, 0.10, 0.28, 0.35, 0.43, 0.51, 0.59, 0.67, 0.78, 0.94]
    row_y = 0.895
    row_h = 0.026
    
    ax.text(0.06, row_y + 0.015, "TABLE 2: Granular Per-Class Performance and Morphological Profiles at 1% Labeled Data (N = 899)",
            fontsize=8.5, fontweight="bold", color=NAVY)
    
    for r_idx, row in enumerate(class_table):
        curr_y = row_y - (r_idx * row_h)
        if r_idx == 0:
            bg_color = NAVY
            text_color = "#FFFFFF"
            font_wt = "bold"
        elif r_idx in [3, 8]:  # Highlight Debris and Stroma
            bg_color = LIGHT_AMBER
            text_color = DARK_SLATE
            font_wt = "bold"
        else:
            bg_color = "#FFFFFF" if (r_idx % 2 == 1) else "#F7FAFC"
            text_color = DARK_SLATE
            font_wt = "normal"
            
        rect = Rectangle((0.06, curr_y - 0.007), 0.88, row_h, facecolor=bg_color, edgecolor="#CBD5E0", lw=0.5)
        ax.add_patch(rect)
        
        for c_idx, cell in enumerate(row):
            ha_align = "left" if c_idx in [1, 8] else "center"
            cx = col_x[c_idx] + 0.01 if c_idx in [1, 8] else (col_x[c_idx] + col_x[c_idx+1]) / 2
            ax.text(cx, curr_y + 0.005, cell, fontsize=6.8 if r_idx > 0 else 7.0,
                    fontweight=font_wt, color=text_color, ha=ha_align, va="center")
            
    # Biochemical Analysis Box
    bio_patch = FancyBboxPatch((0.06, 0.07), 0.88, 0.52, boxstyle="round,pad=0.012",
                               edgecolor=CARD_BORDER, facecolor="#FFFFFF", lw=1.0)
    ax.add_patch(bio_patch)
    
    ax.text(0.08, 0.565, "HISTOMORPHOLOGICAL BEHAVIOR & CLASS CONFUSION ANALYSIS", fontsize=9.5, fontweight="bold", color=NAVY)
    
    bio_text = (
        "1. Debris (Class 2) - Robustness to Structural Amorphism:\n"
        "   On necrotic debris, SimCLR Linear Probe (0.5777) and SimCLR Fine-Tuning (0.5505) outperform Supervised Scratch\n"
        "   (0.4355, +11.5% F1) and ImageNet Transfer (0.4566, +9.4% F1). Debris consists of amorphous necrosis, dust, and stain precipitate\n"
        "   lacking consistent structural borders. With only ~80 samples in training, supervised models overfit to specific artifact shapes.\n"
        "   Contrastive pretraining exposes the model to varied multi-scale crops and color perturbations without label supervision, learning\n"
        "   a broader visual envelope for unorganized textures.\n\n"
        "2. Cancer-Associated Stroma (Class 7) vs. Smooth Muscle (Class 5) - Desmoplastic Morphologic Overlap:\n"
        "   Class 7 is universally the most difficult class across all models (F1 = 0.33 to 0.43 at 1% labels). Desmoplastic stroma arises\n"
        "   when invasive carcinoma induces reactive collagen synthesis and myofibroblast proliferation. These spindle-shaped myofibroblasts\n"
        "   share eosinophilic staining and fascicular orientation with smooth muscle (Class 5). In 28x28 patches, spatial compression\n"
        "   blurs delicate collagen crimping and nuclear atypia, causing reciprocal false-positive classifications.\n\n"
        "3. Adipose (Class 0) and Background (Class 1) - High Morphological Separability:\n"
        "   Adipose tissue and clear slide background achieve >0.91 to >0.98 F1 even with only 1% labels across all models. Adipose consists\n"
        "   of large empty lipid voids, while background is optically clear glass. Both occupy distinct low-density regions that are easily\n"
        "   separated by linear decision boundaries.\n\n"
        "4. Colorectal Adenocarcinoma (Class 8) - Epithelial Atypia:\n"
        "   Malignant tumor nests achieve 0.7842 F1 with SimCLR Fine-Tuning at 1% labels (vs. 0.7219 for Scratch). SimCLR pretraining successfully\n"
        "   encodes cribriform gland formations, architectural back-to-back gland crowding, and hyperchromatic basophilic nuclear clustering."
    )
    ax.text(0.08, 0.54, bio_text, fontsize=7.5, color=DARK_SLATE, va="top", family="monospace", linespacing=1.28)
    
    pdf.savefig(fig)
    plt.close(fig)


def build_page_7(pdf, total_pages=8):
    """Page 7: Representation Geometry & Exploratory t-SNE Analysis"""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    draw_header_footer(ax, 7, total_pages)
    
    ax.text(0.06, 0.925, "8. Representation Geometry & Exploratory t-SNE Analysis", 
            fontsize=13, fontweight="bold", color=NAVY)
    
    # Embed t-SNE multi-panel image
    tsne_img_path = os.path.join(PLOTS_DIR, "tsne_representations.png")
    if os.path.exists(tsne_img_path):
        tsne_img = mpimg.imread(tsne_img_path)
        img_ax = fig.add_axes([0.08, 0.44, 0.84, 0.46])
        img_ax.imshow(tsne_img)
        img_ax.axis("off")
        
    # Cluster Validity Metrics Box
    geo_patch = FancyBboxPatch((0.06, 0.07), 0.88, 0.34, boxstyle="round,pad=0.012",
                               edgecolor=CARD_BORDER, facecolor="#F8FAFC", lw=1.0)
    ax.add_patch(geo_patch)
    
    ax.text(0.08, 0.385, "UNSUPERVISED CLUSTER VALIDITY METRICS (EXPLORATORY TEST SUBSET, N = 1,500)", fontsize=9.0, fontweight="bold", color=NAVY)
    
    geo_table = [
        ["Representation Space", "Silhouette Score (uparrow)", "Davies-Bouldin Index (downarrow)", "Geometric Manifold Interpretation"],
        ["Untrained ResNet-18 (Random)", "-0.0412", "4.8912", "Disorganized manifold; completely overlapping tissue distributions."],
        ["SimCLR SSL (Pretrained Only)", "+0.1162", "3.2115", "Discovers natural histological clusters with ZERO supervision."],
        ["ImageNet Transfer (10% Labels)", "+0.1415", "2.1569", "General visual primitives group tissues with moderate separation."],
        ["SimCLR Fine-Tuned (10% Labels)", "+0.1924", "2.0727", "In-domain SSL fine-tuning produces tighter clusters than ImageNet."],
        ["Supervised Scratch (10% Labels)", "+0.3197", "1.4801", "HIGHEST COMPACTION. Supervised cross-entropy directly forces separation."]
    ]
    
    g_col_x = [0.08, 0.34, 0.49, 0.65, 0.92]
    g_row_y = 0.355
    g_row_h = 0.024
    
    for r_idx, row in enumerate(geo_table):
        curr_y = g_row_y - (r_idx * g_row_h)
        bg_color = NAVY if r_idx == 0 else ("#EDF2F7" if r_idx == 5 else ("#FFFFFF" if r_idx % 2 == 1 else "#F7FAFC"))
        text_color = "#FFFFFF" if r_idx == 0 else DARK_SLATE
        font_wt = "bold" if r_idx in [0, 5] else "normal"
        
        rect = Rectangle((0.08, curr_y - 0.005), 0.84, g_row_h, facecolor=bg_color, edgecolor="#CBD5E0", lw=0.5)
        ax.add_patch(rect)
        
        for c_idx, cell in enumerate(row):
            ha = "left" if c_idx in [0, 3] else "center"
            cx = g_col_x[c_idx] + 0.01 if c_idx in [0, 3] else (g_col_x[c_idx] + g_col_x[c_idx+1]) / 2
            ax.text(cx, curr_y + 0.006, cell, fontsize=6.8 if r_idx > 0 else 7.0,
                    fontweight=font_wt, color=text_color, ha=ha, va="center")
            
    notes = (
        "Evaluation Protocol Distinction & Geometric Interpretation:\n"
        "  • Classification vs. Exploration: Table 1 evaluates all N = 7,180 test samples. This section evaluates an exploratory N = 1,500 subset.\n"
        "  • Unsupervised Cluster Formation: Contrastive pretraining shifts silhouette score from -0.0412 (random) to +0.1162 without labels.\n"
        "  • Supervised Margin Compaction: Cross-entropy loss forces intra-class compaction, yielding the highest silhouette score (0.3197).\n"
        "  • Qualitative Value: 2D t-SNE projections serve as qualitative exploratory visualizations of feature space organization."
    )
    ax.text(0.08, 0.18, notes, fontsize=7.2, color=DARK_SLATE, family="monospace", linespacing=1.25)
    
    pdf.savefig(fig)
    plt.close(fig)


def build_page_8(pdf, total_pages=8):
    """Page 8: Limitations, Clinical Disclaimer & Setup Instructions"""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    draw_header_footer(ax, 8, total_pages)
    
    ax.text(0.06, 0.925, "9. Limitations, Reproducibility & Setup Instructions", 
            fontsize=13, fontweight="bold", color=NAVY)
    
    # Limitations Card
    lim_patch = FancyBboxPatch((0.06, 0.61), 0.88, 0.29, boxstyle="round,pad=0.012",
                               edgecolor=CARD_BORDER, facecolor="#FFFFFF", lw=1.0)
    ax.add_patch(lim_patch)
    
    ax.text(0.08, 0.875, "LIMITATIONS & TECHNICAL CONSTRAINTS", fontsize=9.0, fontweight="bold", color=ALERT_RED)
    
    limitations_text = (
        "1. Single-Seed Pilot Scope: All 16 experimental conditions were evaluated under a single controlled seed (Seed 42). While sample\n"
        "   selection and batch shuffling were deterministic, this single-seed design does not quantify run-to-run stochastic variance or\n"
        "   establish formal statistical significance. Multi-seed replication (3-5 seeds) is required for definitive confidence bounds.\n"
        "2. Standardized Low Resolution (28x28 Pixels): The PathMNIST benchmark compresses high-power histology into 28x28 tiles,\n"
        "   attenuating subcellular cytological features (nuclear atypia, mitotic figures, chromatin margins) essential for clinical grading.\n"
        "3. Single Organ Domain: Pretraining and evaluation were limited to colorectal carcinoma resection specimens. Transferability across\n"
        "   diverse anatomical sites (e.g., breast, prostate, lung) and non-neoplastic pathologies requires broader multi-cohort validation.\n"
        "4. Instance Discrimination False Negatives: Standard SimCLR treats all distinct patches in a batch as negative pairs, penalizing\n"
        "   intra-class alignment between patches of the same histological class. Supervised contrastive learning (SupCon) could alleviate this."
    )
    ax.text(0.08, 0.855, limitations_text, fontsize=7.4, color=DARK_SLATE, va="top", family="monospace", linespacing=1.25)
    
    # Disclaimer Card
    disc_patch = FancyBboxPatch((0.06, 0.455), 0.88, 0.14, boxstyle="round,pad=0.012",
                                edgecolor=ALERT_RED, facecolor=LIGHT_AMBER, lw=1.0)
    ax.add_patch(disc_patch)
    
    ax.text(0.08, 0.57, "NON-CLINICAL USE DISCLAIMER", fontsize=9.0, fontweight="bold", color=ALERT_RED)
    disc_text = (
        "This project represents an academic machine learning engineering investigation into representation learning mechanics and\n"
        "label-efficiency dynamics on standardized benchmark datasets. The models, representations, checkpoints, and evaluations\n"
        "documented herein HAVE NOT BEEN CLEARED, APPROVED, OR VALIDATED FOR CLINICAL DIAGNOSTIC USE, PROGNOSTIC DECISION-MAKING,\n"
        "OR PATIENT MANAGEMENT. They must not be deployed in clinical practice or used to guide real-world patient care."
    )
    ax.text(0.08, 0.55, disc_text, fontsize=7.2, color="#7B341E", va="top", family="monospace", linespacing=1.25)
    
    # Setup Instructions & References Card
    ref_patch = FancyBboxPatch((0.06, 0.07), 0.88, 0.37, boxstyle="round,pad=0.012",
                               edgecolor=CARD_BORDER, facecolor="#F8FAFC", lw=1.0)
    ax.add_patch(ref_patch)
    
    ax.text(0.08, 0.415, "SETUP COMMANDS & SELECTED REFERENCES", fontsize=9.0, fontweight="bold", color=NAVY)
    
    setup_and_refs = (
        "Quickstart Reproduction Commands:\n"
        "  • Run Unit Tests:           python3 -m unittest discover -s tests  (22/22 Passing)\n"
        "  • SimCLR Pretraining:       python3 src/pretrain_simclr.py         (15 epochs, ~24.5 min)\n"
        "  • Run 16 Experiments:       python3 scripts/run_experiments.py     (16 conditions, ~2.2 hrs)\n"
        "  • Summarize Results:        python3 scripts/summarize_results.py   (Updates Markdown & LaTeX tables)\n"
        "  • Recompile PDF Report:     python3 scripts/generate_pdf_report.py (Generates technical_project_report.pdf)\n\n"
        "Key References:\n"
        "  [1] J. N. Kather et al., 'Multi-class texture analysis in colorectal cancer histology,' Scientific Reports, 2016.\n"
        "  [2] T. Chen et al., 'A simple framework for contrastive learning of visual representations,' ICML, PMLR, 2020.\n"
        "  [3] J. Yang et al., 'MedMNIST v2 - A large-scale lightweight benchmark for 2D/3D biomedical image classification,' 2023.\n"
        "  [4] K. He et al., 'Deep residual learning for image recognition,' CVPR, pp. 770-778, 2016.\n"
        "  [5] I. Loshchilov & F. Hutter, 'Decoupled weight decay regularization,' ICLR, 2019.\n"
        "  [6] J. Deng et al., 'ImageNet: A large-scale hierarchical image database,' CVPR, 2009.\n"
        "  [7] O. Ciga et al., 'Self-supervised contrastive learning for digital histopathology,' MLWA, 2022."
    )
    ax.text(0.08, 0.395, setup_and_refs, fontsize=7.0, color=DARK_SLATE, va="top", family="monospace", linespacing=1.24)
    
    pdf.savefig(fig)
    plt.close(fig)


def main():
    print(f"Generating Technical Project Report PDF: {PDF_OUTPUT_PATH}...")
    with PdfPages(PDF_OUTPUT_PATH) as pdf:
        build_page_1(pdf)
        build_page_2(pdf)
        build_page_3(pdf)
        build_page_4(pdf)
        build_page_5(pdf)
        build_page_6(pdf)
        build_page_7(pdf)
        build_page_8(pdf)
        
    file_size_kb = os.path.getsize(PDF_OUTPUT_PATH) / 1024
    print(f"Technical Project Report PDF successfully generated! Size: {file_size_kb:.1f} KB ({PDF_OUTPUT_PATH})")

if __name__ == "__main__":
    main()
