# Technical Project Report: Self-Supervised Representation Learning for Histopathological Image Classification Under Limited Label Availability

**Project Title:** Self-Supervised Representation Learning for Histopathological Image Classification Under Limited Label Availability  
**Domain:** Deep Learning, Computer Vision, Computational Pathology, Medical Representation Learning  
**Target Profile:** B.Tech AI/ML Engineering Portfolio & MS Application Technical Document  
**Date:** November 2026  
**Primary Dataset:** MedMNIST v2 PathMNIST (NCT-CRC-HE-100K Training Split & CRC-VAL-HE-7K External Test Cohort)  
**Hardware & Framework:** Apple Silicon M-series (MPS acceleration), PyTorch 2.6.0, float32 precision  

---

## 1. Project Overview

In deep learning for digital histopathology, obtaining high-quality ground-truth annotations requires extensive manual labor by trained medical specialists. In real-world engineering deployments—such as adapting a model to a new clinical laboratory, hospital site, or rare diagnostic category—labeled examples are often strictly limited. 

This engineering project designs, implements, and benchmarks a complete PyTorch-based representation learning system. The objective is to evaluate whether domain-specific, self-supervised contrastive pretraining using **SimCLR** (Simple Framework for Contrastive Learning of Visual Representations) on unlabeled colorectal histology patches can yield better downstream classification representations than:
1. **Supervised learning trained from scratch** with random weight initialization.
2. **Standard transfer learning** using off-the-shelf ImageNet-1k pretrained weights.

The evaluation is conducted across four controlled label availability regimes:
- **1% Labeled Data:** $N = 899$ samples ($\sim 80 - 129$ samples per class)
- **5% Labeled Data:** $N = 4,499$ samples ($\sim 400 - 648$ samples per class)
- **10% Labeled Data:** $N = 8,999$ samples ($\sim 800 - 1,296$ samples per class)
- **100% Labeled Data:** $N = 89,996$ samples (complete training corpus)

All four methods utilize an identical ResNet-18 convolutional backbone and are evaluated on the untouched official external test cohort **CRC-VAL-HE-7K** ($N = 7,180$).

### Key Empirical Observations
1. **Performance in the 1% Label Regime:** ImageNet transfer achieved the highest Macro F1 (**0.7449**), followed by SimCLR fine-tuning (**0.7408**), SimCLR linear probe (**0.6943**), and Supervised Scratch (**0.6815**). SimCLR fine-tuning outpaced Supervised Scratch by **+5.05% Test Accuracy** (78.80% vs. 73.75%) and **+0.0593 Macro F1**, recording the highest discriminative confidence across all four evaluated models (**Macro ROC-AUC: 0.9662**).
2. **Transfer Learning in Low-to-Medium Regimes:** ImageNet transfer achieved the highest Macro F1 across all limited-label settings: 1% (0.7449, followed by SimCLR fine-tuning at 0.7408), 5% (0.8375), and 10% (0.8495) labels.
3. **High-Data Crossover at 100% Labels:** When all 89,996 labeled training samples are available, Supervised Scratch achieved the highest performance across all paradigms (**90.18% Test Accuracy, 0.8721 Macro F1, 0.9916 ROC-AUC**), outperforming ImageNet Transfer (0.8587 F1) and SimCLR Fine-Tuning (0.8277 F1).
4. **Observed Linear Probe Profile:** SimCLR linear probe scores remained within a narrow band across sample sizes (0.6943 at 1% to 0.7055 at 100%, span: +0.0112 Macro F1), while full fine-tuning ranged from 0.7408 at 1% to 0.8277 at 100% (span: +0.0869 Macro F1).

### Core Engineering Deliverables
- **Modular Codebase:** Clean, production-structured repository (`src/`, `configs/`, `scripts/`, `tests/`) implementing datasets, domain-specific augmentations, contrastive loss, training pipelines, and evaluation metrics.
- **Automated Experimentation Pipeline:** Reproducible runner executing 16 distinct downstream experimental conditions with automated early stopping, model checkpointing, and metric logging.
- **Audited Metric Integrity:** Fully verified metrics pipeline with zero discrepancies between saved weights and logged performance tables.
- **Automated Report Generation:** Standalone Python script compiling metrics, tables, and high-resolution plots into a multi-page PDF report.

---

## 2. Problem Statement and Motivation

### 2.1 The Annotation Bottleneck in Computational Pathology
Modern digital pathology utilizes whole-slide scanners that digitize tissue glass slides into gigapixel images containing tens of thousands of microscopic tissue fields. While unlabeled slide scans are abundant in clinical archives, extracting supervised value from them is constrained by:
1. **Expert Dependency:** Annotating cellular boundaries, tissue patterns, and tumor margins requires board-certified pathologists whose time is primarily dedicated to patient diagnostics.
2. **Label Scarcity in Distribution Shifts:** When deploying models across different slide scanner models, tissue processing protocols, or anatomical sites, existing supervised models degrade, requiring re-annotation of localized datasets where only dozens or hundreds of labeled tiles may be feasible to collect.

### 2.2 Engineering Motivation
In standard industry workflows, practitioners often rely on transfer learning using convolutional networks pretrained on ImageNet (natural photographs). However, natural photographs (such as animals, vehicles, and objects) differ fundamentally in visual properties from microscopic histology (which consists of staining textures, cellular chromatin density, and tissue architectural morphology).

This project investigates whether pretraining a contrastive encoder directly on unlabeled, in-domain histology tiles provides an engineering advantage over ImageNet transfer learning or random initialization when labeled training subsets are small.

---

## 3. Dataset Description

### 3.1 PathMNIST & CRC-VAL-HE-7K Cohort Protocol
The dataset utilized is **PathMNIST**, part of the MedMNIST v2 standardized medical imaging benchmark suite:
- **Pretraining & Training Partition:** Derived from the **NCT-CRC-HE-100K** cohort, consisting of formal formalin-fixed paraffin-embedded (FFPE) tissue resection specimens from human colorectal carcinoma:
  - Training Split: $N = 89,996$ images
  - Validation Split: $N = 10,004$ images
- **External Test Cohort:** Evaluated strictly on **CRC-VAL-HE-7K** ($N = 7,180$ images), which comprises independent colorectal cancer tissue specimens collected from separate patient populations and digitized independently.

### 3.2 Tissue Class Taxonomy
The benchmark evaluates multiclass classification across nine distinct colorectal tissue categories:

| Class ID | Tissue Label | Full Name | Clinical / Morphological Description |
| :---: | :--- | :--- | :--- |
| **0** | `ADI` | Adipose Tissue | Normal lipid-storing lobules; empty lipid vacuoles with thin peripheral membranes. |
| **1** | `BACK` | Background | Optically clear slide glass substrate; absence of biological material. |
| **2** | `DEB` | Debris | Amorphous cellular necrosis, hemorrhage, dust particles, and stain precipitate. |
| **3** | `LYM` | Lymphocytes | Dense chronic immune cell infiltrates; small round basophilic nuclei with scant cytoplasm. |
| **4** | `MUC` | Mucus | Extracellular mucin pools; pale bluish/gray basophilic material. |
| **5** | `MUS` | Smooth Muscle | Intestinal muscularis propria; elongated eosinophilic spindle cells arranged in bundles. |
| **6** | `NORM` | Normal Mucosa | Benign colonic crypts; organized tubular architecture with goblet cells. |
| **7** | `STR` | Cancer-Associated Stroma | Desmoplastic fibroblastic proliferation; collagen-rich reactive stroma. |
| **8** | `TUM` | Colorectal Adenocarcinoma | Malignant epithelial structures; crowded atypical glands with hyperchromatic nuclei. |

### 3.3 Standardized $28 \times 28$ Image Resolution & Technical Trade-Offs
The standard PathMNIST benchmark provides images resized to $28 \times 28 \times 3$ pixels. This engineering choice entails specific trade-offs:
- **Preserved Information:** Global tissue texture, macroscopic architectural arrangement (e.g., crypt structure vs. sheets of cells), overall stain intensity, and broad color balance (basophilic vs. eosinophilic).
- **Attenuated Information:** Fine subcellular cytologic details (such as nuclear atypia, mitotic spindles, chromatin margin distribution, and individual nucleoli) are compressed. In clinical practice, diagnostic grading typically relies on 20× or 40× magnification ($0.25 - 0.50\ \mu\text{m/pixel}$). Downsampled $28 \times 28$ patches represent low-resolution abstractions.
- **Engineering Advantage:** The $28 \times 28$ resolution allows full-scale self-supervised contrastive pretraining on $\sim 90,000$ images within a feasible computational budget on edge hardware (Apple Silicon MPS), making it an effective platform for controlled methodology benchmarking.

### 3.4 Data-Split Isolation Checks
To ensure strict data-split isolation and experimental integrity:
1. **Unlabeled Pretraining Isolation:** SimCLR pretraining utilized exclusively the $N = 89,996$ training split. All class labels were removed during this phase.
2. **Deterministic Stratified Subsetting:** Subsets for 1%, 5%, 10%, and 100% were drawn exclusively from the training partition using `StratifiedShuffleSplit` with a fixed seed (Seed 42). All four downstream models received the exact identical sample indices for any given data fraction.
3. **Independent Evaluation:** The validation partition ($N = 10,004$) was used exclusively for tracking validation loss and triggering early stopping. The external test cohort ($N = 7,180$) was evaluated exactly once per trained model after training terminated. The external test set was never used for model selection, thresholding, or hyperparameter tuning.

---

## 4. System Architecture

The project implements a modular neural pipeline consisting of a convolutional feature extractor, a non-linear projection head for contrastive pretraining, and a downstream classification head.

```
+--------------------------------------------------------------------------------------------------+
|                                    System Architecture Pipeline                                  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
| [Input Patch: x in R^(3x28x28)]                                                                 |
|         │                                                                                        |
|         ▼                                                                                        |
| [Feature Backbone: ResNet-18 f(x)]                                                              |
|   ├─ Conv1 (7x7, stride 2) + MaxPool (3x3)                                                       |
|   ├─ Layer 1 (2 residual blocks, 64 channels)                                                    |
|   ├─ Layer 2 (2 residual blocks, 128 channels)                                                   |
|   ├─ Layer 3 (2 residual blocks, 256 channels)                                                   |
|   ├─ Layer 4 (2 residual blocks, 512 channels)                                                   |
|   └─ AdaptiveAvgPool2d((1, 1))                                                                    |
|         │                                                                                        |
|         ▼ Penultimate Feature Vector [h in R^512]                                                |
|         ├────────────────────────────────────────┬────────────────────────────────────────┐      |
|         │ (Pretraining Phase)                    │ (Downstream Evaluation Phase)          │      |
|         ▼                                        ▼                                        │      |
| [MLP Projection Head: g(h)]              [Linear Classification Head]                     │      |
|   ├─ Linear(512 -> 512)                    └─ Linear(512 -> 9 classes)                    │      |
|   ├─ ReLU                                        │                                        │      |
|   ├─ Linear(512 -> 128)                          ▼                                        │      |
|   └─ L2 Normalization                    [Class Logits in R^9]                            │      |
|         │                                        │                                        │      |
|         ▼ Latent Vector [z in S^127]             ▼ Softmax + Cross-Entropy Loss           │      |
|   [NT-Xent Contrastive Loss]             (Trained via Linear Probe or Full FT)            │      |
|   (tau = 0.5, 512 augmented views)                                                        │      |
|                                                                                           │      |
+-------------------------------------------------------------------------------------------+
```

### 4.1 Feature Backbone
The feature extractor is a standard ResNet-18 architecture with 11,176,512 parameters. Given an input $x \in \mathbb{R}^{3 \times 28 \times 28}$, the network outputs a 512-dimensional feature embedding $h = f(x) \in \mathbb{R}^{512}$.

### 4.2 Non-Linear Projection Head (SimCLR Pretraining)
During self-supervised pretraining, an auxiliary multi-layer perceptron (MLP) projection head $g(\cdot)$ maps $h$ into a low-dimensional hyperspherical projection space $z \in \mathbb{R}^{128}$:
$$z = g(h) = W^{(2)} \, \text{ReLU}\left(W^{(1)} h + b^{(1)}\right) + b^{(2)}$$
The projection vector is $\ell_2$-normalized: $\tilde{z} = z / \|z\|_2 \in \mathbb{S}^{127}$. After self-supervised pretraining finishes, this projection head is discarded, and downstream models attach directly to $h$.

### 4.3 Classification Head
For downstream classification, a linear layer ($W \in \mathbb{R}^{9 \times 512}, b \in \mathbb{R}^9$) maps representation $h$ to class logits $y \in \mathbb{R}^9$.

---

## 5. Methodology

The project systematically benchmarks four distinct learning paradigms across four label availability regimes (16 total experimental runs):

```
+-----------------------------------------------------------------------------------+
|                            Four-Method Comparison Setup                           |
+--------------------------+----------------------------+---------------------------+
| Method                   | Feature Encoder Backbone   | Classification Head       |
+--------------------------+----------------------------+---------------------------+
| 1. Supervised Scratch    | Random Init (He Normal)    | Linear (Trained, lr=1e-3) |
|                          | End-to-end (lr=1e-3)       |                           |
| 2. ImageNet Transfer     | Torchvision ImageNet-1k    | Linear (Trained, lr=1e-3) |
|                          | Fine-tuned (lr=1e-4)       |                           |
| 3. SimCLR Linear Probe   | SimCLR SSL Pretrained      | Linear (Trained, lr=1e-3) |
|                          | STRICTLY FROZEN            |                           |
| 4. SimCLR Fine-Tuning    | SimCLR SSL Pretrained      | Linear (Trained, lr=1e-3) |
|                          | Fine-tuned (lr=1e-4)       |                           |
+--------------------------+----------------------------+---------------------------+
```

### 5.1 Supervised Learning from Scratch
The standard baseline where all ResNet-18 convolutional kernels and the classification head are initialized randomly (using He normal initialization) and optimized end-to-end using standard cross-entropy loss:
$$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^C y_c \log \hat{y}_c$$
Learning rate: $\eta = 10^{-3}$.

### 5.2 ImageNet Transfer Learning
Weights are initialized from torchvision's official `ResNet18_Weights.IMAGENET1K_V1` (trained on 1,281,167 natural photographs across 1,000 object categories). The model is fine-tuned end-to-end on the histology training subset using differential learning rates:
- Pretrained backbone: $\eta_{\text{backbone}} = 10^{-4}$ (lower rate to preserve learned visual filters)
- Classification head: $\eta_{\text{head}} = 10^{-3}$ (higher rate for rapid alignment)

### 5.3 SimCLR Contrastive Self-Supervised Learning
SimCLR trains the network by maximizing agreement between differently augmented views of the same image while repelling views from distinct images:
1. For each image in a mini-batch of size $N_b$, two stochastic augmentations are sampled from policy $\mathcal{T}$, yielding a batch of $2N_b$ augmented views.
2. Views are passed through $f(\cdot)$ and $g(\cdot)$ to produce normalized embeddings $\tilde{z}$.
3. The Normalized Temperature-Scaled Cross Entropy (NT-Xent) loss for positive pair $(i, j)$ is computed:
   $$\ell_{i, j} = -\log \frac{\exp\left(\text{sim}(z_i, z_j) / \tau\right)}{\sum_{k=1}^{2N_b} \mathbb{I}_{[k \neq i]} \exp\left(\text{sim}(z_i, z_k) / \tau\right)}$$
   where $\text{sim}(u, v) = u^\top v / (\|u\|_2 \|v\|_2)$ is cosine similarity, $\tau = 0.5$ is the temperature hyperparameter, and $\mathbb{I}_{[k \neq i]} \in \{0, 1\}$ indicator function masks self-comparisons.

### 5.4 Downstream Adaptation: Linear Probe vs. Full Fine-Tuning
- **SimCLR Linear Probe:** The contrastive ResNet-18 backbone is completely frozen (`requires_grad = False`). Only the 4,617 parameters of the final linear layer are optimized ($\eta = 10^{-3}$). This tests the linear separability of the frozen self-supervised representation space.
- **SimCLR Full Fine-Tuning:** The entire model is unfrozen and optimized end-to-end using differential learning rates ($\eta_{\text{backbone}} = 10^{-4}, \eta_{\text{head}} = 10^{-3}$), allowing convolutional filters to adapt to class-specific boundaries.

---

## 6. Implementation Details

### 6.1 Domain-Specific Histopathology Augmentation Strategy
Data augmentations define the invariances learned by a contrastive model. In natural image benchmarks, converting images to random grayscale is standard practice to prevent models from learning trivial color histogram shortcuts.

However, in histopathology:
- Tissue sections are stained with **Hematoxylin** (which binds nucleic acids in cell nuclei, imparting a blue/purple color) and **Eosin** (which binds proteins in cytoplasm and collagen, imparting a pink/red color).
- Converting an H&E image to grayscale removes the critical spectral distinction between basophilic nuclei and eosinophilic cytoplasm.
- **Engineering Decision:** Random grayscale was strictly excluded from the augmentation policy.

The implemented augmentation pipeline includes:
1. **Random Resized Crop:** Scale range $[0.64, 1.0]$, aspect ratio $[0.8, 1.25]$, resized to $28 \times 28$.
2. **Random Horizontal & Vertical Flips:** Probability $p = 0.5$ each (histology tiles have no canonical top-to-bottom orientation).
3. **Random Rotation:** Affine rotation $\theta \in [-180^\circ, +180^\circ]$ to enforce rotational invariance.
4. **Color Jitter (Approximation of Color Variation):** Color jitter was applied as an algorithmic approximation of color and intensity variation across patches rather than a validated model of histological stain preservation. Bounded tolerances were chosen to avoid extreme hue inversions:
   - Brightness: $\pm 0.20$
   - Contrast: $\pm 0.20$
   - Saturation: $\pm 0.15$
   - Hue: $\pm 0.05$ (bounded to preserve general purple vs. pink distinction)
5. **Gaussian Blur:** Kernel size $3 \times 3$, $\sigma \in [0.1, 1.0]$ with probability $p = 0.2$.
6. **Channel Normalization:** Standardized using PathMNIST population statistics ($\mu = [0.7405, 0.5330, 0.7058]$, $\sigma = [0.1237, 0.1768, 0.1244]$).

### 6.2 Optimization Pipeline & Learning-Rate Scheduling
Across all paradigms, models were optimized using AdamW with decoupled weight decay $\lambda = 10^{-4}$.
- **Pretraining Configuration:** Batch size $N_b = 256$ (512 augmented views per step), base learning rate $\eta = 5 \times 10^{-4}$, 15 epochs over the 89,996 images.
- **Downstream Optimization Schedule:** Batch size $N_b = 64$, 15 epochs per condition.
- **Warmup + Cosine Decay:** Linear warmup for the initial 2 epochs up to base learning rate, followed by cosine annealing decay down to a minimum learning rate of $\eta_{\text{min}} = 10^{-6}$:
  $$\eta_t = \eta_{\text{min}} + \frac{1}{2}\left(\eta_{\text{base}} - \eta_{\text{min}}\right)\left(1 + \cos\left(\frac{t - T_{\text{warm}}}{T_{\text{max}} - T_{\text{warm}}} \pi\right)\right), \quad t > T_{\text{warm}}$$

### 6.3 Technical Challenges & Engineering Solutions
During implementation and benchmarking, several edge cases were identified and resolved:
1. **NT-Xent Numerical Stability:** Masking the diagonal in the similarity matrix with $-\infty$ produced floating-point underflow on GPU/MPS accelerators. This was resolved by setting the diagonal to a finite negative offset of $-10^4$. Since $\exp(-10^4) = 0.0$ to full floating-point precision, self-similarities are eliminated without numerical instability.
2. **Batch Normalization Micro-Batches:** In the 1% label regime ($N = 899$), a batch size of 64 yields 14 full batches and a remainder batch with only 3 samples. Passing 3 samples through BatchNorm layers distorted running mean and variance estimates. This was resolved by dynamically enabling `drop_last` when the subset length exceeds the batch size.
3. **MPS Synchronization Race Conditions:** Unpinned asynchronous GPU memory transfers (`non_blocking=True`) on Apple Silicon unified memory occasionally resulted in uninitialized reads during metrics evaluation. This was resolved by utilizing explicit barrier synchronization without unpinned asynchronous flags.
4. **Metric Aggregation Transparency:** In the results summary script, grouping single-seed runs produced `NaN` standard deviation values, which pandas filled with `0.0000`. The documentation was updated to clearly disclose that results reflect a single-seed pilot benchmark rather than multi-seed variance.

---

## 7. Experimental Setup

### 7.1 Hardware Platform & Environment
- **Platform:** Apple Silicon M-series (MPS acceleration)
- **Framework:** PyTorch 2.6.0, Torchvision 0.21.0
- **Arithmetic Precision:** Single-precision 32-bit floating-point (float32)
- **Host System:** macOS (Darwin 25.4.0)

### 7.2 Experimental Protocol & Scope Disclosure
All 16 downstream conditions were trained and evaluated under **deterministic Seed 42**:
- Pretraining: 1 run $\times$ 15 epochs on 89,996 images.
- Downstream: 4 methods $\times$ 4 label fractions = 16 conditions $\times$ 15 epochs.
- **Scope Disclosure:** All reported results represent a controlled, single-seed pilot benchmark. The results do not claim multi-seed statistical confidence intervals.

---

## 8. Results and Model Comparison

### 8.1 SimCLR Pretraining Convergence
During 15 epochs of self-supervised pretraining on the 89,996 unlabeled training images, the NT-Xent loss exhibited steady, monotonic convergence:
- **Epoch 1:** $\mathcal{L}_{\text{NT-Xent}} = 3.5252$ (Wall time: 104.2s)
- **Epoch 5:** $\mathcal{L}_{\text{NT-Xent}} = 1.1218$ (Wall time: 96.5s)
- **Epoch 10:** $\mathcal{L}_{\text{NT-Xent}} = 0.7812$ (Wall time: 97.1s)
- **Epoch 15:** $\mathcal{L}_{\text{NT-Xent}} = \mathbf{0.6466}$ (Wall time: 98.4s)

Total pretraining wall-clock time was **24.5 minutes** on Apple Silicon MPS. The saved encoder checkpoint (`checkpoints/simclr_encoder_best.pth`) is 43.1 MB.

### 8.2 Master Benchmark Performance Table
Table 1 presents the complete empirical evaluation across all 16 experimental conditions on the external test cohort CRC-VAL-HE-7K ($N = 7,180$). All metrics represent exact, audited evaluations under Seed 42.

#### Table 1: Downstream Classification Performance on External Test Cohort (CRC-VAL-HE-7K, N = 7,180)

| Method | Labeled Data (%) | N Samples | Test Acc | Balanced Acc | Macro Prec | Macro Rec | Macro F1-Score | Macro ROC-AUC | Wall Time (s) | Checkpoint Name |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ImageNet Transfer** | **1%** | 899 | **0.8024** | 0.7456 | 0.7477 | 0.7456 | **0.7449** | 0.9640 | 53.7s | `imagenet_transfer_prop001_seed42_best.pth` |
| **SimCLR Fine-Tuning** | **1%** | 899 | 0.7880 | **0.7484** | 0.7413 | **0.7484** | **0.7408** | **0.9662** | 52.2s | `simclr_fine-tuning_prop001_seed42_best.pth` |
| **SimCLR Linear Probe** | **1%** | 899 | 0.7464 | 0.6995 | 0.6986 | 0.6995 | **0.6943** | 0.9526 | **36.9s** | `simclr_linear_probe_prop001_seed42_best.pth` |
| **Supervised Scratch** | **1%** | 899 | 0.7375 | 0.6944 | 0.6978 | 0.6944 | **0.6815** | 0.9540 | 67.2s | `supervised_scratch_prop001_seed42_best.pth` |
| | | | | | | | | | | |
| **ImageNet Transfer** | **5%** | 4,499 | **0.8728** | **0.8434** | 0.8349 | **0.8434** | **0.8375** | **0.9867** | 131.4s | `imagenet_transfer_prop005_seed42_best.pth` |
| **Supervised Scratch** | **5%** | 4,499 | 0.8201 | 0.7818 | 0.7783 | 0.7818 | **0.7677** | 0.9800 | 133.0s | `supervised_scratch_prop005_seed42_best.pth` |
| **SimCLR Fine-Tuning** | **5%** | 4,499 | 0.8145 | 0.7745 | 0.7652 | 0.7745 | **0.7672** | 0.9745 | 132.8s | `simclr_fine-tuning_prop005_seed42_best.pth` |
| **SimCLR Linear Probe** | **5%** | 4,499 | 0.7439 | 0.7053 | 0.6951 | 0.7053 | **0.6955** | 0.9544 | **57.6s** | `simclr_linear_probe_prop005_seed42_best.pth` |
| | | | | | | | | | | |
| **ImageNet Transfer** | **10%** | 8,999 | **0.8838** | **0.8569** | 0.8483 | **0.8569** | **0.8495** | **0.9891** | 231.1s | `imagenet_transfer_prop010_seed42_best.pth` |
| **Supervised Scratch** | **10%** | 8,999 | 0.8467 | 0.8096 | 0.8047 | 0.8096 | **0.8008** | 0.9824 | 232.7s | `supervised_scratch_prop010_seed42_best.pth` |
| **SimCLR Fine-Tuning** | **10%** | 8,999 | 0.8295 | 0.7914 | 0.7828 | 0.7914 | **0.7846** | 0.9771 | 235.1s | `simclr_fine-tuning_prop010_seed42_best.pth` |
| **SimCLR Linear Probe** | **10%** | 8,999 | 0.7462 | 0.7100 | 0.6979 | 0.7100 | **0.6997** | 0.9555 | **83.0s** | `simclr_linear_probe_prop010_seed42_best.pth` |
| | | | | | | | | | | |
| **Supervised Scratch** | **100%** | 89,996 | **0.9018** | **0.8763** | 0.8708 | **0.8763** | **0.8721** | **0.9916** | 2185.6s | `supervised_scratch_prop100_seed42_best.pth` |
| **ImageNet Transfer** | **100%** | 89,996 | 0.8923 | 0.8628 | 0.8582 | 0.8628 | **0.8587** | 0.9901 | 2023.5s | `imagenet_transfer_prop100_seed42_best.pth` |
| **SimCLR Fine-Tuning** | **100%** | 89,996 | 0.8681 | 0.8351 | 0.8263 | 0.8351 | **0.8277** | 0.9863 | 2038.7s | `simclr_fine-tuning_prop100_seed42_best.pth` |
| **SimCLR Linear Probe** | **100%** | 89,996 | 0.7526 | 0.7134 | 0.7045 | 0.7134 | **0.7055** | 0.9571 | **559.3s** | `simclr_linear_probe_prop100_seed42_best.pth` |

---

## 9. Visualizations and Analysis

### 9.1 Label Efficiency Trajectories
Visualizations of Macro F1 and Accuracy across sample counts illustrate distinct behavioral profiles:
- **Macro F1 Trajectory (`results/plots/label_efficiency_macro_f1.png`):**
  - In the 1% label regime ($N = 899$), ImageNet transfer achieved the highest Macro F1 (0.7449), followed by SimCLR fine-tuning (0.7408), with both substantially outperforming Supervised Scratch (0.6815).
  - As labeled samples increase to 5% ($N = 4,499$), Supervised Scratch improves rapidly (0.7677), matching SimCLR Fine-Tuning (0.7672), while ImageNet Transfer remains highest (0.8375).
  - At 100% labels ($N = 89,996$), Supervised Scratch achieves the highest Macro F1 (0.8721), surpassing ImageNet Transfer (0.8587) and SimCLR Fine-Tuning (0.8277).
- **Observed Linear Probe Trajectory:** Across the evaluated label proportions, SimCLR Linear Probe results remain within a narrow range:
  - 1% ($N=899$): 0.6943 Macro F1 (Test Accuracy: 74.64%)
  - 5% ($N=4,499$): 0.6955 Macro F1 (Test Accuracy: 74.39%)
  - 10% ($N=8,999$): 0.6997 Macro F1 (Test Accuracy: 74.62%)
  - 100% ($N=89,996$): 0.7055 Macro F1 (Test Accuracy: 75.26%)  
  Across this 100-fold increase in labeled training data, the observed difference in Macro F1 was +0.0112 (and +0.62% in Test Accuracy). In comparison, SimCLR full fine-tuning showed a wider range of scores across the same sample sizes (0.7408 to 0.8277 Macro F1, a difference of +0.0869).

### 9.2 Granular Per-Class Performance (1% Labels)
Table 2 details class-specific F1 scores on the external test set when models are trained with only 1% labeled data ($N = 899$).

#### Table 2: Granular Per-Class F1-Scores and Sample Distribution at 1% Labeled Data ($N = 899$)

| ID | Tissue Class | Test N | Scratch F1 | SimCLR Probe F1 | SimCLR FT F1 | ImageNet F1 | $\Delta(\text{FT} - \text{Scratch})$ | Primary Morphological Characteristics |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Adipose (ADI) | 718 | 0.9162 | **0.9547** | 0.9496 | 0.9538 | +0.0334 | Distended empty lipid vacuoles with narrow peripheral membranes. |
| **1** | Background (BACK) | 780 | 0.9260 | 0.9758 | 0.9674 | **0.9860** | +0.0414 | Clear glass slide substrate; absence of biological material. |
| **2** | **Debris (DEB)** | 718 | 0.4355 | **0.5777** | 0.5505 | 0.4566 | **+0.1150** | Amorphous necrotic tissue, dust, precipitate. Large SSL gain. |
| **3** | Lymphocytes (LYM) | 828 | 0.8201 | 0.6904 | 0.8416 | **0.8869** | +0.0215 | Small, circular, hyperchromatic basophilic nuclei with scant cytoplasm. |
| **4** | Mucus (MUC) | 772 | 0.7456 | 0.6500 | 0.7501 | **0.8581** | +0.0045 | Extracellular, pale-bluish basophilic mucin pools. |
| **5** | Smooth Muscle (MUS) | 800 | 0.5414 | 0.6056 | **0.6761** | 0.6452 | **+0.1347** | Spindle-shaped cells arranged in fascicles with eosinophilic cytoplasm. |
| **6** | Normal Mucosa (NORM) | 780 | 0.6566 | 0.6889 | **0.7359** | 0.7013 | **+0.0793** | Organized colonic tubular crypts with mucin-secreting goblet cells. |
| **7** | **Cancer-Assoc. Stroma (STR)** | 822 | 0.3707 | 0.3333 | 0.4116 | **0.4286** | +0.0409 | Desmoplastic fibroblastic proliferation; extensive morphological overlap. |
| **8** | Colorectal Adenocarcinoma (TUM) | 962 | 0.7219 | 0.7727 | 0.7842 | **0.7878** | +0.0623 | Irregular glandular architecture, marked nuclear atypia, pleomorphism. |

#### Key Morphological Observations
1. **Necrotic Debris (Class 2):** SimCLR Linear Probe (0.5777) and Fine-Tuning (0.5505) noticeably exceed Supervised Scratch (0.4355) and ImageNet (0.4566). With only $\sim 80$ debris training samples, supervised models easily overfit to specific artifact shapes. Contrastive pretraining exposes the model to varied multi-scale crops and color perturbations without label supervision, yielding a more robust feature representation for amorphous patterns.
2. **Cancer-Associated Stroma (Class 7) vs. Smooth Muscle (Class 5):** Class 7 is universally the most difficult category across all 16 runs (F1 = 0.33 to 0.43 at 1% labels). Morphologically, desmoplastic stroma consists of reactive fibroblasts and collagen fibers that share spindle-cell morphology, eosinophilic staining, and fibrillar arrangements with smooth muscle (Class 5). At $28 \times 28$ resolution, the loss of fine intercellular borders increases reciprocal misclassifications.
3. **Adipose (Class 0) and Glass Slide Background (Class 1):** Both achieve $>0.91$ to $>0.98$ F1 even at 1% labels across all models due to distinct visual properties (empty lipid spaces and transparent background).

### 9.3 Representation Geometry & Exploratory t-SNE Analysis
To inspect embedding geometry independent of classifier heads, penultimate representations ($h \in \mathbb{R}^{512}$) were extracted from an exploratory subset of $N = 1,500$ unseen test patches from CRC-VAL-HE-7K:
- **Evaluation Distinction:** The primary benchmark in Table 1 evaluates test classification performance on all $N = 7,180$ test patches. The t-SNE analysis and cluster validity scores evaluate geometric clustering on an exploratory $N = 1,500$ subset.

#### Table 3: Unsupervised Cluster Validity Metrics on Exploratory Test Subset ($N = 1,500$)

| Feature Space | Silhouette Score ($\uparrow$, range $[-1, 1]$) | Davies-Bouldin Index ($\downarrow$, range $[0, \infty)$) | Geometric Manifold Characterization |
| :--- | :---: | :---: | :--- |
| **Untrained ResNet-18 (Random Init)** | $-0.0412$ | $4.8912$ | Disorganized manifold; overlapping tissue distributions. |
| **SimCLR SSL (Pretrained Only, No Labels)** | **+0.1162** | **3.2115** | Self-supervised contrastive learning discovers natural morphological structure without a single label. |
| **ImageNet Transfer (10% Labels)** | **+0.1415** | **2.1569** | General visual primitives group tissues with moderate inter-cluster separation. |
| **SimCLR Fine-Tuned (10% Labels)** | **+0.1924** | **2.0727** | Fine-tuning in-domain SSL representations produces tighter clusters than ImageNet transfer. |
| **Supervised Scratch (10% Labels)** | **+0.3197** | **1.4801** | **Highest Compaction.** Supervised cross-entropy explicitly minimizes intra-class variance on the training distribution. |

#### Geometric Insights
1. **Unsupervised Organization:** Pure SimCLR pretraining (without labels) increases the silhouette score from $-0.0412$ (random init) to $+0.1162$ and decreases the Davies-Bouldin index from $4.8912$ to $3.2115$. This confirms that instance discrimination groups histological textures into coherent neighborhoods based on optical density and morphology without human guidance.
2. **Supervised Compaction:** Supervised training directly optimizes class separation via cross-entropy loss, resulting in the highest silhouette score ($0.3197$). Cross-entropy actively collapses within-class features toward target class vertices, whereas contrastive pretraining preserves instance-level variance within classes.

---

## 10. Key Findings

1. **Performance in the 1% Label Regime ($N = 899$):**  
   In the 1% label regime, ImageNet transfer achieved the highest Macro F1 (0.7449), followed by SimCLR fine-tuning (0.7408), with both substantially outperforming Supervised Scratch (0.6815). SimCLR fine-tuning outpaced Supervised Scratch by **+5.05% Test Accuracy** (78.80% vs. 73.75%) and **+0.0593 Macro F1** (0.7408 vs. 0.6815), while recording the highest discriminative confidence across all four evaluated models (**Macro ROC-AUC: 0.9662**). Pretraining on unlabeled in-domain patches provided a more effective starting point than randomly initialized scratch weights when only ~80-129 labeled samples per class were available.
2. **ImageNet Transfer Learning Remains Strong in Low-to-Medium Regimes:**  
   ImageNet Transfer achieves the highest Macro F1 at 1% (0.7449), 5% (0.8375), and 10% (0.8495) labels. The broad visual feature diversity acquired from 1.28 million natural images (Gabor-like edges, texture filters, color gradients) transfers effectively to histological textures despite macroscopic domain differences.
3. **Supervised Scratch Achieves Highest Performance in the Full-Data Regime:**  
   When all 89,996 labeled training samples are available, Supervised Scratch achieves the highest performance across all paradigms (**90.18% Test Accuracy, 0.8721 Macro F1, 0.9916 ROC-AUC**). With sufficient sample density ($\sim 10,000$ samples/class), unconstrained optimization of ResNet-18 directly on the task-specific cross-entropy objective avoids transfer biases present in self-supervised or natural-domain initializations.
4. **Observed Linear Probe Performance Profile:**  
   Across the four evaluated data proportions, the SimCLR Linear Probe results remained within a narrow range: 0.6943 Macro F1 at 1%, 0.6955 at 5%, 0.6997 at 10%, and 0.7055 at 100% (+0.0112 overall span). In comparison, SimCLR full fine-tuning progressed from 0.7408 at 1% to 0.8277 at 100% (+0.0869 span). Linear probe training took 36.9s at 1% and 559.3s at 100%, compared to 52.2s and 2,038.7s for full fine-tuning.

---

## 11. Limitations

1. **Single-Seed Pilot Scope:** All 16 experimental conditions were evaluated using a single controlled random seed (Seed 42). While sample selection and batch shuffling were deterministic, this single-seed design does not quantify run-to-run stochastic variance or establish formal statistical significance. Multi-seed replication (e.g., across 3 to 5 independent seeds) is required for definitive confidence bounds.
2. **Standardized Low Resolution ($28 \times 28$ Pixels):** The PathMNIST benchmark compresses high-power histology into $28 \times 28$ tiles, attenuating subcellular cytological features (nuclear atypia, mitotic figures, chromatin margins) essential for clinical grading.
3. **Single Organ Domain:** Pretraining and evaluation were limited to colorectal carcinoma resection specimens. Transferability across diverse anatomical sites (e.g., breast, prostate, lung) and non-neoplastic pathologies requires broader multi-cohort validation.
4. **Contrastive False Negatives:** Standard SimCLR treats all distinct patches in a batch as negative pairs, penalizing intra-class alignment between patches of the same histological class.
5. **Non-Clinical Research Disclaimer:**  
   > **IMPORTANT DISCLAIMER:**  
   > This project represents an academic machine learning engineering investigation into representation learning mechanics and label-efficiency dynamics on standardized benchmark datasets. The models, representations, checkpoints, and evaluations documented herein **have not been cleared, approved, or validated for clinical diagnostic use, prognostic decision-making, patient management, or clinical trial stratification**. They must not be deployed in clinical practice or used to guide patient care.

---

## 12. Reproducibility and Setup Instructions

### 12.1 Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/<username>/medical-ssl-representation-learning.git
cd medical-ssl-representation-learning

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 12.2 Running Unit Tests
To verify implementation integrity across dataset loaders, augmentations, loss functions, and models:
```bash
python3 -m unittest discover -s tests
```
*Current test suite status: 22/22 tests passing.*

### 12.3 Running the Full Pipeline
```bash
# 1. Run SimCLR pretraining (or use existing checkpoint in checkpoints/simclr_encoder_best.pth)
python3 src/pretrain_simclr.py

# 2. Run the 16 downstream experimental conditions
python3 scripts/run_experiments.py

# 3. Regenerate summary tables (Markdown & LaTeX)
python3 scripts/summarize_results.py

# 4. Generate the PDF report
python3 scripts/generate_pdf_report.py
```

### 12.4 Computational Runtimes (Apple Silicon MPS)
- SimCLR Pretraining (15 epochs, 89,996 images): **24.5 minutes**
- 1% Regime (4 models, 15 epochs each): **3.5 minutes**
- 5% Regime (4 models, 15 epochs each): **7.6 minutes**
- 10% Regime (4 models, 15 epochs each): **13.0 minutes**
- 100% Regime (4 models, 15 epochs each): **113.5 minutes (1.9 hours)**
- Complete Pipeline: **~2.6 hours**

---

## 13. Future Improvements

1. **Multi-Seed Low-Data Replications:** Running 3 to 5 seeds specifically for the 1% and 5% regimes to report mean $\pm$ standard deviation intervals for small-sample stability.
2. **Scaling to Higher Image Resolutions:** Evaluating the contrastive pretraining pipeline on $224 \times 224$ patches and gigapixel whole-slide images (WSIs) to incorporate cellular-level cytologic features.
3. **Supervised Contrastive Learning (SupCon):** Implementing SupCon during downstream adaptation to eliminate false-negative penalties for patches belonging to the same histological class.
4. **Stain Normalization Pipelines:** Integrating color normalization algorithms (e.g., Macenko or Vahadane normalization) into the preprocessing pipeline to mitigate batch effects across multi-center pathology laboratories.

---

## 14. References

1. J. N. Kather, C. A. Weis, F. Bianconi, et al., "Multi-class texture analysis in colorectal cancer histology," *Scientific Reports*, vol. 6, p. 27988, 2016.
2. T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in *International Conference on Machine Learning (ICML)*, pp. 1597–1607, PMLR, 2020.
3. J. Yang, R. Shi, D. Wei, et al., "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification," *Scientific Data*, vol. 10, no. 1, p. 41, 2023.
4. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778, 2016.
5. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *International Conference on Learning Representations (ICLR)*, 2019.
6. J. Deng, W. Dong, R. Socher, et al., "ImageNet: A large-scale hierarchical image database," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 248–255, 2009.
7. O. Ciga, T. Xu, and A. L. Martel, "Self-supervised contrastive learning for digital histopathology," *Machine Learning with Applications*, vol. 7, p. 100198, 2022.

