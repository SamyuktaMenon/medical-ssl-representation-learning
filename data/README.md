# Dataset Documentation: PathMNIST (MedMNIST v2)

## 1. Overview and Source
PathMNIST is a biomedical benchmark dataset included in **MedMNIST v2**, derived from the colorectal cancer histology study:
- **Reference**: Kather JN, Halama N, Marx A. (2016). *100,000 histological images of human colorectal cancer and healthy tissue*. Zenodo. DOI: 10.5281/zenodo.1214456.
- **MedMNIST Reference**: Yang J, Shi R, Wei D, et al. (2023). *MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Scientific Data, 10(1), 41. DOI: 10.1038/s41597-023-01996-1.
- **Official Portal**: [https://medmnist.com/](https://medmnist.com/)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

## 2. Dataset Structure and Splits
PathMNIST is comprised of non-overlapping image patches extracted from hematoxylin and eosin (H&E) stained histological whole slide images:
- **Source Training Cohort**: NCT-CRC-HE-100K (100,000 non-overlapping patches from formalin-fixed paraffin-embedded samples).
  - Split deterministically by MedMNIST into:
    - **Training set**: 89,996 images (90% of NCT-CRC-HE-100K)
    - **Validation set**: 10,004 images (10% of NCT-CRC-HE-100K)
- **External Validation Test Cohort**: CRC-VAL-HE-7K (7,180 patches from an independent clinical center).
  - **Official Test set**: 7,180 images.
  - Crucially, the test set constitutes true external clinical validation across clinical sites, providing realistic domain robustness evaluation.

## 3. Classes and Tissue Annotations
PathMNIST contains 9 distinct colorectal tissue classes:
| Class ID | Tissue Name | Diagnostic Description |
|:---:|:---|:---|
| 0 | Adipose (ADI) | Mature fat tissue cells |
| 1 | Background (BACK) | Clear background / slide void without tissue |
| 2 | Debris (DEB) | Necrotic debris, mucus remnants, or staining artifacts |
| 3 | Lymphocytes (LYM) | Inflammatory infiltrate / immune cell clusters |
| 4 | Mucus (MUC) | Extracellular mucin pools |
| 5 | Smooth Muscle (MUS) | Muscularis mucosae and propria smooth muscle fibers |
| 6 | Normal Colon Mucosa (NORM) | Healthy, benign colonic epithelial crypts and glands |
| 7 | Cancer-Associated Stroma (STR) | Desmoplastic tumor stroma and reactive myofibroblasts |
| 8 | Colorectal Adenocarcinoma (TUM) | Invasive malignant colonic adenocarcinoma epithelium |

## 4. Image Resolution and Computational Trade-Offs
- Native NCT-CRC-HE patches were scanned at 0.5 microns/pixel (224×224 pixels).
- MedMNIST provides standard 28×28, 64×64, 128×128, and 224×224 resolutions in `.npz` format.
- Hardware Assessment:
  - 28×28 resolution: 107,180 images require ~206 MB storage, allowing fast in-memory batch caching, rapid contrastive pretraining, and extensive multi-seed label efficiency sweeps without out-of-memory thrashing on unified memory systems.
  - High resolutions (224×224): Require ~13.2 GB raw tensor storage in memory, which would exceed available RAM on standard workstations and prevent multi-experiment replication.

## 5. Strict Data Leakage Prevention Protocol
To guarantee defensible scientific integrity:
1. **Self-Supervised Pretraining Leakage Guard**: Pretraining of SimCLR encoders strictly accesses only the unlabeled images from the official training split (`split='train'`). Under no circumstances are validation or test images touched during pretraining.
2. **Deterministic Stratified Subsetting**: Label efficiency experiments evaluate 1%, 5%, 10%, and 100% labeled training samples. Subsets are generated exclusively from the training split using fixed random seeds and stratified sampling to preserve relative class distributions.
3. **Cross-Method Subset Parity**: The exact same subset indices are loaded across all evaluated baselines (Supervised Scratch, ImageNet Transfer, and SimCLR Linear Probe / Fine-Tuning).
4. **Validation Isolation**: The validation split (`split='val'`) is reserved exclusively for model selection, early stopping, and hyperparameter tuning.
5. **Test Set Quarantine**: The official external test set (`split='test'`) remains completely quarantined and is evaluated only once on the final selected checkpoint per configuration.

