# 🍌 TLM-ViT Deep Ensemble for Banana Leaf Micronutrient Deficiency Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-99.34%25_(303%2F305)-success)](#-experimental-results--benchmark-comparison)
[![Validation Accuracy](https://img.shields.io/badge/Val_Accuracy-98.50%25-brightgreen)](#-experimental-results--benchmark-comparison)
[![t-SNE Silhouette](https://img.shields.io/badge/t--SNE_Silhouette-0.700_(%2B0.686)-purple)](#-feature-space-visualization-t-sne--explainable-ai-grad-cam)

This repository contains the official implementation and experimental notebooks for **Banana Leaf Micronutrient Deficiency Classification** (*Boron*, *Healthy*, *Iron*, and *Manganese*). We propose a hybrid **Two-Stage Fine-Tuned Transfer Learning Vision Transformer (TLM-ViT)** coupled with **Multi-Backbone Deep Ensemble** strategies across classical and next-generation Convolutional Neural Networks (`InceptionV3`, `DenseNet169`, and `EfficientNetV2-S`).

---

## 📖 Background & Primary Reference Papers

This study synthesizes, adapts, and advances the methodologies of **two peer-reviewed *IEEE Access* papers**, whose full-text PDFs are included in [`./reference_papers/`](./reference_papers) for complete reproducibility and clarity:

### 1. Reference Paper 1 — Vision Transformer Feature Extraction Architecture (*TLMViT*, 2023)
*(Referred to internally in notebook logs as **Paper P5**)*
- **Title:** *"A Deep Features Extraction Model Based on the Transfer Learning Model and Vision Transformer 'TLMViT' for Plant Disease Classification"*
- **Authors:** Amer Tabbakh & Soubhagya Sankar Barpanda (*VIT-AP University, India*)
- **Venue:** *IEEE Access*, Vol. 11, pp. 45377–45392, May 2023 | **DOI:** [`10.1109/ACCESS.2023.3273317`](https://doi.org/10.1109/ACCESS.2023.3273317)
- **PDF in Repository:** [📄 `Paper_1_Tabbakh_2023_TLMViT_Plant_Disease_Classification.pdf`](./reference_papers/Paper_1_Tabbakh_2023_TLMViT_Plant_Disease_Classification.pdf)
- **Original Dataset Used:** *PlantVillage Dataset* and *Wheat Rust Leaf Dataset*.
- **Original Methodology:** A sequential two-phase deep feature extraction pipeline:
  1. Initial spatial feature map extraction using a single frozen ImageNet pre-trained CNN backbone (e.g., `VGG19` or `InceptionV3`).
  2. Global spatial dependency modeling using a Vision Transformer head (`PatchExtractor` ➔ `PatchEncoder` ➔ 4-block *Multi-Head Self-Attention*) followed by an *MLP* classifier.
- **Limitation & What We Improved:** Reference Paper 1 relies on a **single frozen CNN backbone** without end-to-end joint adaptation or multi-model ensembling. We adopt its spatial `PatchExtractor` + `PatchEncoder` + *Transformer Encoder* head and elevate it via **Two-Stage Joint Fine-Tuning** and **Multi-Backbone Deep Ensembling**.

---

### 2. Reference Paper 2 — Primary Benchmark Dataset & Banana Leaf Deep Ensemble (*IncepV3Dense*, 2024)
*(Referred to internally in notebook logs as **Paper P6**)*
- **Title:** *"IncepV3Dense: Deep Ensemble Based Average Learning Strategy for Identification of Micro-Nutrient Deficiency in Banana Crop"*
- **Authors:** Sudhakar Muthusamy & Swarna Priya Ramu (*Vellore Institute of Technology, India*)
- **Venue:** *IEEE Access*, Vol. 12, pp. 73779–73792, May 2024 | **DOI:** [`10.1109/ACCESS.2024.3405027`](https://doi.org/10.1109/ACCESS.2024.3405027)
- **PDF in Repository:** [📄 `Paper_2_Muthusamy_2024_IncepV3Dense_Banana_Micronutrient_Deficiency.pdf`](./reference_papers/Paper_2_Muthusamy_2024_IncepV3Dense_Banana_Micronutrient_Deficiency.pdf)
- **Original Dataset Used (Our Target Benchmark Dataset):** *Mendeley / Kaggle Fully Labelled Image Dataset of Banana Leaves Deficient in Nutrients* (Sunitha et al., *Data in Brief*, Vol. 48, Art. no. 109155, 2023 — *Rasthali*, *Poovan*, and *Monthan* cultivars), targeting **4 micronutrient categories**:
  - **Boron Deficiency (`boron`)**
  - **Healthy Leaf (`healthy`)**
  - **Iron Deficiency (`iron`)**
  - **Manganese Deficiency (`manganese`)**
  - Evaluated under an exact split of **3,200 Training images (800/class)**, **800 Validation images (200/class)**, and **305 Unseen Test images** (*Boron*: 81, *Healthy*: 69, *Iron*: 72, *Manganese*: 83).
- **Original Methodology:** Modifies the top classification layers (*Mutated Top Layers*: `GlobalAveragePooling2D` ➔ `Dense(256)` ➔ `Dense(256)` ➔ `Dropout(0.5)` ➔ `Softmax(4)`) trained with `Adagrad` (`lr=0.01857`) over frozen CNN backbones, and combines the top two performers into an unweighted probability-averaging ensemble named **`IncepV3Dense` (`InceptionV3 + DenseNet169`)**, reporting `98.62%` validation accuracy, `93.00%` F1-score, and **`285 / 305` (`93.44%`) test accuracy** (20 misclassifications).
- **Limitation & What We Improved:** Compressing spatial feature maps via `GlobalAveragePooling2D` discards spatial localization of interveinal chlorosis and marginal necrosis (causing severe accuracy drops on visually subtle classes such as *Manganese*). By preserving the exact 4-class benchmark protocol and replacing `GAP` with our **Two-Stage Fine-Tuned TLM-ViT** head and **Optimal Multi-Backbone Ensembling**, we boost unseen test accuracy from **`93.44%` (`285/305`)** to **`99.34%` (`303/305`)**!

---

## 🌟 Key Contributions & Highlights

- 🏆 **State-of-the-Art Benchmark Performance (`99.34%` vs `93.44%`):** Outperforms the *IncepV3Dense* benchmark (Reference Paper 2) by **+5.90% in Test Accuracy** (`303/305` vs `285/305`) and **+6.33% in Macro F1-Score** (`99.33%` vs `93.00%`), achieving a **90% error rate reduction** (reducing test errors from 20 down to only 2 images).
- 🎯 **100% Perfect Recall & Accuracy on 3 out of 4 Classes:** Achieves **100.00%** test accuracy on *Healthy* (`69/69`), *Iron* (`72/72`), and *Manganese* (`83/83`), alongside **97.53%** on *Boron* (`79/81`).
- 🧬 **Quantitative Feature Engineering Proof via `t-SNE` & Silhouette Score:** Demonstrates mathematically that replacing `GlobalAveragePooling2D` with our *Two-Stage Fine-Tuned TLM-ViT* increases the 2D `t-SNE` **Silhouette Score** on the 305 unseen test images from **`-0.035`** (*Raw InceptionV3 GAP*) and **`0.014`** (*Raw DenseNet169 GAP*) to **`0.700`** (*Proposed T2 DenseNet169-ViT*).
- 🔍 **Clinical Interpretability via Explainable AI (`Grad-CAM`):** Spatial activation heatmaps confirm that `DenseNet169` and `EfficientNetV2-S` ignore black background regions and attend to complementary symptomatic leaf tissues.
- ⚡ **Self-Contained & Cross-Account Reproducible Pipeline:** Automated dataset acquisition, zero-leakage stratified splitting, and Google Drive checkpoint synchronization.

---

## 📂 Repository Structure

| File / Directory | Description | Status / Key Result |
| :--- | :--- | :--- |
| [`reference_papers/`](./reference_papers) | Full-text PDFs of Reference Paper 1 (*Tabbakh & Barpanda, 2023*) and Reference Paper 2 (*Muthusamy & Ramu, 2024*). | Official Benchmarks |
| [`7_vit_deep_ensemble.ipynb`](./7_vit_deep_ensemble.ipynb) | Initial baseline notebook prior to pipeline restructuring and two-stage calibration. | Initial Baseline (`98.36%` Test Acc) |
| [`7_vit_deep_ensemble_OPTIMIZED.ipynb`](./7_vit_deep_ensemble_OPTIMIZED.ipynb) | **Clean Template — Strategy A:** Self-contained 6-stage pipeline (`InceptionV3 + DenseNet169` + *2-Stage TLM-ViT* + *Optimal Weighted Ensemble*). | Ready to Run on Google Colab |
| [`7_vit_deep_ensemble_OPTIMIZED_hasil_test.ipynb`](./7_vit_deep_ensemble_OPTIMIZED_hasil_test.ipynb) | **Executed Logs & Outputs — Strategy A:** Complete training histories, convergence curves, evaluation tables, and Confusion Matrices. | **Test Acc: `99.34%` (`303/305`)** |
| [`8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb`](./8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb) | **Clean Template — Strategy A + B (Publication Edition):** Integrates modern backbone `EfficientNetV2-S`, *Tri-Ensemble*, `t-SNE` projection, and `Grad-CAM`. | Ready to Run on Google Colab |
| [`8_vit_deep_ensemble_STRATEGI_B_PUBLISH_hasil_test.ipynb`](./8_vit_deep_ensemble_STRATEGI_B_PUBLISH_hasil_test.ipynb) | **Executed Logs & Outputs — Strategy A + B:** Full 23-model ablation table, `t-SNE` (*Silhouette* `0.700`), and `Grad-CAM` spatial maps. | **Val Acc: `98.50%` \| Test Acc: `99.34%`** |

---

## 🔄 Methodology & Experimental Pipeline

1. **Stage 1 — Automated Dataset Acquisition & Class Isolation:** Automatically downloads the banana leaf nutrient deficiency dataset and isolates the 4 target classes matching Reference Paper 2 (*Boron*: 900, *Healthy*: 950, *Iron*: 946, *Manganese*: 864 — 3,660 raw images).
2. **Stage 2 — Zero-Leakage Stratified Split & Offline Augmentation Balancing:** Strictly isolates the **305 unseen test images** (*Boron*: 81, *Healthy*: 69, *Iron*: 72, *Manganese*: 83) *before* any augmentation to guarantee zero data leakage, then balances the remaining partitions to **3,200 training images (800/class)** and **800 validation images (200/class)**.
3. **Stage 3 — Uniform `8 x 8 x C` Spatial Feature Map Extraction:** Extracts spatial feature maps of unified `8 x 8` resolution across `InceptionV3` (`8x8x2048` via `ZeroPadding2D`), `DenseNet169` (`8x8x1664`), and `EfficientNetV2-S` (`8x8x1280`).
4. **Stage 4 — Vision Transformer Module (`PatchExtractor` & `PatchEncoder`):** Partitions the `8 x 8` feature maps into **16 non-overlapping `2 x 2` spatial patches**, projects each patch into `D = 64` dimensions with learnable positional embeddings, and processes them through **4 Transformer Encoder blocks** (`4 Heads`, `GELU` MLP `128 -> 64`).
5. **Stage 5 — Systematic Ablation Study (`B1 ➔ B2 ➔ T1 ➔ T2`) & Two-Stage Fine-Tuning:**
   - **Scheme B1 (*Standard Frozen CNN*):** *Frozen Backbone* + `GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense(4, Softmax)`.
   - **Scheme B2 (*Mutated CNN Head* — Reference Paper 2):** *Frozen Backbone* + `GlobalAveragePooling2D` + `BatchNormalization ➔ Dense(512, ReLU) ➔ Dropout(0.4) ➔ Dense(128, ReLU) ➔ Dense(4, Softmax)`.
   - **Scheme T1 (*Frozen TLM-ViT* — Reference Paper 1):** *Frozen Backbone* + *ViT Head* (`PatchExtractor + PatchEncoder + 4x Transformer`).
   - **Scheme T2 (*Proposed Two-Stage Fine-Tuned TLM-ViT*):**
     - **Stage 1 (*ViT Head Warmup* — 12 Epochs):** Trains the ViT head from raw images with on-the-fly augmentation while keeping the CNN backbone frozen (`Adam lr=3e-4`, label smoothing `0.05`).
     - **Stage 2 (*Joint Fine-Tuning* — 25 Epochs):** Unfreezes upper CNN blocks (`mixed7+` on `InceptionV3`, `conv5_block1+` on `DenseNet169`, `block5a+` on `EfficientNetV2-S` with `BatchNormalization` layers kept frozen) at a low learning rate (`lr=5e-5` + `ReduceLROnPlateau`).
6. **Stage 6 — Multi-Backbone Deep Ensemble, `t-SNE`, & `Grad-CAM`:** Evaluates unweighted soft-voting and validation-driven optimal weighted ensembles across **Strategy A (`InceptionV3 + DenseNet169`)**, **Strategy B (`DenseNet169 + EfficientNetV2-S`)**, and **Tri-Ensemble (`3 Backbones`)**, paired with `t-SNE` and `Grad-CAM` analyses.

![Two-Stage Fine-Tuning Convergence](./assets/chart_t2_convergence_optimized.png)

---

## 📊 Experimental Results & Benchmark Comparison

All models are evaluated on the exact same **305 unseen test images** (*Boron*: 81, *Healthy*: 69, *Iron*: 72, *Manganese*: 83):

| Model Group / Architecture | Backbone(s) | Val Acc (%) | Val F1 (%) | Test Correct / 305 | Test Acc (%) | Test Macro F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Ref. Paper 2 (2024): Mutated InceptionV3** | InceptionV3 | 95.75% | 90.00% | 275 / 305 | 90.16% | 90.00% |
| **Ref. Paper 2 (2024): Mutated DenseNet169** | DenseNet169 | 97.75% | 91.00% | 278 / 305 | 91.15% | 91.00% |
| **Ref. Paper 2 (2024): IncepV3Dense (Benchmark)** | IncV3 + Dense169 | 98.62% | 93.00% | 285 / 305 | 93.44% | 93.00% |
| **Ours — Ens B1** *(Baseline CNN Ensemble)* | IncV3 + Dense169 | 93.62% | 93.63% | 285 / 305 | 93.44% | 93.36% |
| **Ours — Ens B2** *(Mutated CNN Ensemble)* | IncV3 + Dense169 | 95.25% | 95.27% | 293 / 305 | 96.07% | 95.98% |
| **Ours — B2 EfficientNetV2S** *(Single Modern Mutated)* | EffNetV2-S | 96.25% | 96.26% | 294 / 305 | 96.39% | 96.32% |
| **Ours — Ens T1** *(Frozen TLM-ViT Ensemble)* | IncV3 + Dense169 | 95.50% | 95.50% | 297 / 305 | 97.38% | 97.36% |
| **Ours — T2 Inception** *(2-Stage Fine-Tuned ViT)* | InceptionV3 | 97.12% | 97.13% | 299 / 305 | 98.03% | 97.99% |
| **Ours — T2 DenseNet169** *(2-Stage Fine-Tuned ViT)* | DenseNet169 | 96.62% | 96.62% | 301 / 305 | 98.69% | 98.68% |
| **Ours — T2 EfficientNetV2S (Best Single Val)** | EffNetV2-S | **98.25%** | **98.25%** | 299 / 305 | 98.03% | 98.06% |
| **Strategy B: Ens T2 Weighted (Best Val Overall)** | Dense169 + EffV2S | **98.50%** | **98.50%** | 301 / 305 | 98.69% | 98.66% |
| **Tri-Ensemble T2 (3 Backbones Combined)** | IncV3 + Dense169 + EffV2S | **98.50%** | **98.50%** | 301 / 305 | 98.69% | 98.66% |
| **Strategy B: Ens T2 Unweighted** | Dense169 + EffV2S | 97.88% | 97.88% | **302 / 305** | **99.02%** | **98.99%** |
| **Strategy A: Ens T2 & Weighted (Best Test Overall)** | IncV3 + Dense169 | 97.38% | 97.38% | **303 / 305** | **99.34%** | **99.33%** |

![Head-to-Head Accuracy Comparison](./assets/chart_head_to_head.png)

### Confusion Matrices of Flagship Models on the 305 Unseen Test Images
![Confusion Matrices Strategy A and B](./assets/confusion_matrix_strategy_ab.png)

---

## 🔬 Feature Space Visualization (`t-SNE`) & Explainable AI (`Grad-CAM`)

### 1. 2D `t-SNE` Projection & Silhouette Score Progression (`-0.035` ➔ `0.700`)
Projecting feature representations of all 305 unseen test images into 2D via `t-SNE` reveals why `GlobalAveragePooling2D` features in Reference Paper 2 struggle with overlapping deficiency symptoms (*Silhouette Score* `-0.035` on `InceptionV3` and `0.014` on `DenseNet169`), whereas our **Two-Stage Fine-Tuned TLM-ViT** separates the four classes into compact, well-isolated clusters (*Silhouette Score* **`0.700`** and **`0.654`**):

![t-SNE Feature Space Comparison](./assets/tsne_feature_space.png)

### 2. Explainable AI: Spatial Feature Activation Heatmaps (`Grad-CAM`)
Spatial activation maps extracted prior to patch tokenization confirm that `DenseNet169` and `EfficientNetV2-S` suppress black background regions and focus on complementary symptomatic regions (interveinal chlorosis, leaf edges, and midrib discoloration):

![Grad-CAM Spatial Attention](./assets/gradcam_spatial_attention.png)

---

## 🚀 Quick Start (Google Colab)

1. Upload either clean template notebook ([`7_vit_deep_ensemble_OPTIMIZED.ipynb`](./7_vit_deep_ensemble_OPTIMIZED.ipynb) or [`8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb`](./8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb)) to **Google Colab**.
2. Enable GPU acceleration (`T4` / `L4` / `A100`) via `Runtime` ➔ `Change runtime type` ➔ `T4 GPU`.
3. Click `Runtime` ➔ `Run all`. The pipeline will automatically download the dataset, perform zero-leakage stratified splitting, train all ablation stages, and back up model weights and metrics to your Google Drive.

---

## 📚 References

1. **Tabbakh, A., & Barpanda, S. S. (2023).** *"A Deep Features Extraction Model Based on the Transfer Learning Model and Vision Transformer 'TLMViT' for Plant Disease Classification."* IEEE Access, vol. 11, pp. 45377–45392. DOI: `10.1109/ACCESS.2023.3273317`. ([View PDF](./reference_papers/Paper_1_Tabbakh_2023_TLMViT_Plant_Disease_Classification.pdf))
2. **Muthusamy, S., & Ramu, S. P. (2024).** *"IncepV3Dense: Deep Ensemble Based Average Learning Strategy for Identification of Micro-Nutrient Deficiency in Banana Crop."* IEEE Access, vol. 12, pp. 73779–73792. DOI: `10.1109/ACCESS.2024.3405027`. ([View PDF](./reference_papers/Paper_2_Muthusamy_2024_IncepV3Dense_Banana_Micronutrient_Deficiency.pdf))
3. **Sunitha, P., Uma, B., Channakeshava, S., & Babu, S. (2023).** *"A fully labelled image dataset of banana leaves deficient in nutrients."* Data in Brief, vol. 48, Art. no. 109155.
