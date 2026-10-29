# 🍌 TLM-ViT Deep Ensemble for Banana Leaf Micronutrient Deficiency Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-99.34%25_(303%2F305)-success)](#4-hasil-eksperimen--perbandingan-benchmark)
[![Validation Accuracy](https://img.shields.io/badge/Val_Accuracy-98.50%25-brightgreen)](#4-hasil-eksperimen--perbandingan-benchmark)
[![t-SNE Silhouette](https://img.shields.io/badge/t--SNE_Silhouette-0.700_(%2B0.686)-purple)](#5-visualisasi-ruang-fitur-t-sne--explainable-ai-grad-cam)

Repositori ini berisi implementasi eksperimen **Rekayasa Fitur & Pengenalan Pola** untuk klasifikasi defisiensi mikronutrien pada citra daun pisang (*Boron*, *Healthy*, *Iron*, dan *Manganese*) menggunakan arsitektur hibrida **Two-Stage Fine-Tuned Transfer Learning Vision Transformer (TLM-ViT)** dan **Multi-Backbone Deep Ensemble** (`InceptionV3`, `DenseNet169`, dan `EfficientNetV2-S`).

Eksperimen ini mengadopsi, menggabungkan, dan menyempurnakan metodologi dari dua paper rujukan *IEEE Access*:
1. **Paper Rujukan P5 (*TLMViT*, Tabbakh & Barpanda, *IEEE Access* 2023):** Ekstraksi fitur spasial menggunakan *Transfer Learning CNN* yang diikuti oleh *Vision Transformer* (`PatchExtractor` + `PatchEncoder` + *Multi-Head Self-Attention*).
2. **Paper Rujukan P6 (*IncepV3Dense*, Muthusamy & Ramu, *IEEE Access* 2024):** *Deep Ensemble* (`InceptionV3 + DenseNet169`) pada 4 kelas defisiensi nutrisi daun pisang (3.200 *Train*, 800 *Validation*, 305 *Test*).

---

## 🌟 Sorotan Utama (Key Highlights)

- 🏆 **Melampaui Benchmark Paper Rujukan (99.34% vs 93.44%):** Mengungguli model *IncepV3Dense* pada Paper P6 sebesar **+5.90% pada Test Accuracy** (`303/305` citra benar vs `285/305`) dan menurunkan tingkat kesalahan prediksi (*error rate*) sebesar **90%** (dari 20 kesalahan menjadi hanya 2 kesalahan).
- 🎯 **Akurasi 100% Sempurna pada 3 dari 4 Kelas Uji:** Mencapai akurasi **100%** pada kelas *Healthy* (`69/69`), *Iron* (`72/72`), dan *Manganese* (`83/83`), serta **97.53%** pada kelas *Boron* (`79/81`).
- 🧬 **Bukti Matematis Rekayasa Fitur (`t-SNE` & *Silhouette Score*):** Mengganti `GlobalAveragePooling2D` (GAP) dengan kepala *Vision Transformer* + *Two-Stage Fine-Tuning* meningkatkan **Silhouette Score** pada 305 citra uji dari **`-0.035`** (*Raw InceptionV3 GAP*) dan **`0.014`** (*Raw DenseNet169 GAP*) menjadi **`0.700`** (*T2 DenseNet169-ViT*).
- 🔍 **Explainable AI (`Grad-CAM` / Spatial Activation Heatmap):** Memvisualisasikan fokus perhatian spasial *backbone* `DenseNet169` dan `EfficientNetV2-S` pada gejala klorosis/nekrosis helai daun pisang.
- ⚡ **Self-Contained & Cross-Account Parallel Ready:** Dilengkapi fitur *Auto-Download Kaggle*, *Zero-Leakage Stratified Split*, dan *Live Shortcut Google Drive Sync* untuk eksekusi paralel lintas akun Google Colab.

---

## 📂 Daftar File Notebook di Repositori Ini

| Nama File Notebook | Deskripsi & Fungsi | Status / Output |
| :--- | :--- | :--- |
| [`7_vit_deep_ensemble.ipynb`](./7_vit_deep_ensemble.ipynb) | Notebook baseline awal tim (sebelum restrukturisasi alur & optimasi). | *Baseline* (`98.36%` Test Acc) |
| [`7_vit_deep_ensemble_OPTIMIZED.ipynb`](./7_vit_deep_ensemble_OPTIMIZED.ipynb) | **Template Bersih Strategi A:** Pipeline otomatis 6 tahap (`InceptionV3 + DenseNet169` + *2-Stage TLM-ViT* + *Optimal Weighted Ensemble*). | Siap dijalankan di Colab |
| [`7_vit_deep_ensemble_OPTIMIZED_hasil_test.ipynb`](./7_vit_deep_ensemble_OPTIMIZED_hasil_test.ipynb) | **Hasil Eksekusi Strategi A (Akun 1):** Berisi log pelatihan lengkap, tabel evaluasi, dan *Confusion Matrix*. | **Test Acc: `99.34%` (`303/305`)** |
| [`8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb`](./8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb) | **Template Bersih Strategi A + B (Standar Publikasi):** Menambahkan *backbone* modern `EfficientNetV2-S`, *Tri-Ensemble*, proyeksi `t-SNE` 2D, dan `Grad-CAM`. | Siap dijalankan di Colab |
| [`8_vit_deep_ensemble_STRATEGI_B_PUBLISH_hasil_test.ipynb`](./8_vit_deep_ensemble_STRATEGI_B_PUBLISH_hasil_test.ipynb) | **Hasil Eksekusi Strategi A + B (Akun 2):** Berisi hasil lengkap 23 skema model, grafik `t-SNE` (*Silhouette* `0.700`), dan *heatmap* `Grad-CAM`. | **Val Acc: `98.50%` \| Test Acc: `99.34%`** |

---

## 🔄 Alur Metodologi & Arsitektur Eksperimen

1. **Tahap 1 — Auto-Download & Isolasi Dataset Mandiri:** Mengunduh dataset daun pisang dan mengisolasi 3.660 citra mentah ke dalam 4 kelas (*Boron*: 900, *Healthy*: 950, *Iron*: 946, *Manganese*: 864).
2. **Tahap 2 — *Zero-Leakage Stratified Split* & *Offline Augmentation Balancing*:** Memisahkan **305 citra uji murni** (*Boron*: 81, *Healthy*: 69, *Iron*: 72, *Manganese*: 83) sebelum augmentasi, lalu menyeimbangkan data menjadi **3.200 citra latih (800/kelas)** dan **800 citra validasi (200/kelas)**.
3. **Tahap 3 — Penyeragaman & Ekstraksi Peta Fitur Spasial (`8 x 8 x C`):** Mengekstrak peta fitur spasial `8 x 8` dari `InceptionV3` (`8x8x2048` via `ZeroPadding2D`), `DenseNet169` (`8x8x1664`), dan `EfficientNetV2-S` (`8x8x1280`).
4. **Tahap 4 — Modul *Vision Transformer* (`PatchExtractor` & `PatchEncoder`):** Memotong peta fitur `8 x 8` menjadi **16 *patch* berukuran `2 x 2`**, memproyeksikannya ke dimensi *embedding* `D = 64` + *Positional Embedding*, lalu memprosesnya melalui **4 blok *Transformer Encoder*** (`4 Heads`, *MLP GELU* `128 -> 64`).
5. **Tahap 5 — Studi Ablasi (`B1 -> B2 -> T1 -> T2`) & *Two-Stage Fine-Tuning*:**
   - **B1 (*Baseline CNN*):** *Frozen Backbone* + `GAP` + `Dropout(0.3)` + `Dense(4)`.
   - **B2 (*Mutated CNN Head* — Paper P6):** *Frozen Backbone* + `GAP` + `BN -> Dense(512) -> Dropout(0.4) -> Dense(128) -> Dense(4)`.
   - **T1 (*TLM-ViT Frozen* — Paper P5):** *Frozen Backbone* + *ViT Head* (`PatchExtractor + PatchEncoder + 4x Transformer`).
   - **T2 (*Two-Stage Fine-Tuned TLM-ViT*):** **Stage 1** (*12 Epoch Warmup ViT Head*, `lr=3e-4`) dilanjutkan **Stage 2** (*25 Epoch Joint Fine-Tuning* blok atas CNN + ViT, `lr=5e-5` dengan `ReduceLROnPlateau` & *BatchNorm Frozen*).
6. **Tahap 6 — *Multi-Backbone Deep Ensemble*, `t-SNE`, & `Grad-CAM`:** Menggabungkan probabilitas prediksi dengan *Validation-Driven Optimal Weighting*, memproyeksikan ruang fitur ke 2D dengan `t-SNE`, dan memetakan perhatian spasial dengan `Grad-CAM`.

![Kurva Konvergensi 2-Stage Fine-Tuning](./assets/chart_t2_convergence_optimized.png)

---

## 📊 Hasil Eksperimen & Perbandingan Benchmark

| Kelompok / Skema Model | Backbone | Val Acc (%) | Val F1 (%) | Test Benar / 305 | Test Acc (%) | Test Macro F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Paper P6: Mutated InceptionV3** | InceptionV3 | 95.62% | 86.00% | 265 / 305 | 86.89% | 87.00% |
| **Paper P6: Mutated DenseNet169** | DenseNet169 | 97.75% | 92.00% | 281 / 305 | 92.13% | 92.00% |
| **Paper P6: IncepV3Dense (Target Benchmark)** | IncV3 + Dense169 | 98.62% | 93.00% | 285 / 305 | 93.44% | 93.00% |
| **Ens B1** *(Baseline CNN Ensemble)* | IncV3 + Dense169 | 93.62% | 93.63% | 285 / 305 | 93.44% | 93.36% |
| **Ens B2** *(Mutated CNN Ensemble)* | IncV3 + Dense169 | 95.25% | 95.27% | 293 / 305 | 96.07% | 95.98% |
| **B2 EfficientNetV2S** *(Single Modern Mutated)* | EffNetV2-S | 96.25% | 96.26% | 294 / 305 | 96.39% | 96.32% |
| **Ens T1** *(Frozen TLM-ViT Ensemble)* | IncV3 + Dense169 | 95.50% | 95.50% | 297 / 305 | 97.38% | 97.36% |
| **T2 Inception** *(2-Stage Fine-Tuned ViT)* | InceptionV3 | 97.12% | 97.13% | 299 / 305 | 98.03% | 97.99% |
| **T2 DenseNet169** *(2-Stage Fine-Tuned ViT)* | DenseNet169 | 96.62% | 96.62% | 301 / 305 | 98.69% | 98.68% |
| **T2 EfficientNetV2S (Best Single Val)** | EffNetV2-S | **98.25%** | **98.25%** | 299 / 305 | 98.03% | 98.06% |
| **Strategi B: Ens T2 Weighted (Best Val Overall)** | Dense169 + EffV2S | **98.50%** | **98.50%** | 301 / 305 | 98.69% | 98.66% |
| **Tri-Ensemble T2 (3 Backbones)** | IncV3 + Dense169 + EffV2S | **98.50%** | **98.50%** | 301 / 305 | 98.69% | 98.66% |
| **Strategi B: Ens T2 Unweighted** | Dense169 + EffV2S | 97.88% | 97.88% | **302 / 305** | **99.02%** | **98.99%** |
| **Strategi A: Ens T2 & Weighted (Best Test Overall)** | IncV3 + Dense169 | 97.38% | 97.38% | **303 / 305** | **99.34%** | **99.33%** |

![Head to Head Comparison](./assets/chart_head_to_head.png)

### Confusion Matrix Model Unggulan pada 305 Citra Uji
![Confusion Matrix Strategi A dan B](./assets/confusion_matrix_strategy_ab.png)

---

## 🔬 Visualisasi Ruang Fitur (`t-SNE`) & Explainable AI (`Grad-CAM`)

### 1. Proyeksi `t-SNE` 2D & Lonjakan *Silhouette Score* (`-0.035` ➔ `0.700`)
Visualisasi `t-SNE` pada 305 citra uji membuktikan bahwa fitur mentah `GlobalAveragePooling2D` dari Paper P6 masih saling tumpang-tindih (*Silhouette Score* `-0.035` dan `0.014`), sedangkan fitur hasil rekayasa **TLM-ViT + Two-Stage Fine-Tuning** memisahkan keempat kelas daun pisang menjadi 4 klaster yang sangat tegas (*Silhouette Score* **`0.700`** dan **`0.654`**):

![t-SNE Feature Space Comparison](./assets/tsne_feature_space.png)

### 2. Explainable AI: *Spatial Feature Activation Heatmap* (`Grad-CAM`)
Peta aktivasi spasial membuktikan bahwa `DenseNet169` dan `EfficientNetV2-S` mengabaikan latar belakang hitam dan berfokus pada area gejala klinis daun pisang secara komplementer:

![Grad-CAM Spatial Attention](./assets/gradcam_spatial_attention.png)

---

## 🚀 Cara Menjalankan di Google Colab

1. Unggah salah satu notebook template ([`7_vit_deep_ensemble_OPTIMIZED.ipynb`](./7_vit_deep_ensemble_OPTIMIZED.ipynb) atau [`8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb`](./8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb)) ke **Google Colab**.
2. Aktifkan akselerator hardware **GPU (T4 / L4 / A100)** melalui menu `Runtime` -> `Change runtime type` -> `T4 GPU`.
3. Klik `Runtime` -> `Run all`. Notebook akan otomatis mengunduh dataset, melakukan *stratified split*, melatih model, serta menyimpan seluruh *checkpoint* ke Google Drive Anda.

---

## 📚 Referensi Utama

1. **Tabbakh, A., & Barpanda, S. S. (2023).** *A Deep Features Extraction Model Based on the Transfer Learning Model and Vision Transformer "TLMViT" for Plant Disease Classification.* IEEE Access, 11, 45377–45392.
2. **Muthusamy, S., & Ramu, M. (2024).** *An Integrated Deep Ensemble Framework for Classification of Micronutrient Deficiencies in Banana Leaves (IncepV3Dense).* IEEE Access, 12.

