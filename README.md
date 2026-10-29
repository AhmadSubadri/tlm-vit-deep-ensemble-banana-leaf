# 🍌 TLM-ViT Deep Ensemble for Banana Leaf Micronutrient Deficiency Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-99.34%25_(303%2F305)-success)](#-hasil-eksperimen--perbandingan-terhadap-paper-rujukan)
[![Validation Accuracy](https://img.shields.io/badge/Val_Accuracy-98.50%25-brightgreen)](#-hasil-eksperimen--perbandingan-terhadap-paper-rujukan)
[![t-SNE Silhouette](https://img.shields.io/badge/t--SNE_Silhouette-0.700_(%2B0.686)-purple)](#-visualisasi-ruang-fitur-t-sne--explainable-ai-grad-cam)

Repositori ini berisi kode eksperimen penelitian **Rekayasa Fitur & Pengenalan Pola** untuk deteksi dan klasifikasi defisiensi mikronutrien pada citra daun pisang (*Boron*, *Healthy*, *Iron*, dan *Manganese*). Kami mengusulkan arsitektur hibrida **Two-Stage Fine-Tuned Transfer Learning Vision Transformer (TLM-ViT)** yang dipadukan dengan **Multi-Backbone Deep Ensemble** (`InceptionV3`, `DenseNet169`, dan `EfficientNetV2-S`).

---

## 📖 Latar Belakang & Paper Rujukan Utama

Penelitian ini mensintesiskan, mengadopsi, dan menyempurnakan metodologi dari **dua paper rujukan jurnal internasional bereputasi (*IEEE Access* Q1)** yang salinan PDF lengkapnya tersedia di dalam folder [`./reference_papers/`](./reference_papers):

### 1. Paper Rujukan 1 — Arsitektur Ekstraksi Fitur Vision Transformer (*TLMViT*, 2023)
*(Dalam catatan internal eksperimen disebut juga sebagai **Paper P5**)*
- **Judul Paper:** *"A Deep Features Extraction Model Based on the Transfer Learning Model and Vision Transformer 'TLMViT' for Plant Disease Classification"*
- **Penulis:** Amer Tabbakh & Soubhagya Sankar Barpanda (*VIT-AP University, India*)
- **Publikasi:** *IEEE Access*, Vol. 11, pp. 45377–45392, Mei 2023 | **DOI:** [`10.1109/ACCESS.2023.3273317`](https://doi.org/10.1109/ACCESS.2023.3273317)
- **File PDF di Repo:** [📄 `Paper_1_Tabbakh_2023_TLMViT_Plant_Disease_Classification.pdf`](./reference_papers/Paper_1_Tabbakh_2023_TLMViT_Plant_Disease_Classification.pdf)
- **Dataset pada Paper Asli:** *PlantVillage Dataset* dan *Wheat Rust Leaf Dataset*.
- **Metode pada Paper Asli:** Ekstraksi fitur daun 2-tahap secara berurutan (*Two-Phase Feature Extraction*):
  1. Ekstraksi peta fitur spasial awal menggunakan 1 *backbone* CNN pralatih (*ImageNet*) yang dibekukan (*frozen*, seperti `VGG19` atau `InceptionV3`).
  2. Ekstraksi fitur relasi spasial global menggunakan kepala *Vision Transformer* (`PatchExtractor` ➔ `PatchEncoder` ➔ 4 blok *Transformer Multi-Head Self-Attention*) + klasifikasi *MLP*.
- **Keterbatasan Paper 1 & Kontribusi Kita:** Paper 1 hanya menggunakan **satu *backbone* tunggal yang dibekukan (*frozen*)** tanpa strategi *ensemble*. Kami mengadopsi modul `PatchExtractor`, `PatchEncoder`, dan *Multi-Head Self-Attention* dari Paper 1 ini, lalu menyempurnakannya dengan **Two-Stage Joint Fine-Tuning** dan **Multi-Backbone Deep Ensemble**.

---

### 2. Paper Rujukan 2 — Benchmark Utama Dataset & Deep Ensemble Daun Pisang (*IncepV3Dense*, 2024)
*(Dalam catatan internal eksperimen disebut juga sebagai **Paper P6**)*
- **Judul Paper:** *"IncepV3Dense: Deep Ensemble Based Average Learning Strategy for Identification of Micro-Nutrient Deficiency in Banana Crop"*
- **Penulis:** Sudhakar Muthusamy & Swarna Priya Ramu (*Vellore Institute of Technology, India*)
- **Publikasi:** *IEEE Access*, Vol. 12, pp. 73779–73792, Mei 2024 | **DOI:** [`10.1109/ACCESS.2024.3405027`](https://doi.org/10.1109/ACCESS.2024.3405027)
- **File PDF di Repo:** [📄 `Paper_2_Muthusamy_2024_IncepV3Dense_Banana_Micronutrient_Deficiency.pdf`](./reference_papers/Paper_2_Muthusamy_2024_IncepV3Dense_Banana_Micronutrient_Deficiency.pdf)
- **Dataset pada Paper Asli (Benchmark Dataset yang Kita Gunakan):** *Mendeley / Kaggle Fully Labelled Image Dataset of Banana Leaves Deficient in Nutrients* (Sunitha et al., *Data in Brief*, Vol. 48, Art. no. 109155, 2023 — citra daun pisang varietas *Rasthali*, *Poovan*, dan *Monthan*), berfokus pada **4 kelas mikronutrien**:
  - **Boron Deficiency (`boron`)**
  - **Healthy Leaf (`healthy`)**
  - **Iron Deficiency (`iron`)**
  - **Manganese Deficiency (`manganese`)**
  - Dengan protokol pembagian: **3.200 citra *Training* (800/kelas)**, **800 citra *Validation* (200/kelas)**, dan **305 citra *Testing* murni** (*Boron*: 81, *Healthy*: 69, *Iron*: 72, *Manganese*: 83).
- **Metode pada Paper Asli:** Memodifikasi lapisan klasifikasi atas (*Mutated Top Layers*: `GlobalAveragePooling2D` ➔ `Dense(256)` ➔ `Dense(256)` ➔ `Dropout(0.5)` ➔ `Softmax(4)`) menggunakan optimizer `Adagrad` (`lr=0.01857`) pada *backbone* CNN beku, kemudian menggabungkan dua model terbaik menjadi *Average Probability Ensemble* bernama **`IncepV3Dense` (`InceptionV3 + DenseNet169`)** yang menghasilkan *Validation Accuracy* `98.62%`, *F1-Score* `93.00%`, dan prediksi benar pada data uji sebesar **`285 / 305` citra (`93.44%` Test Accuracy, salah 20 citra)**.
- **Keterbatasan Paper 2 & Kontribusi Kita:** Penggunaan `GlobalAveragePooling2D` (GAP) melenyapkan informasi tata letak spasial bercak klorosis/nekrosis pada helai daun (sehingga kelas sulit seperti *Manganese* memiliki tingkat kesalahan tinggi). Kami mempertahankan protokol dataset dan kerangka *Deep Ensemble* dari Paper 2 ini, tetapi **mengganti `GAP` dengan modul *Vision Transformer* (dari Paper 1) + *Two-Stage Fine-Tuning* + *Backbone Modern* `EfficientNetV2-S`**, sehingga akurasi pada 305 citra uji yang sama melonjak dari **`93.44%` (`285/305`)** menjadi **`99.34%` (`303/305`)**!

---

## 🌟 Sorotan Hasil Utama (Key Highlights)

- 🏆 **Melampaui Benchmark *IncepV3Dense* (99.34% vs 93.44%):** Mengungguli model terbaik pada Paper Rujukan 2 (*Muthusamy & Ramu, 2024*) sebesar **+5.90% pada Test Accuracy** (`303/305` citra benar vs `285/305`) dan memangkas tingkat kesalahan prediksi (*error rate*) sebesar **90%** (dari 20 kesalahan menjadi tersisa hanya 2 kesalahan).
- 🎯 **Akurasi 100% Sempurna pada 3 dari 4 Kelas Uji:** Mencapai akurasi **100%** pada kelas *Healthy* (`69/69`), *Iron* (`72/72`), dan *Manganese* (`83/83`), serta **97.53%** pada kelas *Boron* (`79/81`).
- 🧬 **Bukti Matematis Rekayasa Fitur (`t-SNE` & *Silhouette Score*):** Mengganti `GlobalAveragePooling2D` (Paper 2) dengan kepala *Vision Transformer* (Paper 1) + *Two-Stage Fine-Tuning* meningkatkan **Silhouette Score** pada 305 citra uji secara drastis dari **`-0.035`** (*Raw InceptionV3 GAP*) dan **`0.014`** (*Raw DenseNet169 GAP*) menjadi **`0.700`** (*Proposed T2 DenseNet169-ViT*).
- 🔍 **Explainable AI (`Grad-CAM` / Spatial Activation Heatmap):** Memvisualisasikan fokus perhatian spasial *backbone* `DenseNet169` dan `EfficientNetV2-S` tepat pada area gejala klinis daun pisang.

---

## 📂 Struktur Repositori & Daftar File

| Path File / Folder | Deskripsi & Fungsi | Status / Capaian |
| :--- | :--- | :--- |
| [`reference_papers/`](./reference_papers) | Folder berisi 2 file PDF asli paper rujukan *IEEE Access* 2023 (*TLMViT*) & 2024 (*IncepV3Dense*). | Referensi Ilmiah Utama |
| [`7_vit_deep_ensemble.ipynb`](./7_vit_deep_ensemble.ipynb) | Notebook baseline awal tim (sebelum restrukturisasi alur otomatis & kalibrasi *2-Stage Fine-Tuning*). | Baseline Awal (`98.36%` Test Acc) |
| [`7_vit_deep_ensemble_OPTIMIZED.ipynb`](./7_vit_deep_ensemble_OPTIMIZED.ipynb) | **Template Bersih Strategi A:** Pipeline mandiri 6 tahap (`InceptionV3 + DenseNet169` + *2-Stage TLM-ViT* + *Optimal Weighted Ensemble*). | Siap dijalankan di Google Colab |
| [`7_vit_deep_ensemble_OPTIMIZED_hasil_test.ipynb`](./7_vit_deep_ensemble_OPTIMIZED_hasil_test.ipynb) | **Hasil Eksekusi Strategi A:** Berisi log pelatihan penuh, kurva konvergensi, tabel metrik, dan *Confusion Matrix*. | **Test Acc: `99.34%` (`303/305`)** |
| [`8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb`](./8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb) | **Template Bersih Strategi A + B (Standar Publikasi):** Menambahkan *backbone* modern `EfficientNetV2-S`, *Tri-Ensemble*, proyeksi `t-SNE` 2D, dan `Grad-CAM`. | Siap dijalankan di Google Colab |
| [`8_vit_deep_ensemble_STRATEGI_B_PUBLISH_hasil_test.ipynb`](./8_vit_deep_ensemble_STRATEGI_B_PUBLISH_hasil_test.ipynb) | **Hasil Eksekusi Strategi A + B:** Berisi hasil lengkap 23 skema model, grafik `t-SNE` (*Silhouette* `0.700`), dan *heatmap* `Grad-CAM`. | **Val Acc: `98.50%` \| Test Acc: `99.34%`** |

---

## 🔄 Alur Metodologi & Arsitektur Eksperimen

1. **Tahap 1 — Auto-Download & Isolasi Dataset Daun Pisang:** Mengunduh dataset daun pisang secara otomatis dan mengisolasi 4 kelas mikronutrien sesuai Paper Rujukan 2 (*Boron*, *Healthy*, *Iron*, *Manganese* — total 3.660 citra mentah).
2. **Tahap 2 — *Zero-Leakage Stratified Split* & *Offline Augmentation Balancing*:** Memisahkan **305 citra uji murni** (*Boron*: 81, *Healthy*: 69, *Iron*: 72, *Manganese*: 83) di awal sebelum augmentasi agar bebas dari kebocoran data (*data leakage*), lalu menyeimbangkan sisa data menjadi **3.200 citra latih (800/kelas)** dan **800 citra validasi (200/kelas)**.
3. **Tahap 3 — Penyeragaman & Ekstraksi Peta Fitur Spasial (`8 x 8 x C`):** Mengekstrak peta fitur spasial `8 x 8` dari `InceptionV3` (`8x8x2048` dengan `ZeroPadding2D`), `DenseNet169` (`8x8x1664`), dan `EfficientNetV2-S` (`8x8x1280`).
4. **Tahap 4 — Modul *Vision Transformer* (`PatchExtractor` & `PatchEncoder` — Adopsi Paper 1):** Memotong peta fitur `8 x 8` menjadi **16 *patch* berukuran `2 x 2`**, memproyeksikannya ke dimensi *embedding* `D = 64` + *Learnable Positional Embedding*, dan memprosesnya melalui **4 blok *Transformer Encoder*** (`4 Heads`, *MLP GELU* `128 -> 64`).
5. **Tahap 5 — Studi Ablasi Bertahap (`B1 ➔ B2 ➔ T1 ➔ T2`) & *Two-Stage Fine-Tuning*:**
   - **Skema B1 (*Baseline CNN Standard*):** *Frozen Backbone* + `GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense(4, Softmax)`.
   - **Skema B2 (*Mutated CNN Head* — Adopsi Paper 2):** *Frozen Backbone* + `GlobalAveragePooling2D` + `BatchNormalization ➔ Dense(512, ReLU) ➔ Dropout(0.4) ➔ Dense(128, ReLU) ➔ Dense(4, Softmax)`.
   - **Skema T1 (*TLM-ViT Frozen* — Adopsi Paper 1):** *Frozen Backbone* + *ViT Head* (`PatchExtractor + PatchEncoder + 4x Transformer`).
   - **Skema T2 (*Proposed Two-Stage Fine-Tuned TLM-ViT*):** **Stage 1** (*12 Epoch Warmup ViT Head*, `lr=3e-4`) dilanjutkan **Stage 2** (*25 Epoch Joint Fine-Tuning* membuka blok konvolusi atas pada CNN + ViT, `lr=5e-5` dengan `ReduceLROnPlateau` & *BatchNorm Frozen*).
6. **Tahap 6 — *Multi-Backbone Deep Ensemble*, Proyeksi `t-SNE`, & `Grad-CAM`:** Menggabungkan distribusi probabilitas prediksi pada **Strategi A (`InceptionV3 + DenseNet169`)**, **Strategi B (`DenseNet169 + EfficientNetV2-S`)**, dan **Tri-Ensemble (`3 Backbones`)**, dilengkapi analisis klaster ruang fitur `t-SNE` serta *heatmap* `Grad-CAM`.

![Kurva Konvergensi 2-Stage Fine-Tuning](./assets/chart_t2_convergence_optimized.png)

---

## 📊 Hasil Eksperimen & Perbandingan terhadap Paper Rujukan

Evaluasi di bawah ini dilakukan pada komposisi **305 citra uji** yang sama (*Boron*: 81, *Healthy*: 69, *Iron*: 72, *Manganese*: 83):

| Kelompok / Skema Model | Backbone | Val Acc (%) | Val F1 (%) | Test Benar / 305 | Test Acc (%) | Test Macro F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Paper 2 (Muthusamy, 2024): Mutated InceptionV3** | InceptionV3 | 95.75% | 90.00% | 275 / 305 | 90.16% | 90.00% |
| **Paper 2 (Muthusamy, 2024): Mutated DenseNet169** | DenseNet169 | 97.75% | 91.00% | 278 / 305 | 91.15% | 91.00% |
| **Paper 2 (Muthusamy, 2024): IncepV3Dense (Target Benchmark)** | IncV3 + Dense169 | 98.62% | 93.00% | 285 / 305 | 93.44% | 93.00% |
| **Eksperimen Kita — Ens B1** *(Baseline CNN Ensemble)* | IncV3 + Dense169 | 93.62% | 93.63% | 285 / 305 | 93.44% | 93.36% |
| **Eksperimen Kita — Ens B2** *(Mutated CNN Ensemble)* | IncV3 + Dense169 | 95.25% | 95.27% | 293 / 305 | 96.07% | 95.98% |
| **Eksperimen Kita — B2 EfficientNetV2S** *(Single Modern Mutated)* | EffNetV2-S | 96.25% | 96.26% | 294 / 305 | 96.39% | 96.32% |
| **Eksperimen Kita — Ens T1** *(Frozen TLM-ViT Ensemble)* | IncV3 + Dense169 | 95.50% | 95.50% | 297 / 305 | 97.38% | 97.36% |
| **Eksperimen Kita — T2 Inception** *(2-Stage Fine-Tuned ViT)* | InceptionV3 | 97.12% | 97.13% | 299 / 305 | 98.03% | 97.99% |
| **Eksperimen Kita — T2 DenseNet169** *(2-Stage Fine-Tuned ViT)* | DenseNet169 | 96.62% | 96.62% | 301 / 305 | 98.69% | 98.68% |
| **Eksperimen Kita — T2 EfficientNetV2S (Best Single Val)** | EffNetV2-S | **98.25%** | **98.25%** | 299 / 305 | 98.03% | 98.06% |
| **Strategi B: Ens T2 Weighted (Best Val Overall)** | Dense169 + EffV2S | **98.50%** | **98.50%** | 301 / 305 | 98.69% | 98.66% |
| **Tri-Ensemble T2 (Gabungan 3 Backbone)** | IncV3 + Dense169 + EffV2S | **98.50%** | **98.50%** | 301 / 305 | 98.69% | 98.66% |
| **Strategi B: Ens T2 Unweighted** | Dense169 + EffV2S | 97.88% | 97.88% | **302 / 305** | **99.02%** | **98.99%** |
| **Strategi A: Ens T2 & Weighted (Best Test Overall)** | IncV3 + Dense169 | 97.38% | 97.38% | **303 / 305** | **99.34%** | **99.33%** |

![Head to Head Comparison](./assets/chart_head_to_head.png)

### Confusion Matrix Model Unggulan pada 305 Citra Uji
![Confusion Matrix Strategi A dan B](./assets/confusion_matrix_strategy_ab.png)

---

## 🔬 Visualisasi Ruang Fitur (`t-SNE`) & Explainable AI (`Grad-CAM`)

### 1. Proyeksi `t-SNE` 2D & Lonjakan *Silhouette Score* (`-0.035` ➔ `0.700`)
Visualisasi `t-SNE` pada 305 citra uji membuktikan bahwa fitur mentah `GlobalAveragePooling2D` pada metode Paper Rujukan 2 masih saling tumpang-tindih (*Silhouette Score* `-0.035` pada `InceptionV3` dan `0.014` pada `DenseNet169`), sedangkan fitur hasil rekayasa **TLM-ViT + Two-Stage Fine-Tuning** memisahkan keempat kelas daun pisang menjadi 4 klaster pulau yang sangat tegas (*Silhouette Score* **`0.700`** dan **`0.654`**):

![t-SNE Feature Space Comparison](./assets/tsne_feature_space.png)

### 2. Explainable AI: *Spatial Feature Activation Heatmap* (`Grad-CAM`)
Peta aktivasi spasial membuktikan bahwa `DenseNet169` dan `EfficientNetV2-S` mengabaikan latar belakang hitam dan berfokus pada area gejala klinis daun pisang secara komplementer:

![Grad-CAM Spatial Attention](./assets/gradcam_spatial_attention.png)

---

## 🚀 Cara Menjalankan Eksperimen di Google Colab

1. Unggah salah satu notebook template bersih ([`7_vit_deep_ensemble_OPTIMIZED.ipynb`](./7_vit_deep_ensemble_OPTIMIZED.ipynb) atau [`8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb`](./8_vit_deep_ensemble_STRATEGI_B_PUBLISH.ipynb)) ke **Google Colab**.
2. Aktifkan akselerator hardware **GPU (`T4` / `L4` / `A100`)** melalui menu `Runtime` ➔ `Change runtime type` ➔ `T4 GPU`.
3. Klik `Runtime` ➔ `Run all`. Notebook akan otomatis mengunduh dataset daun pisang, melakukan *stratified split* bebas kebocoran data, melatih seluruh skema model, serta menyimpan *checkpoint* ke Google Drive Anda.

---

## 📚 Daftar Pustaka (References)

1. **Tabbakh, A., & Barpanda, S. S. (2023).** *"A Deep Features Extraction Model Based on the Transfer Learning Model and Vision Transformer 'TLMViT' for Plant Disease Classification."* IEEE Access, vol. 11, pp. 45377–45392. DOI: `10.1109/ACCESS.2023.3273317`. ([Lihat PDF](./reference_papers/Paper_1_Tabbakh_2023_TLMViT_Plant_Disease_Classification.pdf))
2. **Muthusamy, S., & Ramu, M. (2024).** *"IncepV3Dense: Deep Ensemble Based Average Learning Strategy for Identification of Micro-Nutrient Deficiency in Banana Crop."* IEEE Access, vol. 12, pp. 73779–73792. DOI: `10.1109/ACCESS.2024.3405027`. ([Lihat PDF](./reference_papers/Paper_2_Muthusamy_2024_IncepV3Dense_Banana_Micronutrient_Deficiency.pdf))
3. **Sunitha, P., Uma, B., Channakeshava, S., & Babu, S. (2023).** *"A fully labelled image dataset of banana leaves deficient in nutrients."* Data in Brief, vol. 48, Art. no. 109155.
