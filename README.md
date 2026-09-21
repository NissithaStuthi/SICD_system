# 🛰️ Satellite Image Change Detection (SICD) System

An end-to-end Deep Learning and Geospatial pipeline for high-resolution Satellite Image Change Detection built with PyTorch, OpenCV, Rasterio, GeoPandas, and Albumentations. This repository implements a **Hybrid CNN + Vision Transformer Siamese U-Net** featuring Spatial-Channel Attention mechanisms, 5-Class Semantic Change Analysis, Grad-CAM explainability, GIS GeoTIFF raster processing, GeoJSON vector exports, and an automated production pipeline.

---

## 📌 System Architecture & Pipeline

`	ext
  [Date 1 Image (T1)]           [Date 2 Image (T2)]
           │                             │
           ▼                             ▼
┌────────────────────┐        ┌────────────────────┐
│ Preprocessing &    │        │ Preprocessing &    │
│ ORB Registration   │        │ Spatial Alignment  │
└──────────┬─────────┘        └──────────┬─────────┘
           │                             │
           └──────────────┬──────────────┘
                          ▼
            ┌──────────────────────────┐
            │  Siamese ResNet Encoder  │
            └─────────────┬────────────┘
                          ▼
            ┌──────────────────────────┐
            │ ViT Bottleneck +         │
            │ Spatial-Channel Attention│
            └─────────────┬────────────┘
                          ▼
            ┌──────────────────────────┐
            │      U-Net Decoder       │
            └─────────────┬────────────┘
                          ▼
            ┌──────────────────────────┐
            │ 5-Class Semantic Mask &  │
            │ Grad-CAM Explainability  │
            └─────────────┬────────────┘
                          ▼
            ┌──────────────────────────┐
            │ Geospatial Raster/Vector │
            │ (GeoTIFF / GeoJSON / m²) │
            └──────────────────────────┘
`

---

## 🚀 Key Features Across Modules

1. **Dataset Pipeline (Modules 1–3):** Standardized LEVIR-CD+ dataset handling, spatial resizing ( \times 256$), and train/val/test partitioning.
2. **Traditional Baseline (Module 4):** Change Vector Analysis (CVA) baseline for benchmark comparisons.
3. **Hybrid Architecture (Modules 5–9):** Siamese ResNet/U-Net backbone combined with **Transformer Bottlenecks** and **Spatial-Channel Attention Gates**.
4. **Composite Training Engine (Module 10):** Optimized using AdamW, Cosine Annealing, and a composite loss function (Loss = BCE + Dice + Focal).
5. **Multi-Class Change Analysis (Module 12):** Categorizes changes into **Building**, **Vegetation**, **Road**, and **Water/Flood** classes.
6. **Model Explainability (Module 13):** Integrated **Grad-CAM** feature activation maps and prediction confidence heatmaps.
7. **Real-World Pipeline (Module 14):** Automated **ORB feature registration**, tiled scene reconstruction, morphological post-processing, and surface area change percentage analytics.
8. **Geospatial GIS Analysis (Module 15):** GeoTIFF conversion, CRS projections (EPSG:4326 to UTM Zone 43N), and GeoJSON vector polygon extractions.
9. **Final System Integration (Module 16):** Master 8-panel dashboard (main.py) executing complete end-to-end evaluation and baseline comparisons.

---

## 📂 Project Structure

`	ext
SICD_system/
│
├── data/
│   └── dataset_split/
│       ├── train/       # 689 image pairs & masks
│       ├── val/         # 147 image pairs & masks
│       └── test/        # 149 image pairs & masks
│
├── src/
│   ├── dataset.py       # PyTorch Dataset loader
│   └── model.py         # Hybrid CNN-Transformer Siamese U-Net
│
├── train_multiclass.py  # Multi-class model training engine
├── evaluate_multiclass.py # Visual evaluator & metric calculator
├── explain.py           # Module 13 Grad-CAM & Error Analysis
├── pipeline.py          # Module 14 Real-World Production Pipeline
├── geospatial.py        # Module 15 GeoTIFF & GeoJSON Vector Engine
├── main.py              # Module 16 Master Integrated Pipeline
├── predict.py           # Single-command batch inference script
├── app.py               # Interactive Gradio Web Dashboard
└── README.md            # Repository documentation
`

---

## ⚡ Execution Commands

* **Run Master System Integration:**
  `powershell
  python3 main.py
  `
* **Run Geospatial Vector Analysis:**
  `powershell
  python3 geospatial.py
  `
* **Run Production Pipeline & Area Analytics:**
  `powershell
  python3 pipeline.py
  `
* **Run Grad-CAM Explainability:**
  `powershell
  python3 explain.py
  `
* **Launch Interactive Web Dashboard:**
  `powershell
  python3 app.py
  `
