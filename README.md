# MARINE DEBRIS & OIL SPILL DETECTION MODEL
*A Deep Learning Framework for Multi-Class Ocean Pollutant Segmentation*

---

## 📌 Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Phase Breakdown](#phase-breakdown)
   - [1) Encoder Architecture (MiT-B3)](#1-encoder-architecture-mit-b3)
   - [2) Decoder Architecture](#2-decoder-architecture)
   - [3) Dual-Phase Optimization (Novelty)](#3-dual-phase-optimization-novelty)
   - [4) VSCP Augmentation](#4-vscp-augmentation)
   - [5) Scene-Aware Data Splitting](#5-scene-aware-data-splitting)
4. [Stabilization Techniques](#stabilization-techniques)
5. [Results](#results)
6. [Integrated Pipeline](#integrated-pipeline)

---

## Executive Summary
This project develops a high-accuracy, efficient framework for **multi-class ocean pollutant segmentation** using satellite imagery.  
The goal is to detect:

- plastic debris  
- oil spills  
- algae & natural organic material  
- foam  
- jellyfish  
- ships & ship wakes  

The system uses an **Enhanced SegFormer (MiT-B3)** architecture with three key contributions:

1. Dual-Phase Optimization  
2. VSCP Augmentation  
3. Scene-Aware Data Splitting  

Stabilization through **EMA** and **TTA** improves robustness.  
Final performance:

- **mIoU:** 0.8281  
- **Mean F1:** 0.8951  

---

## Project Overview

### 2.1 Problem Statement
Ocean pollutants visually resemble natural structures (waves, foam, shadows), making segmentation challenging.  
CNN models lack global context and fail to generalize to all pollutant types.  
Challenges include:

- severe class imbalance  
- satellite-scene variability  
- data leakage from random patch splitting  

### 2.2 Core Components
1. **SegFormer-MiT-B3 Backbone**  
2. **Custom Decoder for Multi-Scale Fusion**  
3. **Training Innovations**  
   - Weighted CE → Lovász  
   - VSCP augmentation  
   - Scene-aware splitting  
4. **Stabilization & Inference Enhancements**  
   - EMA  
   - TTA  

---

## Phase Breakdown

### 1) Encoder Architecture (MiT-B3)
Four stages extract hierarchical features:

- **Stage 1:** fine debris, thin wakes  
- **Stage 2–3:** algae streaks, turbidity textures  
- **Stage 4:** global wave–pollutant context  

SRA attention enables efficient global reasoning.

---

### 2) Decoder Architecture
- MLP equalization  
- Upsampling of all encoder stages  
- Multi-scale fusion via concat + MLP  
- 1×1 classification head  
- Final upsampling to **240×240**  

Preserves both fine debris structures and large oil slicks.

---

### 3) Dual-Phase Optimization (Novelty)
**Phase 1:** Weighted Cross-Entropy → forces rare-class detection  
**Phase 2:** Lovász-Softmax → directly optimizes IoU  

Improves mIoU by ~18%.

---

### 4) VSCP Augmentation
VSCP extracts salient pollutant regions (plastic, oil, algae) and pastes them into new backgrounds.  
Ensures minority classes appear in every training batch.  
Boosts Plastic F1 to ~84%.

---

### 5) Scene-Aware Data Splitting
Dataset is split by **Satellite Scene ID**, preventing leakage and ensuring true spatial generalization.

---

## Stabilization Techniques
- **EMA:** smooths weight updates → stable boundaries  
- **TTA:** flip/rotation ensemble → robust predictions  

---

## Results

### 5.1 Quantitative
- **mIoU:** 0.8281  
- **Mean F1:** 0.8951  
- High IoU for major water classes (0.95–0.99)  
- Significant improvement in difficult classes:  
  - Plastic: 0.6621  
  - Natural Organic Material: 0.6592  
  - Dense Sargassum: 0.8298  
  - Ship: 0.9134

---

### 5.2 Qualitative
Model successfully segments:

- plastic specks  
- oil spill boundaries  
- ship vs wake  
- algae streaks  
- foam & organic matter  

(See report for sample output images.)

---

## Integrated Pipeline
1. **Input:** 512×512×11 → cropped to 240×240  
2. **Model:** SegFormer encoder → fusion → segmentation mask  
3. **Training Enhancements:** VSCP + dual-phase loss + EMA  
4. **Inference:** TTA → final pollutant map  

---

## Authors
**Harkirat**, **Devam**, **Ruhi**, **Vansh**

---

## License
MIT License recommended.

