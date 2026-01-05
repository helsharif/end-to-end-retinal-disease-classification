---
title: Retinal Disease Classifier
emoji: 👁️
colorFrom: red
colorTo: red
sdk: docker
app_port: 7860
tags:
- streamlit
pinned: false
short_description: Deep learning application for retinal disease classification
license: apache-2.0
---

## 🚀 Project Highlights

- **Task:** Multi-class retinal disease classification
- **Diseases:** Cataract, Diabetic Retinopathy, Glaucoma, Normal
- **Frameworks:** TensorFlow (Keras) & PyTorch
- **Models:** EfficientNet (transfer learning)
- **Deployment:** Docker + Streamlit
- **Input Size:** 456 × 456 retinal fundus images

This repository demonstrates a **production-oriented ML workflow**—from experimentation and reproducible training to a deployable inference application.

---

## 🩺 Motivation & Problem Statement

Retinal diseases such as cataracts, diabetic retinopathy, and glaucoma are leading causes of preventable blindness. Early detection via retinal imaging significantly improves outcomes, but manual review is:

- Time-consuming
- Subjective
- Dependent on specialist availability

**Goal:** Build an accurate, reproducible, and deployable computer vision system that can classify retinal images into common disease categories and support early screening workflows.

---

## 🧠 Model Development

### Data
- 4,000+ labeled retinal fundus images
- Standardized to **456×456 px**
- Stratified train / validation / test splits

### Preprocessing
- Image decoding and normalization
- Aspect-ratio–preserving square padding
- Resize using area interpolation

### Architecture
- **EfficientNet backbone** (ImageNet pretraining)
- Custom classification head with:
  - Global average pooling
  - Dense layers + dropout
  - Softmax output

### Training Strategy
- Stage 1: Freeze backbone, train classifier head
- Stage 2: Fine-tune upper EfficientNet layers
- Loss: Cross-entropy
- Optimizer: Adam
- Class imbalance handled via class weighting

---

## 📊 Evaluation

Evaluation was performed on a held-out test set and includes:

- Overall accuracy
- Per-class precision, recall, and F1
- Confusion matrices
- Learning curves


---

## 🖥️ Streamlit Inference App

A lightweight Streamlit web application allows users to:

- Upload a single retinal image
- Run real-time inference
- View predicted class and confidence
- Inspect full class probability distribution

The app automatically loads the **latest saved TensorFlow model** and applies the same preprocessing pipeline used during training.

---

## 🐳 Dockerized Deployment

The application is fully containerized using Docker for reproducibility and portability.

