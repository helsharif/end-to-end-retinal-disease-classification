# End-to-End Retinal Disease Classification

**End-to-end deep learning system for retinal disease classification**, leveraging CNN-based architectures in **TensorFlow and PyTorch**. The project covers model development, evaluation, and **containerized deployment via Docker** with a **Streamlit web application** for interactive, real-time inference.

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

Evaluation is performed on a held-out test set and includes:

- Overall accuracy
- Per-class precision, recall, and F1
- Confusion matrices
- Learning curves

Example outputs are stored as CSV files for reproducibility and inspection.

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

### Build and Run

```bash
# Build image
docker compose build

# Run app
docker compose up
```

Then open:

```
http://localhost:8502
```

---

## 📁 Repository Structure

```
.
├── images/                         # Sample images and figures
├── logs/                           # Training and experiment logs
├── saved_models_pytorch/           # Trained PyTorch models
├── saved_models_tensorflow/        # Trained TensorFlow models
├── secrets/                        # (Ignored) credentials / local configs
├── streamlit_app/                  # Streamlit application
│   ├── app.py
│   ├── class_names.json
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml              # Container orchestration
├── env003.yml / env004.yml         # Conda environments
├── e01–e04_*.ipynb                 # Model development notebooks
├── test_results_with_predictions_*.csv
└── README.md
```

---

## 🛠️ Tech Stack

- **Languages:** Python
- **Deep Learning:** TensorFlow (Keras), PyTorch
- **Computer Vision:** EfficientNet, transfer learning
- **Data:** NumPy, Pandas
- **Visualization:** Matplotlib, Plotly
- **Deployment:** Docker, Streamlit

---

## 🔍 Reproducibility & Engineering Practices

- Fixed random seeds
- Explicit dataset splits
- Modular preprocessing and inference code
- Cached model loading for efficient serving
- Clear separation of training and deployment artifacts

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only** and is **not intended for clinical diagnosis**. Any real-world medical deployment would require rigorous validation and regulatory approval.

---

## 👤 Author

**Husayn El Sharif**  
PhD, Civil & Environmental Engineering (Hydrology)  
Senior Data Scientist / Machine Learning Engineer

---

## 📌 Portfolio Relevance

This project highlights:

- End-to-end ML system design
- Computer vision in medical imaging
- Framework interoperability (PyTorch & TensorFlow)
- Model deployment and containerization
- Production-oriented ML engineering practices

Highly relevant for **Data Scientist**, **ML Engineer**, and **Applied AI** roles.

