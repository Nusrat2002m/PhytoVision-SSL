# 🌿 PhytoVision-SSL: A Label-Efficient Self-Supervised Computer Vision Framework for Early Plant Leaf Disease Diagnosis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**PhytoVision-SSL** is a self-supervised learning (SSL) framework designed for label-efficient plant leaf disease detection using contrastive learning (SimCLR). By leveraging unlabelled leaf image data for pretraining, this architecture achieves high classification performance on downstream tasks with only a fraction (15%) of labelled training samples.

---

## 📌 Key Features

- **Contrastive Self-Supervised Pretraining:** Utilizes **SimCLR** architecture with a ResNet-50 backbone and Normalized Temperature-scaled Cross Entropy (**NT-Xent**) loss.
- **Label-Efficient Fine-Tuning:** Achieves high downstream classification accuracy using only **15% labelled data**.
- **Data Augmentation Pipeline:** Custom SimCLR augmentation pipeline (Random Resized Crop, Color Jitter, Gaussian Blur, Random Grayscale, Flips).
- **Comprehensive Evaluation & Visualizations:**
  - Automated dataset summary and distribution analysis.
  - Fine-tuning accuracy/loss curves.
  - Multi-class Confusion Matrix heatmap.
  - Low-dimensional feature embedding visualization via **t-SNE**.

---

## 🏗️ Model Architecture & Pipeline


```

[ Unlabelled Leaf Images ]
│
▼  (Augmentations: Crop, Blur, Jitter)
(View 1)     (View 2)
│            │
▼            ▼
[ ResNet-50 Encoder Backbone ] ──► Extracted Embeddings (h) ──► Saved Encoder (.pth)
│            │                                                    │
▼            ▼                                                    ▼
[ Projection Head (MLP) ]                                   [ Linear Classifier (FC) ]
│            │                                                    │
▼            ▼                                                    ▼
(z1)          (z2)                                     [ 15% Labelled Fine-tuning ]
│            │                                                    │
└─────┬──────┘                                                    ▼
▼                                              [ Disease Classification ]
(NT-Xent Contrastive Loss)

```

---

## 📂 Project Structure

```text
├── phytovision_ssl.py             # Complete training, pretraining & evaluation pipeline
├── phytovision_ssl_encoder.pth    # Saved pretrained ResNet-50 encoder weights
├── dataset_summary_table.csv      # Dataset statistical summary
├── class_distribution.csv         # Class-wise sample counts
├── final_metrics_table.csv        # Final Accuracy, Precision, Recall, F1-Score
├── fig1_class_distribution.png    # Class distribution plot
├── fig2_sample_grid.png           # Dataset sample visualization grid
├── fig3_ssl_loss_curve.png        # Contrastive pretraining loss curve
├── fig4_finetune_accuracy.png     # Downstream fine-tuning accuracy curve
├── fig5_confusion_matrix.png      # Validation set confusion matrix
└── fig6_tsne_embeddings.png       # t-SNE plot of learned feature representations

```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/nusratjahan/PhytoVision-SSL.git](https://github.com/nusratjahan/PhytoVision-SSL.git)
cd PhytoVision-SSL
pip install -q kagglehub torch torchvision scikit-learn seaborn matplotlib pandas tqdm

```

### 2. Dataset Setup

This project uses the **New Plant Diseases Dataset** from Kaggle. The script automatically downloads and extracts the dataset using `kagglehub`.

### 3. Execution

Run the complete pipeline (Pretraining ──► Fine-Tuning ──► Evaluation):

```bash
python phytovision_ssl.py

```

---

## 📊 Experimental Results & Figures

### 1. Dataset Overview

The dataset contains multiple plant leaf classes with varying sample distributions.

| Figure 1: Top-15 Class Distribution | Figure 2: Sample Leaf Images |
| --- | --- |
|  |  |

### 2. Self-Supervised Pretraining & Fine-Tuning

Contrastive pretraining steadily reduces NT-Xent loss, allowing the encoder to learn robust visual representations before fine-tuning on 15% labelled data.

| Figure 3: Pretraining NT-Xent Loss | Figure 4: Fine-Tuning Accuracy |
| --- | --- |
|  |  |

### 3. Downstream Evaluation & Embedding Analysis

| Figure 5: Confusion Matrix | Figure 6: t-SNE Feature Embeddings |
| --- | --- |
|  |  |

---

## 📜 Citation & Reference

If you find this repository or framework useful in your research, please cite:

```bibtex
@inproceedings{jahan2026phytovision,
  title={PhytoVision-SSL: A Label-Efficient Self-Supervised Computer Vision Framework for Early Plant Leaf Disease Diagnosis},
  author={Jahan, N. and Kumar, S. and Singh, A. K. and Roy, T. C. and Mondal, P. and Haque, M. A.},
  booktitle={International Conference on Data Science and Pattern Recognition (ICDPN)},
  year={2026}
}

```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

```

```
