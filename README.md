# 👜 Designer Purse Image Classification with ResNet18

Welcome to a fun computer vision project where we teach a model to recognize designer purse styles from images.

The goal is simple: **given a purse image, classify it into the correct designer/style category** — or identify it as a non-target bag using the `Negative` class.

---

## ✨ Project Overview

This project uses deep learning and transfer learning to classify lady purse images into **7 categories**:

| Class | Description |
|---|---|
| Bottega Veneta Mini Jodie | Designer purse style |
| Chanel 255 | Designer purse style |
| Fendi Baguette | Designer purse style |
| Gucci Jackie Hobo | Designer purse style |
| Lady Dior | Designer purse style |
| Prada Cleo | Designer purse style |
| Negative | Other bag styles not part of the six designer categories |

The project starts with exploratory data analysis, trains a baseline ResNet18 model, improves it using fine-tuning, evaluates performance with classification metrics, and adds a small image similarity search feature.

---

## 📦 Dataset

The dataset comes from Kaggle:

**Bags Classification Dataset**  
https://www.kaggle.com/datasets/ravirajsinh45/bags-classification

The dataset contains **2,535 images** across 7 folders:

| Category | Image Count |
|---|---:|
| Bottega Veneta Mini Jodie | 159 |
| Chanel 255 | 431 |
| Fendi Baguette | 257 |
| Gucci Jackie Hobo | 303 |
| Lady Dior | 520 |
| Prada Cleo | 276 |
| Negative | 589 |

The `Negative` folder contains other bag styles that are not part of the target designer classes. This helps the model learn when an image does **not** belong to one of the six designer purse categories.

---


