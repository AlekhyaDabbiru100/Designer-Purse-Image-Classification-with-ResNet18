# 👜 Designer Purse Image Classification with ResNet18

Welcome to a fun computer vision project where we teach a model to recognize designer purse styles👛 from images.

The goal is simple: **given a purse image, classify it into the correct designer/style category** - or identify it as a non-target bag using the `Negative` class.

---

## ✨ Project Overview

This project uses deep learning and transfer learning to classify lady purse images into **7 categories**:

**💼Bottega Veneta Mini Jodie, Chanel 255, Fendi Baguette, Gucci Jackie Hobo, Lady Dior, Prada Cleo, and Negative👛.**

The `Negative` class represents other bag styles that are not part of the six designer purse categories. The project covers the full computer vision workflow: exploring the dataset, trainig a baseline ResNet18 model, finetuning it for a better performance, evaluating the results, and adding image similarity search

---

## 📦 Dataset

The dataset comes from Kaggle:

**Bags Classification Dataset**  
https://www.kaggle.com/datasets/ravirajsinh45/bags-classification

The dataset contains **2,535 images**🛍️ across 7 folders:

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

📁 Data Split

The dataset was split into training, validation, and test sets.

Split	Images
Training	1,774
Validation	380
Test	381

The split was stratified so each class appeared across the training, validation, and test sets.

---

🛠️ Tech Stack
Python 🐍
PyTorch
Torchvision
pandas + NumPy
scikit-learn
Matplotlib
Pillow
tqdm

---

## 📌 What We Did

This project built a computer vision model🧠 to classify designer purse images into seven categories.

Main steps:

- Collected the purse image dataset from Kaggle💼
- Organized images into labeled classes
- Created a train, validation, and test split
- Performed basic exploratory data analysis
- Built a baseline ResNet18 image classifier🎒
- Used transfer learning with pretrained ImageNet weights
- Trained the baseline model with a frozen backbone
- Fine-tuned the final ResNet18 block for better performance
- Evaluated the model using accuracy, precision, recall, and F1-score
- Created confusion matrices to inspect model mistakes
- Checked wrong predictions to understand weak classes
- Added image similarity search using CNN feature embeddings

The project showed that fine-tuning improved model performance and made the classifier better at recognizing purse styles.

## 🔍 Image Similarity Search

Main steps:

- Used the fine-tuned ResNet18 model as a feature extractor
- Converted each image into a 512-dimensional feature vector
- Built a feature bank for all dataset images
- Compared image embeddings using similarity scores
- Returned the most visually similar purse images 👛 for a selected query image

I think this has made the project more interactive because the model could classify images and also find similar-looking bags.

## 📊 Results

Fine-tuning improved the model compared with the baseline.

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| Frozen ResNet18 Baseline | 53.81% | 56.74% |
| Fine-Tuned ResNet18 | 62.99% | 65.61% |

The final model performed better after unfreezing and fine-tuning the last ResNet18 block.

## 🏆 Best Performing Class

The best performing class was **Bottega Veneta Mini Jodie**.💼

| Metric | Score |
|---|---:|
| Precision | 0.91 |
| Recall | 0.88 |
| F1-score | 0.89 |

This class likely performed well because its purse style🛍️ was more visually distinct than some of the other categories.

## ✅ Main Takeaways

The fine-tuned model performed better than the baseline model.

Main takeaways:

- Transfer learning worked well for this image classification task
- Fine-tuning improved accuracy and Macro F1 score
- The model performed best on visually distinct purse styles
- The `Negative` class was the hardest because it contains many different bag types
- Macro F1 score was more useful than accuracy because the dataset was imbalanced

Overall, the project produced a solid computer vision baseline for designer purse classification.

## 🔮 Future Recommendations

This project can be improved in several ways:

- Add more diverse images to improve model generalization
- Clean mislabeled or low-quality images
- Improve the `Negative` class with better examples
- Try stronger models such as ResNet50, EfficientNet, or ConvNeXt
- Use stronger image augmentation
- Tune learning rate, batch size, and number of epochs
- Build a simple demo app for image upload and prediction🎒
- Improve similarity search by filtering duplicates and showing top recommendations

These improvements could make the model more accurate and more useful in a real-world setting.


## 🎯💄👠 Note from the Author

As someone fascinated by fashion 👗✨, I created this project to explore how computer vision can recognize designer purse styles 👜🤖.

This project is dedicated to *The Devil Wears Prada* 👠❤️🎬 - a tribute to fashion, confidence, and iconic style 💃✨.

It helped me practice transfer learning, model evaluation, and image similarity search using PyTorch 🧠📊🔍.
