# Pneumonia Detection from Chest X-Rays

Pneumonia is a serious lung infection that can be diagnosed through chest X-ray imaging. In this project,
you will build an AI model capable of detecting pneumonia from X-ray images using deep learning. The
aim is to explore how computer vision techniques can assist radiologists in diagnosing pneumonia
accurately and efficiently.

---

## Objective

Develop and evaluate a convolutional neural network (CNN)-based model that classifies chest X-ray images as either:

- **NORMAL**
- **PNEUMONIA**

---

## Dataset

- **Name:** Chest X-Ray Images (Pneumonia) by Kermany et al.
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Description:**  
  ~5,000 labeled X-ray images categorized into `NORMAL` and `PNEUMONIA`, organized into:
  - `train/`
  - `val/`
  - `test/`

---

## Tasks

### 1. Understand the Dataset
- Load and visualize X-ray samples
- Analyze class distribution

### 2. Data Preprocessing
- Resize, normalize, and augment images
- Use `ImageDataGenerator` for loading and augmentation

### 3. Model Development
- **Custom CNN**: A scratch-built convolutional network
- **Transfer Learning**: Using `ResNet50` as a base model
- Apply class balancing and regularization
- Compare the performance of both models

### 4. Evaluation
- Metrics: **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**
- Plot training vs. validation **accuracy/loss curves**

### 5. Interpretability (XAI)
- Apply **Grad-CAM** to visualize model attention
- Highlight regions the model uses to predict pneumonia

---

## How to Run

1. Download the dataset from Kaggle and unzip it into your working directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python main.py
   ```
4. View evaluation metrics and Grad-CAM visualizations.
