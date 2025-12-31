### 1. Problem Summary

This project solves the **real-time car view classification** problem.
Given an input image, the model predicts **one of the following 6 views**:

* `front`
* `frontleft`
* `frontright`
* `rear`
* `rearleft`
* `rearright`

If the image does not match any of the above views, it is classified as **`unknown`**.

The solution is designed for **edge/mobile deployment** and exported as a **TensorFlow Lite (TFLite)** model.

---

### 2. Dataset Preparation

* Dataset annotations are provided in **VIA (VGG Image Annotator) JSON format**.
* Each image contains **polygon annotations** for multiple car parts.
* Polygon **identity labels** (e.g., `frontws`, `rearws`, `lefttaillamp`, etc.) are mapped to a **single image-level view label** using deterministic rules.
* One image may contain **multiple annotated parts**, but only **one final view label** is assigned per image.

**Note on class imbalance:**

* Some views (especially `rearleft` and `rearright`) are **under-represented** in the dataset.
* Class imbalance is handled using **class weighting during training**.
* An additional `unknown` class is included to handle non-matching or random images.

---

### 3. Data Preprocessing

* Images are resized to **224×224**
* RGB normalization (`pixel / 255.0`)
* Preprocessed images and labels are **cached to disk** to avoid recomputation and prevent RAM crashes during training.
* Train/validation split: **80% / 20% (stratified)**

---

### 4. Model Architecture

* **Backbone:** MobileNetV2 (ImageNet pretrained)
* **Why MobileNetV2?**

  * Lightweight and fast
  * Designed for mobile and edge devices
  * Lower memory footprint compared to VGG16 or ResNet50
* **Custom head:**

  * Global Average Pooling
  * Dense (ReLU)
  * Dropout
  * Softmax output (7 classes including `unknown`)

---

### 5. Training & Fine-Tuning

* Initial training with frozen backbone
* Controlled fine-tuning by unfreezing last **30 layers** (BatchNorm layers kept frozen)
* Optimizer: **Adam**
* Learning rate scheduling for stable convergence
* Loss: **Categorical Cross-Entropy**
* Metrics monitored: Accuracy, Precision, Recall, F1 (validation)

**Result:**
High training accuracy with strong validation performance, suitable for real-world inference.

---

### 6. Model Export

* Final model converted to **TFLite (FP16 optimized)**
* Ready for **edge / mobile deployment**

---

### 7. Inference Pipeline (HR Testing)

All inference files are located inside the `predict_result` folder.

#### Folder Structure:

```
predict_result/
├── car_view_fp16.tflite
├── test_images/
├── test_predict.py
```

#### Setup Instructions:

```bash
pip install tensorflow numpy pandas opencv-python tqdm
```

#### Run Prediction:

```bash
cd predict_result
python test_predict.py
```

#### Output:

* A CSV file `predictions.csv` containing:

  * Image name
  * Predicted class
  * Confidence score

---

### 8. Notes for Evaluators

* The solution is **robust to random images** using the `unknown` class.
* The pipeline is **fully reproducible** and does not require training to run inference.
* The code is optimized for **clarity, stability, and edge deployment**.
