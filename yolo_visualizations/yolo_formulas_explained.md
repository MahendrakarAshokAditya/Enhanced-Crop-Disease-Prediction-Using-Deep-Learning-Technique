# YOLOv8 Mathematical Formulas Explained

This document provides a clear explanation of the key mathematical formulas that power the YOLOv8 model used in our crop disease detection system.

## 1. Object Detection Fundamentals

### Bounding Box Prediction

YOLOv8 predicts bounding boxes using a center-based approach:

```
b_x = σ(t_x) + c_x
b_y = σ(t_y) + c_y
b_w = p_w × e^(t_w)
b_h = p_h × e^(t_h)
```

**Where:**
- **b_x, b_y**: Center coordinates of the predicted bounding box
- **b_w, b_h**: Width and height of the predicted bounding box
- **t_x, t_y, t_w, t_h**: Raw outputs from the network
- **c_x, c_y**: Grid cell offset coordinates
- **p_w, p_h**: Prior dimensions (anchor-free in YOLOv8)
- **σ**: Sigmoid activation function constraining values between 0 and 1

**In crop disease detection:** These formulas enable precise localization of diseased areas on plant leaves, stems, and fruits.

## 2. Loss Function Components

YOLOv8 uses a composite loss function combining several components:

```
L_total = λ_box × L_box + λ_cls × L_cls + λ_dfl × L_dfl
```

**Where:**
- **L_total**: Total loss
- **L_box**: Bounding box regression loss
- **L_cls**: Classification loss
- **L_dfl**: Distribution Focal Loss for better localization
- **λ_box, λ_cls, λ_dfl**: Weighting coefficients for each loss component

**In crop disease detection:** This balanced loss function ensures the model accurately identifies both the location and type of plant disease.

### 2.1 Bounding Box Regression Loss

YOLOv8 uses CIoU (Complete IoU) loss for bounding box regression:

```
L_box = 1 - IoU + ρ²(b, b_gt)/c² + α×v
```

**Where:**
- **IoU**: Intersection over Union between predicted and ground truth boxes
- **ρ**: Euclidean distance between centers of predicted and ground truth boxes
- **c**: Diagonal length of the smallest enclosing box covering both boxes
- **v**: Aspect ratio consistency term
- **α**: Trade-off parameter
- **b**: Predicted bounding box
- **b_gt**: Ground truth bounding box

**In crop disease detection:** CIoU loss helps the model precisely outline irregular disease patterns on plant surfaces.

### 2.2 Classification Loss

YOLOv8 uses Binary Cross-Entropy (BCE) loss with logits for multi-label classification:

```
L_cls = -∑[y_i × log(σ(x_i)) + (1 - y_i) × log(1 - σ(x_i))]
```

**Where:**
- **y_i**: Ground truth label (0 or 1)
- **x_i**: Predicted logit
- **σ**: Sigmoid activation function

**In crop disease detection:** BCE loss enables the model to distinguish between multiple disease types that may appear visually similar.

### 2.3 Distribution Focal Loss (DFL)

YOLOv8 introduces DFL for more precise bounding box regression:

```
L_dfl = -∑ y_i × log(p_i)
```

**Where:**
- **y_i**: Ground truth distribution
- **p_i**: Predicted distribution

**In crop disease detection:** DFL improves the model's ability to detect small lesions and early-stage disease symptoms.

## 3. Non-Maximum Suppression (NMS)

After obtaining raw predictions, YOLOv8 applies NMS to filter redundant detections:

```
S = {s_1, s_2, ..., s_n} sorted by confidence
R = {}
while S is not empty:
    m = argmax(S)
    R = R ∪ {m}
    S = S - {m}
    for each s in S:
        if IoU(m, s) > τ_nms:
            S = S - {s}
return R
```

**Where:**
- **S**: Set of all detections
- **R**: Set of kept detections
- **m**: Detection with highest confidence
- **τ_nms**: NMS threshold (typically 0.45-0.7)
- **IoU**: Intersection over Union

**In crop disease detection:** NMS prevents multiple detections of the same disease spot, improving accuracy in densely infected areas.

## 4. Crop Disease-Specific Formulas

### 4.1 Disease Confidence Score

```
C_disease = σ(p_cls) × IoU_pred
```

**Where:**
- **C_disease**: Final confidence score for disease detection
- **p_cls**: Raw classification logit
- **IoU_pred**: Predicted IoU with ground truth
- **σ**: Sigmoid activation function

**In crop disease detection:** This formula balances classification confidence with localization accuracy to provide reliable disease diagnoses.

### 4.2 Multi-Disease Detection Threshold

```
T_disease = T_base + α × (1 - IoU_pred)
```

**Where:**
- **T_disease**: Adaptive threshold for disease detection
- **T_base**: Base threshold (typically 0.25)
- **α**: Scaling factor (typically 0.15)
- **IoU_pred**: Predicted IoU with ground truth

**In crop disease detection:** Adaptive thresholding helps detect multiple diseases on the same plant with varying confidence levels.

### 4.3 Disease Severity Estimation

```
S_disease = β × A_rel + (1 - β) × C_disease
```

**Where:**
- **S_disease**: Estimated disease severity (0-1)
- **A_rel**: Relative affected area (ratio of disease bounding box to plant area)
- **C_disease**: Disease confidence score
- **β**: Weighting factor (typically 0.7)

**In crop disease detection:** This formula quantifies disease severity, helping farmers prioritize treatment actions.

## 5. Training Hyperparameters

Our YOLOv8 model for crop disease detection uses these specific hyperparameters:

- **Learning Rate (η)**: 0.01 with cosine decay schedule
- **Batch Size**: 16
- **Image Size**: 480×480 pixels
- **Augmentation Strength (μ_aug)**: 0.5
- **Mosaic Probability (p_mosaic)**: 0.3
- **Mixup Probability (p_mixup)**: 0.15
- **Weight Decay (λ_decay)**: 0.0005
- **Warmup Epochs (e_warmup)**: 3
- **Total Epochs (e_total)**: 50

**In crop disease detection:** These carefully tuned hyperparameters optimize the model's ability to recognize diverse disease patterns across different crop types and growing conditions.

## 6. Inference Optimization

During inference, we apply these techniques to optimize performance:

```
T_final = max(T_min, min(T_max, T_base - γ × (1 - C_disease)))
```

**Where:**
- **T_final**: Final detection threshold
- **T_min**: Minimum threshold (0.2)
- **T_max**: Maximum threshold (0.7)
- **T_base**: Base threshold (0.45)
- **γ**: Adaptive factor (0.2)
- **C_disease**: Disease confidence score

**In crop disease detection:** This adaptive threshold balances false positives and false negatives, critical for reliable field deployment.

## 7. Model Performance Analysis

Our YOLOv8 model achieved the following performance metrics after training for 50 epochs on a dataset of 367 images containing 1070 disease instances:

### 7.1 Loss Components

```
L_box = 0.7931
L_cls = 0.7773
L_dfl = 1.103
```

These loss values indicate that the model has reached a reasonable convergence point, with the Distribution Focal Loss (DFL) being slightly higher than the box and classification losses, suggesting that precise localization remains the most challenging aspect of crop disease detection.

### 7.2 Detection Performance

```
Precision = 0.648
Recall = 0.631
mAP@0.5 = 0.643
mAP@0.5:0.95 = 0.483
```

These metrics demonstrate the model's ability to detect crop diseases with moderate accuracy. The mAP@0.5 value of 0.643 indicates that the model correctly identifies diseases in about 64.3% of cases when using a 50% IoU threshold. The lower mAP@0.5:0.95 value (0.483) reflects the challenge of precisely localizing disease boundaries, which often have irregular shapes and gradual transitions.

**In crop disease detection:** These performance metrics provide farmers with a reliable tool for early disease detection, with sufficient accuracy to guide treatment decisions while minimizing false alarms.

---

These mathematical formulas form the foundation of our YOLOv8-based crop disease detection system, enabling accurate identification and classification of plant diseases across multiple crop types and growing conditions.