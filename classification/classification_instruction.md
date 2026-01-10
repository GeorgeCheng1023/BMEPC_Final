
# Role: MATLAB Data Scientist (Classification Specialist)

## Project Context

**Goal**: Distinguish between **Normal** and **Aggressive** physical actions based on EMG signals.
**Input Data**: Pre-computed features from EMG signals (8 channels).
**Selected Features**:

1. **RMS** (Root Mean Square)
2. **WL** (Waveform Length)
3. **VAR** (Variance)
   **Reasoning**: Preliminary analysis indicates MAV and ZC provided poor class separation; thus, the model must rely strictly on RMS, WL, and VAR.

---

## Core Task: Classification Model Development

Generate a MATLAB script focused on training, validating, and comparing machine learning classifiers.

### 1. Data Preparation (Prerequisite)

* **Feature Matrix Construction**:
  * Assume input data is organized into a matrix `X` and label vector `Y`.
  * `X`: Columns should correspond to the 3 features (RMS, WL, VAR) for each channel.
  * `Y`: Binary labels (0 = Normal, 1 = Aggressive).
* **Normalization**: Apply **Z-score normalization** (`zscore`) to the feature matrix to ensure RMS, WL, and VAR are on the same scale before training.

### 2. Model Training & Comparison

Implement and compare the following **three classifiers** to find the best performer:

* **Model A: Support Vector Machine (SVM)**
  * Kernel: Compare `Linear` vs. `RBF`.
  * Objective: Maximize margin separation between normal and aggressive clusters.
* **Model B: k-Nearest Neighbors (k-NN)**
  * Settings: Set `k=5` and use Euclidean distance.
* **Model C: Decision Tree / Random Forest**
  * Purpose: To utilize tree-based logic for clear decision boundaries.
  * *Bonus*: If using Random Forest, calculate **Feature Importance** to see which feature (RMS vs. WL vs. VAR) contributes most.

### 3. Validation Strategy (Strict)

* **Cross-Validation**: Implement **5-Fold Cross-Validation** (`crossval`, `kfoldLoss`) to prevent overfitting. Do not rely on a single train-test split.
* **Metric Calculation**:
  For each model, calculate and output:
  1. **Average Accuracy** (%)
  2. **Sensitivity (Recall)** (Ability to correctly detect Aggression)
  3. **Specificity** (Ability to correctly identify Normal actions)

### 4. Visualization for Presentation

The script must generate high-quality plots for reporting:

1. **Confusion Matrix Heatmap**:
   * Use `confusionchart`.
   * Display distinct matrices for the best-performing model.
2. **ROC Curve & AUC**:
   * Plot ROC curves for all models on the same graph to compare performance visually (`perfcurve`).
3. **Decision Boundary Plot (Optional but Preferred)**:
   * Visualize the 2D or 3D decision boundary of the SVM model using the features.

### 5. Output Format

Generate a summary table in the MATLAB Command Window:

```text
| Model | Accuracy | Sensitivity | Specificity |
|-------|----------|-------------|-------------|
| SVM   | ...      | ...         | ...         |
| k-NN  | ...      | ...         | ...         |
| Tree  | ...      | ...         | ...         |
```
