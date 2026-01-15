# EMG Normal vs Aggressive Action Classification Results

## Overview
This document summarizes the results from the EMG (Electromyography) classification analysis comparing Normal vs Aggressive physical actions using machine learning models.

## Dataset Information
- **Total Samples**: 80 EMG recordings
- **Classification Task**: Binary classification (Normal vs Aggressive)
- **Data Source**: 4 subjects with 2 action types
- **Features Extracted**: 3 features per channel × 4 channels = 12 total features
  - RMS (Root Mean Square)
  - WL (Waveform Length)
  - VAR (Variance)
- **Preprocessing**: Z-score normalization applied

## Cross-Validation Setup
- **Method**: 5-Fold Cross-Validation
- **Random Seed**: 100 (for reproducibility)

## Model Performance Results

| Model | Accuracy | Sensitivity | Specificity |
|-------|----------|-------------|------------|
| SVM (Linear, C=10) | 90.00% | 92.50% | 87.50% |
| **SVM (RBF, C=50)** | **91.25%** | **92.50%** | **90.00%** |
| k-NN (k=5) | 91.25% | 95.00% | 87.50% |
| Decision Tree (ML=10) | 90.00% | 95.00% | 85.00% |

### Best Performing Model
**SVM with RBF Kernel (C=50)** achieved the highest accuracy at **91.25%**
- Balanced performance with 92.50% sensitivity and 90.00% specificity
- Good discrimination between Normal and Aggressive actions
- Robust generalization across folds

## Metric Definitions

### Accuracy
- **Definition**: Percentage of correct predictions out of all predictions
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Interpretation**: Overall correctness of the classifier

### Sensitivity (Recall / True Positive Rate)
- **Definition**: Percentage of actual positive cases correctly identified
- **Formula**: TP / (TP + FN)
- **Interpretation**: Ability to detect Aggressive actions
- **Range**: 92.50% - 95.00% across all models

### Specificity (True Negative Rate)
- **Definition**: Percentage of actual negative cases correctly identified
- **Formula**: TN / (TN + FP)
- **Interpretation**: Ability to identify Normal actions
- **Range**: 85.00% - 90.00% across all models

## Model Comparison Analysis

### SVM Linear vs RBF
- RBF kernel outperforms Linear kernel by 1.25% accuracy
- Both models achieve 92.50% sensitivity
- RBF has better specificity (90.00% vs 87.50%)
- RBF kernel captures non-linear decision boundaries more effectively

### k-NN (k=5)
- Ties with RBF-SVM on accuracy (91.25%)
- Highest sensitivity (95.00%) - best at detecting Aggressive actions
- Lower specificity (87.50%) - more false positives for Normal actions
- Sensitive to local data patterns

### Decision Tree
- Moderate accuracy (90.00%)
- Highest sensitivity (95.00%)
- Lowest specificity (85.00%)
- May be overfitting to training patterns

## Visualizations Generated

### 1. Confusion Matrices Tab
- 4 subplots showing normalized confusion matrices for each model
- Row-normalized format displays prediction accuracy per class
- Reveals classifier strengths and weaknesses per action type

### 2. ROC Curves Tab
- Receiver Operating Characteristic curves for all 4 models
- AUC (Area Under Curve) scores provided for each model
- Compares true positive rate vs false positive rate
- Random baseline shown for reference

### 3. Metrics Comparison Tab
- Bar chart comparing Accuracy, Sensitivity, and Specificity
- All models grouped by metric type
- Visual comparison of relative performance

### 4. Decision Boundary Tab
- 2D visualization using first two features (RMS and WL from Channel 1)
- Shows SVM Linear decision boundary
- Normal actions (Blue circles) vs Aggressive actions (Red triangles)
- Other feature dimensions represented as average values

## Key Findings

1. **High Classification Performance**: All models achieve >90% accuracy
2. **Balanced Sensitivity-Specificity**: Models perform well at detecting both action types
3. **Non-linear Separation**: RBF kernel advantage suggests features are non-linearly separable
4. **Sensitivity Priority**: k-NN and Decision Tree show higher sensitivity (95%), valuable for safety-critical applications requiring aggressive action detection
5. **Specificity Trade-off**: k-NN and Decision Tree sacrifice specificity for sensitivity

## Recommendations

### For Production Deployment
- **Primary Choice**: SVM RBF (best balance of accuracy and specificity)
- **Alternative**: k-NN if detecting all aggressive actions is critical (higher sensitivity)

### For Future Improvements
- Feature engineering: Include more discriminative EMG features (entropy, complexity measures)
- Hyperparameter tuning: Grid search for optimal C and kernel scale values
- Class imbalance handling: Check if Normal/Aggressive distribution is balanced
- Feature selection: Identify most important features for interpretation
- Ensemble methods: Combine multiple classifiers for robustness

## Model Characteristics Summary

| Aspect | SVM RBF | k-NN | Decision Tree |
|--------|---------|------|---------------|
| Training Time | Fast | Very Fast | Very Fast |
| Prediction Time | Fast | Slow (k-search) | Very Fast |
| Interpretability | Low | Low | High |
| Sensitivity to Noise | Low | High | Medium |
| Best for | General purpose | Quick deployment | Interpretability |

## Conclusion

The binary classification of Normal vs Aggressive EMG actions achieves excellent performance across all tested models, with the RBF-SVM model providing the best overall balance. The 91.25% accuracy demonstrates that EMG features (RMS, Waveform Length, Variance) are effective discriminators for action type classification. The high sensitivity (>92.5%) indicates reliable detection of aggressive movements, important for safety applications.

---
**Generated**: January 15, 2026  
**Analysis Tool**: MATLAB Classification Framework  
**Results File**: `classification_aggre_nor_results.mat`
