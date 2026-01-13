# Exploratory Data Analysis Report

## Executive Summary

The Brazilian Malware Dataset contains 21,214 executable samples with 31 features for binary classification (goodware vs. malware). The dataset is **highly imbalanced** with 99.54% goodware samples and only 0.46% malware samples. The exploration revealed significant data quality issues including missing values, duplicates, and constant features that require careful preprocessing.

---

## Dataset Overview

### Size and Composition
- **Total Samples**: 21,214
- **Total Features**: 31 (excluding target label)
- **Target Variable**: Label (0 = Goodware, 1 = Malware)

### Feature Types
| Type | Count | Description |
|------|-------|-------------|
| Numeric (int64) | 22 | Extracted static features from .exe files |
| Non-numeric (object) | 9 | String fields (dates, hashes, etc.) |
| Float | 1 | Calculated metrics |

---

## Target Variable Analysis

### Class Distribution
| Class | Label | Count | Percentage | Type |
|-------|-------|-------|-----------|------|
| Class 0 | 0 | 21,116 | 99.54% | Goodware |
| Class 1 | 1 | 98 | 0.46% | Malware |

### Key Insight
The dataset exhibits **severe class imbalance** (215:1 ratio). This will require:
- Stratified cross-validation (to maintain class proportions in folds)
- Use of **Area Under the ROC Curve (AUC)** as primary metric instead of accuracy
- Possible class weight balancing during model training
- Careful handling during train/test split (already done with stratification)

---

## Data Quality Assessment

### Missing Data
| Column | Missing Count | Missing % | Notes |
|--------|---------------|-----------|-------|
| FileType | 21,116 | 99.54% | Almost entirely missing - consider dropping |
| Fuzzy | 21,116 | 99.54% | Almost entirely missing - consider dropping |
| Identify | 9,373 | 44.18% | Significant missing values - requires imputation strategy |
| Others | 0-few | 0-0.5% | Minimal or no missing values |

**Handling Strategy**: 
- Drop columns with >95% missing data (FileType, Fuzzy)
- Use domain-informed imputation for Identify column

### Duplicate Rows
- **Duplicate Count**: 6 rows (0.03%)
- **Recommendation**: Remove duplicates during preprocessing
- **Impact**: Minimal, but important for data quality

### Constant Features (Zero Variance)
| Feature | Variance | Notes |
|---------|----------|-------|
| Magic | 0.0 | Same value in all samples |
| PE_TYPE | 0.0 | Same value in all samples |
| SizeOfOptionalHeader | 0.0 | Same value in all samples |

**Recommendation**: Drop these 3 features before modeling as they provide no predictive information.

### Feature Correlations
- **Highly Correlated Pair** (>0.95): BaseOfData <-> SizeOfCode (r=0.97)
- **Recommendation**: Consider removing one of these features or keeping both for interpretability
- **Overall**: No severe multicollinearity issues detected beyond this pair

---

## Train/Test Split

### Stratified 80/20 Split (Random Seed: 42)

**Training Set** (80%):
- Total samples: 16,971
- Class 0 (Goodware): 16,893 samples (99.54%)
- Class 1 (Malware): 78 samples (0.46%)

**Test Set** (20%):
- Total samples: 4,243
- Class 0 (Goodware): 4,223 samples (99.53%)
- Class 1 (Malware): 20 samples (0.47%)

### Quality Checks
- [OK] Stratified split maintains class proportions in both sets
- [OK] Test set held out separately (not used for preprocessing)
- [OK] Random seed fixed at 42 for reproducibility
- [OK] No data leakage detected

---

## Preprocessing Recommendations

### Feature Engineering Pipeline (To Be Fit on Training Only)
1. **Drop High-Missing Features**: FileType, Fuzzy
2. **Drop Constant Features**: Magic, PE_TYPE, SizeOfOptionalHeader
3. **Handle Missing Values**: 
   - Strategy for Identify column: mode imputation or special "Unknown" category
4. **Encode Non-Numeric Features**: 
   - One-hot encoding or target encoding for 9 object columns
5. **Scaling**: 
   - StandardScaler for numeric features
   - Important: Fit scaler ONLY on training data, apply to test

### Critical Steps
```
1. Load combined dataset (21,214 x 32)
   ↓
2. Train/Test Split (80/20 STRATIFIED with seed=42) ✓ ALREADY DONE
   ↓
3. Feature cleaning (drop high-missing, constant features)
   ↓
4. Fit preprocessing on TRAINING data only
   ↓
5. Transform both training and test with fitted pipeline
   ↓
6. Ready for model training (cross-validation on training, final eval on test)
```

---

## Feature Summary

### Numeric Features (22 total)
- Range: Mix of binary flags, counts, and computed metrics
- Distribution: To be analyzed during preprocessing
- Scaling needed: Yes (for distance-based and regularized models)

### Non-Numeric Features (9 total)
- FirstSeenDate: Temporal information
- Identify: Classification/type information (44% missing)
- ImportedDlls: Library dependencies
- ImportedSymbols: Function imports
- SHA1: File hash (identifier)
- Others: Domain-specific string fields

---

## Dataset Split Files Generated

The following files have been generated and saved to `data/`:

| File | Format | Rows | Columns | Purpose |
|------|--------|------|---------|---------|
| X_train.csv | CSV | 16,971 | 31 | Training features |
| X_test.csv | CSV | 4,243 | 31 | Test features |
| y_train.csv | CSV | 16,971 | 1 | Training labels |
| y_test.csv | CSV | 4,243 | 1 | Test labels |
| eda_summary.json | JSON | - | - | EDA metrics summary |

---

## Key Insights for Modeling

1. **Imbalanced Classification**: Use stratified k-fold CV with AUC metric
2. **Small Minority Class**: Only 98 malware samples total (20 in test set)
3. **High-Missing Features**: FileType and Fuzzy columns unusable
4. **Constant Features**: Remove 3 non-informative features
5. **Data Quality**: High quality overall after removing problematic features
6. **Feature Engineering**: Consider domain-specific features from PE header fields
7. **Baseline Concerns**: All-goodware baseline = 99.54% accuracy (monitor AUC instead)

---

## Next Steps (Week 1, Days 4-5)

1. ✓ EDA Complete - Data quality assessed
2. Next: Build preprocessing pipeline
3. Then: Apply to both train/test splits
4. Finally: Generate clean data for model training

**Status**: Ready to proceed to preprocessing and modeling phases.

---

*Generated: Brazilian Malware Dataset EDA*  
*Dataset Size: 21,214 samples × 31 features*  
*Split: 80% training (16,971) / 20% testing (4,243)*  
*Class Balance: Maintained through stratified split*
