import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

print("\n" + "="*80)
print("BRAZILIAN MALWARE DATASET - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Define paths
base_path = Path(__file__).parent
data_path = base_path / "brazilian-malware-dataset"
goodware_path = data_path / "goodware-malware" / "goodware.csv"
malware_dir = data_path / "goodware-malware" / "malware-by-day"
output_dir = base_path / "data"
output_dir.mkdir(exist_ok=True)

# STEP 1: Load goodware and combine with malware
print("\n[STEP 1] Loading datasets...")
try:
    goodware = pd.read_csv(goodware_path)
    print(f"  - Goodware samples: {len(goodware)}")
    goodware['Label'] = 0  # 0 = goodware
except Exception as e:
    print(f"  ERROR loading goodware: {e}")
    exit(1)

# Load first malware file to get schema
malware_files = sorted(list(malware_dir.glob("*.csv")))
print(f"  - Found {len(malware_files)} malware daily files")

malware_list = []
for mfile in malware_files[:5]:  # Load first 5 for testing
    try:
        m = pd.read_csv(mfile)
        m['Label'] = 1  # 1 = malware
        malware_list.append(m)
    except Exception as e:
        print(f"    WARNING: Could not load {mfile.name}: {e}")

if malware_list:
    malware = pd.concat(malware_list, ignore_index=True)
    print(f"  - Malware samples loaded: {len(malware)}")
else:
    print("  ERROR: No malware files loaded")
    exit(1)

# Combine datasets
df = pd.concat([goodware, malware], ignore_index=True)
print(f"\n[STEP 2] Combined dataset shape: {df.shape}")
print(f"  - Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# STEP 3: Target variable analysis
print("\n[STEP 3] Target variable distribution:")
class_counts = df['Label'].value_counts().sort_index()
class_pcts = (class_counts / len(df) * 100).round(2)
for label, count in class_counts.items():
    pct = class_pcts[label]
    label_name = "Goodware" if label == 0 else "Malware"
    print(f"  - Class {label} ({label_name}): {count} samples ({pct}%)")

# STEP 4: Missing data analysis
print("\n[STEP 4] Missing data analysis:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_cols = missing[missing > 0].sort_values(ascending=False)
if len(missing_cols) > 0:
    for col_name in missing_cols.head(10).index:
        print(f"  - {col_name}: {missing[col_name]} missing ({missing_pct[col_name]}%)")
else:
    print("  - No missing values detected")

# STEP 5: Duplicate rows
print("\n[STEP 5] Duplicate analysis:")
duplicates = df.duplicated().sum()
dup_pct = (duplicates / len(df) * 100).round(2)
print(f"  - Duplicate rows: {duplicates} ({dup_pct}%)")

# STEP 6: Data types
print("\n[STEP 6] Data type summary:")
dtypes_summary = df.dtypes.value_counts()
for dtype, count in dtypes_summary.items():
    print(f"  - {dtype}: {count} columns")

# STEP 7: Feature variance analysis
print("\n[STEP 7] Feature variance analysis:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Label' in numeric_cols:
    numeric_cols.remove('Label')

print(f"  - Numeric features: {len(numeric_cols)}")
if numeric_cols:
    variances = df[numeric_cols].var()
    low_var = variances[variances == 0]
    if len(low_var) > 0:
        print(f"  - Constant features (variance=0): {len(low_var)}")
        for col in low_var.index[:5]:
            print(f"    * {col}")
    else:
        print(f"  - No constant features detected")

# STEP 8: Correlation analysis
print("\n[STEP 8] Correlation analysis:")
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print(f"  - Highly correlated pairs (>0.95): {len(high_corr_pairs)}")
        for col1, col2, corr in high_corr_pairs[:5]:
            print(f"    * {col1} <-> {col2}: {corr:.4f}")
    else:
        print(f"  - No highly correlated pairs (>0.95) found")
else:
    print(f"  - Insufficient numeric features for correlation analysis")

# STEP 9: Train/Test Split
print("\n[STEP 9] Creating train/test split (80/20)...")
X = df.drop('Label', axis=1)
y = df['Label']

# Stratified split with random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  - Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"    * Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
print(f"    * Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
print(f"  - Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"    * Class 0: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
print(f"    * Class 1: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")

# STEP 10: Save splits
print("\n[STEP 10] Saving train/test splits to CSV...")
X_train.to_csv(output_dir / "X_train.csv", index=False)
X_test.to_csv(output_dir / "X_test.csv", index=False)
y_train.to_csv(output_dir / "y_train.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)
print(f"  - Files saved to: {output_dir}")

# STEP 11: Generate summary report
print("\n[STEP 11] Generating EDA summary report...")
summary = {
    "dataset_info": {
        "total_samples": int(len(df)),
        "total_features": int(df.shape[1] - 1),
        "features_numeric": len(numeric_cols),
        "features_non_numeric": int(df.shape[1] - len(numeric_cols) - 1)
    },
    "class_distribution": {
        "class_0_goodware": int(class_counts[0]),
        "class_0_percent": float(class_pcts[0]),
        "class_1_malware": int(class_counts[1]),
        "class_1_percent": float(class_pcts[1])
    },
    "data_quality": {
        "missing_values_total": int(missing.sum()),
        "duplicate_rows": int(duplicates),
        "duplicate_percent": float(dup_pct),
        "constant_features": len(low_var) if 'low_var' in locals() else 0
    },
    "train_test_split": {
        "random_seed": 42,
        "train_samples": int(len(X_train)),
        "train_percent": 80,
        "test_samples": int(len(X_test)),
        "test_percent": 20,
        "train_class_0": int((y_train == 0).sum()),
        "train_class_1": int((y_train == 1).sum()),
        "test_class_0": int((y_test == 0).sum()),
        "test_class_1": int((y_test == 1).sum())
    },
    "recommendations": [
        "Remove constant features before modeling",
        "Handle missing values in Identify column",
        "Consider removing duplicate rows",
        "Use stratified cross-validation (already done in train/test split)",
        "Apply preprocessing (scaling, encoding) only on training data"
    ]
}

with open(output_dir / "eda_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  - Summary saved to: {output_dir / 'eda_summary.json'}")

print("\n" + "="*80)
print("EDA COMPLETE - All outputs ready for preprocessing!")
print("="*80 + "\n")
