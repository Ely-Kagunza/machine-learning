"""
Exploratory Data Analysis for Brazilian Malware Dataset
Generates comprehensive analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load the Brazilian malware dataset"""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - BRAZILIAN MALWARE DATASET")
    print("="*80)
    
    # Load data
    data_path = Path("brazilian-malware-dataset/brazilian-malware/brazilian-malware.csv")
    print(f"\n[1] Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"    Dataset shape: {df.shape}")
    print(f"    Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
    
    return df

def analyze_target_distribution(df):
    """Analyze target variable distribution"""
    print("\n[2] Target Variable Analysis:")
    
    if 'Label' in df.columns:
        target_counts = df['Label'].value_counts().sort_index()
        target_pcts = (target_counts / len(df) * 100).round(2)
        
        print(f"    Class 0 (Goodware): {target_counts.get(0, 0):,} samples ({target_pcts.get(0, 0)}%)")
        print(f"    Class 1 (Malware):  {target_counts.get(1, 0):,} samples ({target_pcts.get(1, 0)}%)")
        
        # Calculate imbalance ratio
        if 0 in target_counts and 1 in target_counts:
            ratio = max(target_counts) / min(target_counts)
            print(f"    Imbalance ratio: {ratio:.2f}:1")
        
        return target_counts
    else:
        print("    WARNING: 'Label' column not found!")
        return None

def analyze_missing_data(df):
    """Analyze missing data"""
    print("\n[3] Missing Data Analysis:")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    if missing.sum() == 0:
        print("    ✓ No missing values detected!")
    else:
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        print(f"    Total missing values: {missing.sum():,}")
        print("\n    Columns with missing values:")
        for col in missing_cols.index[:10]:
            print(f"      - {col}: {missing[col]:,} ({missing_pct[col]}%)")
    
    return missing

def analyze_data_types(df):
    """Analyze data types"""
    print("\n[4] Data Types:")
    
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"    {dtype}: {count} columns")
    
    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n    Numeric features: {len(numeric_cols)}")
    print(f"    Categorical features: {len(categorical_cols)}")
    
    return numeric_cols, categorical_cols

def analyze_duplicates(df):
    """Check for duplicate rows"""
    print("\n[5] Duplicate Analysis:")
    
    duplicates = df.duplicated().sum()
    dup_pct = (duplicates / len(df) * 100).round(2)
    
    print(f"    Duplicate rows: {duplicates:,} ({dup_pct}%)")
    
    return duplicates

def create_visualizations(df, output_dir):
    """Create and save visualizations"""
    print("\n[6] Generating Visualizations:")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Class Distribution
    if 'Label' in df.columns:
        plt.figure(figsize=(10, 6))
        
        counts = df['Label'].value_counts().sort_index()
        labels = ['Goodware (0)', 'Malware (1)']
        colors = ['#2ecc71', '#e74c3c']
        
        bars = plt.bar(labels, counts.values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}\n({height/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.title('Class Distribution: Goodware vs Malware', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        viz_path = output_dir / 'class_distribution.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {viz_path}")
    
    # 2. Feature Distributions (first 6 numeric features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' in numeric_cols:
        numeric_cols.remove('Label')
    
    if len(numeric_cols) >= 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Distribution of First 6 Numeric Features', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(numeric_cols[:6]):
            row = idx // 3
            col_idx = idx % 3
            
            axes[row, col_idx].hist(df[col].dropna(), bins=30, color='steelblue', 
                                   alpha=0.7, edgecolor='black')
            axes[row, col_idx].set_title(col, fontweight='bold')
            axes[row, col_idx].set_xlabel('Value')
            axes[row, col_idx].set_ylabel('Frequency')
            axes[row, col_idx].grid(alpha=0.3)
        
        plt.tight_layout()
        viz_path = output_dir / 'feature_distributions.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {viz_path}")
    
    # 3. Correlation Heatmap
    if len(numeric_cols) > 0:
        # Select subset for readability (max 20 features)
        cols_for_corr = numeric_cols[:20]
        corr_matrix = df[cols_for_corr].corr()
        
        plt.figure(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Heatmap (Top 20 Features)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        viz_path = output_dir / 'correlation_heatmap.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {viz_path}")

def save_summary(df, output_file):
    """Save EDA summary to text file"""
    print(f"\n[7] Saving EDA Summary:")
    
    output_file = Path(output_file)
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("Brazilian Malware Dataset\n")
        f.write("="*80 + "\n\n")
        
        # Dataset info
        f.write("DATASET OVERVIEW:\n")
        f.write(f"  Total Samples: {len(df):,}\n")
        f.write(f"  Total Features: {df.shape[1] - 1}\n")
        f.write(f"  Shape: {df.shape}\n\n")
        
        # Target distribution
        if 'Label' in df.columns:
            f.write("TARGET DISTRIBUTION:\n")
            target_counts = df['Label'].value_counts().sort_index()
            for label, count in target_counts.items():
                pct = (count / len(df) * 100)
                label_name = "Goodware" if label == 0 else "Malware"
                f.write(f"  Class {label} ({label_name}): {count:,} ({pct:.2f}%)\n")
            f.write("\n")
        
        # Missing data
        missing = df.isnull().sum().sum()
        f.write(f"MISSING DATA:\n")
        f.write(f"  Total Missing Values: {missing:,}\n")
        if missing > 0:
            missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
            for col, count in missing_cols.items():
                f.write(f"    {col}: {count:,}\n")
        f.write("\n")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        f.write(f"DUPLICATES:\n")
        f.write(f"  Duplicate Rows: {duplicates:,}\n\n")
        
        # Data types
        f.write("DATA TYPES:\n")
        for dtype, count in df.dtypes.value_counts().items():
            f.write(f"  {dtype}: {count} columns\n")
        f.write("\n")
        
        # Feature names
        f.write("FEATURE NAMES:\n")
        for col in df.columns:
            f.write(f"  - {col}\n")
    
    print(f"    ✓ Saved: {output_file}")

def main():
    """Main execution"""
    # Load data
    df = load_data()
    
    # Analyze
    analyze_target_distribution(df)
    analyze_missing_data(df)
    analyze_data_types(df)
    analyze_duplicates(df)
    
    # Create visualizations
    viz_dir = Path("data/visualizations")
    create_visualizations(df, viz_dir)
    
    # Save summary
    summary_file = Path("data/eda_summary.txt")
    save_summary(df, summary_file)
    
    print("\n" + "="*80)
    print("EDA COMPLETE!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - Visualizations: data/visualizations/")
    print(f"  - Summary: data/eda_summary.txt")
    print("\n")

if __name__ == "__main__":
    main()
