"""
Model Training & Evaluation Pipeline
Week 2: Train 7+ ML models with 10-fold stratified cross-validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, 
    classification_report, roc_curve, auc
)

# Boosting models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class ModelTrainer:
    """Complete model training and evaluation pipeline"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
        self.scoring = {'auc': 'roc_auc', 'accuracy': 'accuracy'}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.best_auc = 0
        
    def load_data(self):
        """Load preprocessed train and test data"""
        print("\n[STEP 1] Loading preprocessed data...")
        
        data_dir = Path("data")
        
        X_train = pd.read_csv(data_dir / "X_train_processed.csv")
        X_test = pd.read_csv(data_dir / "X_test_processed.csv")
        y_train = pd.read_csv(data_dir / "y_train_processed.csv").squeeze()
        y_test = pd.read_csv(data_dir / "y_test_processed.csv").squeeze()
        
        print(f"  - X_train shape: {X_train.shape}")
        print(f"  - X_test shape: {X_test.shape}")
        print(f"  - y_train shape: {y_train.shape}")
        print(f"  - y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, model_name, X_train, y_train):
        """Train a single model with cross-validation"""
        print(f"\n  Training: {model_name}...", end=" ", flush=True)
        
        start_time = time.time()
        
        # Run cross-validation
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=self.cv,
            scoring=self.scoring,
            return_train_score=False,
            n_jobs=-1
        )
        
        train_time = time.time() - start_time
        
        # Calculate mean and std
        auc_mean = cv_results['test_auc'].mean()
        auc_std = cv_results['test_auc'].std()
        acc_mean = cv_results['test_accuracy'].mean()
        acc_std = cv_results['test_accuracy'].std()
        
        print(f"AUC: {auc_mean:.4f}±{auc_std:.4f}, Time: {train_time:.1f}s")
        
        # Store results
        result = {
            'Model': model_name,
            'CV_AUC_Mean': auc_mean,
            'CV_AUC_Std': auc_std,
            'CV_Accuracy_Mean': acc_mean,
            'CV_Accuracy_Std': acc_std,
            'Train_Time_Sec': train_time,
            'CV_Fold_AUCs': cv_results['test_auc'].tolist(),
            'CV_Fold_Accuracies': cv_results['test_accuracy'].tolist()
        }
        
        self.results.append(result)
        
        # Track best model
        if auc_mean > self.best_auc:
            self.best_auc = auc_mean
            self.best_model_name = model_name
            self.best_model = model
        
        return result
    
    def train_all_models(self, X_train, y_train):
        """Train all 7 models"""
        print("\n" + "="*80)
        print("TRAINING MODELS WITH 10-FOLD STRATIFIED CROSS-VALIDATION")
        print("="*80)
        
        # 1. Logistic Regression
        self.train_model(
            LogisticRegression(random_state=self.random_state, max_iter=1000),
            "Logistic Regression",
            X_train, y_train
        )
        
        # 2. Decision Tree
        self.train_model(
            DecisionTreeClassifier(random_state=self.random_state, max_depth=15),
            "Decision Tree",
            X_train, y_train
        )
        
        # 3. Random Forest
        self.train_model(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "Random Forest",
            X_train, y_train
        )
        
        # 4. Neural Network (MLP)
        self.train_model(
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                random_state=self.random_state,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1
            ),
            "Neural Network (MLP)",
            X_train, y_train
        )
        
        # 5. XGBoost
        self.train_model(
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                eval_metric='auc',
                use_label_encoder=False,
                verbosity=0
            ),
            "XGBoost",
            X_train, y_train
        )
        
        # 6. LightGBM
        self.train_model(
            LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                verbose=-1
            ),
            "LightGBM",
            X_train, y_train
        )
        
        # 7. Gradient Boosting
        self.train_model(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            "Gradient Boosting",
            X_train, y_train
        )
        
        print("\n" + "="*80)
        print("CROSS-VALIDATION TRAINING COMPLETE")
        print("="*80)
    
    def save_cv_results(self, output_file="data/cv_results.csv"):
        """Save CV results to CSV"""
        print(f"\n[STEP 2] Saving cross-validation results...")
        
        # Create results dataframe (without fold-level details)
        df_results = pd.DataFrame([
            {
                'Model': r['Model'],
                'CV_AUC_Mean': f"{r['CV_AUC_Mean']:.4f}",
                'CV_AUC_Std': f"{r['CV_AUC_Std']:.4f}",
                'CV_Accuracy_Mean': f"{r['CV_Accuracy_Mean']:.4f}",
                'CV_Accuracy_Std': f"{r['CV_Accuracy_Std']:.4f}",
                'Train_Time_Sec': f"{r['Train_Time_Sec']:.1f}"
            }
            for r in self.results
        ])
        
        df_results.to_csv(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")
        print("\nCV Results Summary:")
        print(df_results.to_string(index=False))
        
        return df_results
    
    def retrain_best_model(self, X_train, y_train):
        """Retrain best model on full training set"""
        print(f"\n[STEP 3] Retraining best model on full training set...")
        print(f"  Best model: {self.best_model_name} (CV AUC: {self.best_auc:.4f})")
        
        self.best_model.fit(X_train, y_train)
        print(f"  ✓ Model trained")
    
    def evaluate_on_test(self, X_test, y_test):
        """Evaluate best model on test set"""
        print(f"\n[STEP 4] Evaluating best model on hold-out test set...")
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        print(f"\n  Test AUC: {test_auc:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def save_test_results(self, test_results, output_file="data/test_results.txt"):
        """Save test results to file"""
        print(f"\n[STEP 5] Saving test results...")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FINAL MODEL EVALUATION - TEST SET\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write(f"Selected based on highest CV AUC: {self.best_auc:.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("TEST SET PERFORMANCE\n")
            f.write("="*80 + "\n")
            f.write(f"Test AUC: {test_results['test_auc']:.4f}\n")
            f.write(f"Test Accuracy: {test_results['test_accuracy']:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(str(test_results['confusion_matrix']) + "\n\n")
            
            f.write("Classification Report:\n")
            f.write(test_results['classification_report'] + "\n\n")
            
            f.write("="*80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("="*80 + "\n")
            f.write(f"True Negatives (Correctly identified goodware): {test_results['confusion_matrix'][0,0]}\n")
            f.write(f"False Positives (Goodware misclassified as malware): {test_results['confusion_matrix'][0,1]}\n")
            f.write(f"False Negatives (Malware misclassified as goodware): {test_results['confusion_matrix'][1,0]}\n")
            f.write(f"True Positives (Correctly identified malware): {test_results['confusion_matrix'][1,1]}\n\n")
            
            f.write("="*80 + "\n")
            f.write("NEXT STEPS\n")
            f.write("="*80 + "\n")
            f.write("1. Model saved to: models/best_model.pkl\n")
            f.write("2. Ready for deployment in Flask web application (Week 3)\n")
            f.write("3. Ready for CI/CD pipeline setup (Week 4)\n")
        
        print(f"  ✓ Saved to: {output_file}")
    
    def save_best_model(self, output_file="models/best_model.pkl"):
        """Save best model to disk"""
        print(f"\n[STEP 6] Saving best model...")
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"  ✓ Model saved to: {output_file}")
    
    def create_comparison_plot(self, output_file="data/model_comparison.png"):
        """Create model comparison visualization"""
        print(f"\n[STEP 7] Creating model comparison visualization...")
        
        # Prepare data
        models = [r['Model'] for r in self.results]
        auc_means = [r['CV_AUC_Mean'] for r in self.results]
        auc_stds = [r['CV_AUC_Std'] for r in self.results]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: AUC Comparison
        colors = ['green' if m == self.best_model_name else 'steelblue' for m in models]
        x_pos = np.arange(len(models))
        
        ax1.bar(x_pos, auc_means, yerr=auc_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
        ax1.set_title('Cross-Validation AUC Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylim([0.8, 1.0])
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(auc_means, auc_stds)):
            ax1.text(i, mean + std + 0.005, f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Train Time Comparison
        train_times = [r['Train_Time_Sec'] for r in self.results]
        
        ax2.barh(models, train_times, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Model Training Time', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, time in enumerate(train_times):
            ax2.text(time + 1, i, f'{time:.1f}s', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Visualization saved to: {output_file}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("WEEK 2: MODEL TRAINING & EVALUATION PIPELINE")
    print("="*80)
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    
    # Train all models with 10-fold CV
    trainer.train_all_models(X_train, y_train)
    
    # Save CV results
    trainer.save_cv_results()
    
    # Retrain best model on full training set
    trainer.retrain_best_model(X_train, y_train)
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test(X_test, y_test)
    
    # Save test results
    trainer.save_test_results(test_results)
    
    # Save best model
    trainer.save_best_model()
    
    # Create visualization
    trainer.create_comparison_plot()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"CV AUC: {trainer.best_auc:.4f}")
    print(f"Test AUC: {test_results['test_auc']:.4f}")
    print(f"\nOutputs:")
    print(f"  - data/cv_results.csv (all model comparisons)")
    print(f"  - data/test_results.txt (final evaluation)")
    print(f"  - models/best_model.pkl (production model)")
    print(f"  - data/model_comparison.png (visualization)")
    print("\n")


if __name__ == "__main__":
    main()
