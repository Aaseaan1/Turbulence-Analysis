#!/usr/bin/env python3
"""
Machine Learning Coursework 2: AI Model Comparison
Predicting Aircraft Turbulence using Decision Tree, KNN, and Logistic Regression
Author: Student ID 23057128
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("=" * 70)
print("MACHINE LEARNING MODELS: TURBULENCE PREDICTION")
print("=" * 70)

# Load dataset
df = pd.read_csv('../turbulence_dataset.csv')
print(f"\n✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape[0]} samples, {df.shape[1]} features")

# Display basic info
print("\n" + "-" * 70)
print("DATASET OVERVIEW")
print("-" * 70)
print(df.head(10))
print(f"\nDataset Info:")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"\nTarget Distribution:")
print(df['Turbulence'].value_counts())
print(f"  Class 0 (No Turbulence): {(df['Turbulence'] == 0).sum()} samples ({(df['Turbulence'] == 0).sum() / len(df) * 100:.1f}%)")
print(f"  Class 1 (Turbulence): {(df['Turbulence'] == 1).sum()} samples ({(df['Turbulence'] == 1).sum() / len(df) * 100:.1f}%)")

# Statistics
print(f"\nBasic Statistics:")
print(df.describe())

# ============================================================================
# 2. PREPARE DATA
# ============================================================================
print("\n" + "-" * 70)
print("DATA PREPARATION")
print("-" * 70)

# Separate features and target
X = df.drop('Turbulence', axis=1)
y = df['Turbulence']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✓ Data split completed:")
print("  Training set: {X_train.shape[0]} samples")
print("  Testing set: {X_test.shape[0]} samples")

# Scale features (important for KNN and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled using StandardScaler")

# ============================================================================
# 3. TRAIN MODELS
# ============================================================================
print("\n" + "-" * 70)
print("MODEL TRAINING")
print("-" * 70)

# 3.1 Decision Tree
print("\n1️⃣  DECISION TREE CLASSIFIER")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("✓ Decision Tree trained")

# 3.2 K-Nearest Neighbors (KNN)
print("\n2️⃣  K-NEAREST NEIGHBORS (KNN)")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
print("✓ KNN trained (k=5)")

# 3.3 Logistic Regression
print("\n3️⃣  LOGISTIC REGRESSION")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
print("✓ Logistic Regression trained")

# ============================================================================
# 4. EVALUATE MODELS
# ============================================================================
print("\n" + "=" * 70)
print("MODEL EVALUATION & COMPARISON")
print("=" * 70)

def evaluate_model(model_name, y_true, y_pred):
    """Calculate evaluation metrics for a model"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

# Evaluate all models
results = [
    evaluate_model('Decision Tree', y_test, dt_pred),
    evaluate_model('KNN (k=5)', y_test, knn_pred),
    evaluate_model('Logistic Regression', y_test, lr_pred)
]

# Create results dataframe
results_df = pd.DataFrame(results)
print("\n" + "-" * 70)
print("PERFORMANCE METRICS")
print("-" * 70)
print(results_df.to_string(index=False))

# ============================================================================
# 5. DETAILED ANALYSIS
# ============================================================================
print("\n" + "-" * 70)
print("DETAILED ANALYSIS")
print("-" * 70)

print("\n📊 DECISION TREE RESULTS:")
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred, target_names=['No Turbulence', 'Turbulence']))

print("\n📊 KNN RESULTS:")
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred, target_names=['No Turbulence', 'Turbulence']))

print("\n📊 LOGISTIC REGRESSION RESULTS:")
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred, target_names=['No Turbulence', 'Turbulence']))

# ============================================================================
# 6. CROSS-VALIDATION
# ============================================================================
print("\n" + "-" * 70)
print("CROSS-VALIDATION SCORES (5-Fold)")
print("-" * 70)

cv_dt = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
cv_knn = cross_val_score(knn_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

print(f"\nDecision Tree: {cv_dt.mean():.4f} (+/- {cv_dt.std():.4f})")
print(f"KNN: {cv_knn.mean():.4f} (+/- {cv_knn.std():.4f})")
print(f"Logistic Regression: {cv_lr.mean():.4f} (+/- {cv_lr.std():.4f})")

# ============================================================================
# 7. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "-" * 70)
print("GENERATING VISUALIZATIONS")
print("-" * 70)

fig = plt.figure(figsize=(15, 12))

# 1. Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
models = results_df['Model'].values
accuracies = results_df['Accuracy'].values
colors = ['#3498db', '#e74c3c', '#2ecc71']
ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 1])
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 2. All Metrics Comparison
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(results_df))
width = 0.2
ax2.bar(x - width, results_df['Accuracy'], width, label='Accuracy', color='#3498db', alpha=0.8)
ax2.bar(x, results_df['Precision'], width, label='Precision', color='#e74c3c', alpha=0.8)
ax2.bar(x + width, results_df['F1-Score'], width, label='F1-Score', color='#2ecc71', alpha=0.8)
ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax2.legend()
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)

# 3. Confusion Matrix - Decision Tree
ax3 = plt.subplot(2, 3, 3)
cm_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
ax3.set_title('Decision Tree - Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label', fontweight='bold')
ax3.set_xlabel('Predicted Label', fontweight='bold')

# 4. Confusion Matrix - KNN
ax4 = plt.subplot(2, 3, 4)
cm_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Reds', ax=ax4, cbar=False)
ax4.set_title('KNN - Confusion Matrix', fontsize=12, fontweight='bold')
ax4.set_ylabel('True Label', fontweight='bold')
ax4.set_xlabel('Predicted Label', fontweight='bold')

# 5. Confusion Matrix - Logistic Regression
ax5 = plt.subplot(2, 3, 5)
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=ax5, cbar=False)
ax5.set_title('Logistic Regression - Confusion Matrix', fontsize=12, fontweight='bold')
ax5.set_ylabel('True Label', fontweight='bold')
ax5.set_xlabel('Predicted Label', fontweight='bold')

# 6. Cross-Validation Scores
ax6 = plt.subplot(2, 3, 6)
cv_data = [cv_dt, cv_knn, cv_lr]
bp = ax6.boxplot(cv_data, labels=['Decision Tree', 'KNN', 'Logistic Reg'], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_ylabel('Accuracy Score', fontsize=11, fontweight='bold')
ax6.set_title('Cross-Validation Scores (5-Fold)', fontsize=12, fontweight='bold')
ax6.set_ylim([0.5, 1])
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'model_comparison.png'")
plt.close()

