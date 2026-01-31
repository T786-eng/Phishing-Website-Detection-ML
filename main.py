import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Load Data
# ==========================================
print("Loading dataset...")
try:
    # Ensure 'phishing.csv' is in the same folder as this script
    df = pd.read_csv('phishing.csv')
    print(f"Data Loaded Successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'phishing.csv' file not found.")
    print("Please download the dataset and place it in the same directory as this script.")
    exit()

# Drop 'Index' column if it exists as it is not needed for training
if 'Index' in df.columns:
    df = df.drop('Index', axis=1)

# Rename 'class' column to 'Result' for consistency
df.rename(columns={'class': 'Result'}, inplace=True)

# ==========================================
# 2. Visualization (Pop-up windows)
# ==========================================
print("Generating visualizations...")

# Plot 1: Target Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Result', data=df, hue='Result', palette='viridis', legend=False)
plt.title('Class Distribution: Phishing (-1) vs Legitimate (1)')
plt.xlabel('Result')
plt.ylabel('Count')
plt.savefig('class_distribution.png')  # Save plot instead of showing
print("Class distribution plot saved as 'class_distribution.png'")

# Plot 2: Correlation Heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')  # Save plot instead of showing
print("Correlation heatmap saved as 'correlation_heatmap.png'")

# ==========================================
# 3. Data Splitting
# ==========================================
# Features (X) are all columns except 'Result'
X = df.drop('Result', axis=1)
# Target (y) is the 'Result' column
y = df['Result']

# Split: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ==========================================
# 4. Model Training (Random Forest)
# ==========================================
print("\nTraining Random Forest model...")
# n_estimators=100 means we are using 100 decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Training Complete.")

# ==========================================
# 5. Model Evaluation
# ==========================================
print("\nEvaluating model...")
y_pred = rf_model.predict(X_test)

# Calculate Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# Print Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Phishing', 'Legitimate'],
            yticklabels=['Phishing', 'Legitimate'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save plot instead of showing
print("Confusion matrix plot saved as 'confusion_matrix.png'")

# ==========================================
# 6. Feature Importance (Bonus)
# ==========================================
# This shows which features were most useful for detection
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 5 Most Important Features for Detection:")
for i in range(5):
    print(f"{i+1}. {X.columns[indices[i]]} ({importances[indices[i]]:.4f})")