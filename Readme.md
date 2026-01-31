# Phishing Website Detection System 🛡️

A Machine Learning project that detects phishing websites using the **Random Forest** algorithm. This tool analyzes website URL features to classify them as either **Phishing (-1)** or **Legitimate (1)** with high accuracy.

## 📊 Project Overview
This project processes a dataset of website features, visualizes data correlations, trains a Random Forest Classifier, and evaluates the model's performance. It is designed to assist in cybersecurity efforts by automating the identification of malicious links.

### Key Features
* **Data Visualization:** Automatically generates class distribution plots and feature correlation heatmaps.
* **Model Training:** Uses a Random Forest Classifier (100 estimators) for robust predictions.
* **Performance Metrics:** Outputs accuracy score, detailed classification report, and confusion matrix.
* **Feature Importance:** Identifies the top 5 features contributing to phishing detection.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## 📂 Project Structure
```text
├── main.py                   # Main script for training and evaluation
├── phishing.csv              # Dataset (Required)
├── class_distribution.png    # Generated plot: Class balance
├── correlation_heatmap.png   # Generated plot: Feature correlations
└── confusion_matrix.png      # Generated plot: Model performance