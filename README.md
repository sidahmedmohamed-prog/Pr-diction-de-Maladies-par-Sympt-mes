# 🩺 Disease Prediction by Symptoms

> A Machine Learning application for predicting diseases from clinical symptoms using multiple classification algorithms and an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-MachineLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Overview

Disease diagnosis is a complex process that often relies on analyzing multiple symptoms simultaneously. This project leverages Machine Learning techniques to automatically predict diseases based on patient symptoms.

The system analyzes **132 symptoms** and predicts one of **42 diseases** with high accuracy.

> **Disclaimer:** This project is intended for educational and research purposes only. It does not replace professional medical diagnosis.

---

## Objectives

* Analyze medical symptom datasets.
* Identify the most informative symptoms.
* Compare multiple Machine Learning algorithms.
* Optimize model performance using hyperparameter tuning.
* Deploy a user-friendly web application.

---

## 📊 Dataset Information

| Feature          | Value                      |
| ---------------- | -------------------------- |
| Diseases         | 42                         |
| Symptoms         | 132                        |
| Training Samples | 4920                       |
| Problem Type     | Multi-Class Classification |

Each record contains a set of binary symptoms associated with a specific disease.

---

## Project Workflow

```text
Data Collection
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
SMOTE Balancing
       ↓
Model Training
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Streamlit Deployment
```

---

##  Exploratory Data Analysis

The project includes:

* Disease distribution analysis
* Symptom frequency analysis
* Correlation heatmaps
* Mutual Information analysis
* Principal Component Analysis (PCA)
* Class balancing visualization using SMOTE
* Model comparison visualizations

---

## Machine Learning Models

The following models were trained and evaluated:

* Random Forest
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Gradient Boosting
* Decision Tree

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Random Forest       | 100%     |
| Logistic Regression | 100%     |
| KNN                 | 100%     |
| Naive Bayes         | 96.72%   |
| Gradient Boosting   | 85.25%   |
| Decision Tree       | 63.93%   |

### Best Performing Model

**Random Forest Classifier**

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 100%  |
| Precision | 100%  |
| Recall    | 100%  |
| F1-Score  | 100%  |
| AUC       | 1.00  |
| CV Mean   | 1.00  |
| CV Std    | 0.00  |

The Random Forest model achieved perfect classification performance while maintaining excellent stability across cross-validation folds.

---

## Visualizations

The project includes 12 visualizations:

* Disease Distribution
* Top Symptoms Frequency
* Correlation Heatmap
* Mutual Information Scores
* PCA Explained Variance
* SMOTE Effect
* Model Comparison
* Cross Validation Results
* Feature Importance
* Confusion Matrix
* ROC Curves
* Multi-Model Confusion Matrix Comparison

---

## Streamlit Application

### Features

#### Individual Prediction

* Select symptoms manually
* Instant disease prediction
* Confidence score
* Top predicted diseases

#### Batch Prediction

* Upload CSV files
* Upload Excel files
* Predict multiple patients simultaneously
* Export prediction results

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sidahmedmohamed-prog/Pr-diction-de-Maladies-par-Sympt-mes.git
cd Pr-diction-de-Maladies-par-Sympt-mes
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* Streamlit
* Joblib

---

## 📁 Project Structure

```text
├── data/
├── models/
│   ├── model_rf_tuned.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── symptom_cols.pkl
├── Projet6_Prediction_Maladies.ipynb/
├── app.py
├── requirements.txt
└── README.md
```

---

##  Future Improvements

* Integration with real-world clinical datasets
* Deep Learning models
* Disease recommendation system
* Mobile application
* Multi-language support

---

## Author

**Sidahmed Mohamed**

Faculty of Science and Technology
University of Nouakchott
Academic Year 2025–2026



Your support is greatly appreciated!
