# Disease Prediction System Using Machine Learning

## Project Overview

This project aims to predict diseases based on patients' symptoms using Machine Learning techniques. Several classification algorithms were trained and evaluated to identify the most accurate model for disease prediction.

The project includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature selection
* Class imbalance handling using SMOTE
* Model training and comparison
* Cross-validation
* Model evaluation
* Disease prediction application

---

## Dataset

The dataset contains symptoms as input features and diseases as target classes.

### Files

* Training.csv
* Testing.csv

Each row represents a patient and the symptoms they experience.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Imbalanced-Learn (SMOTE)
* Joblib
* Streamlit

---

## Project Structure

Disease-Prediction-System/

├── data/

│ ├── Training.csv

│ └── Testing.csv

├── notebooks/

│ └── Projet6_Prediction_Maladies.ipynb

├── models/

│ ├── model_rf_tuned.pkl

│ ├── scaler.pkl

│ ├── label_encoder.pkl

│ └── symptom_cols.pkl

├── images/

│ ├── disease_distribution.png

│ ├── symptoms_distribution.png

│ ├── correlation_heatmap.png

│ ├── model_comparison.png

│ └── confusion_matrix.png

├── results/

│ ├── predictions_resultats.csv

│ └── predictions_export.csv

├── app/

│ └── app.py

├── requirements.txt

├── README.md

└── .gitignore

---

## Methodology

### 1. Data Preprocessing

* Handling missing values
* Feature scaling
* Label encoding
* Data cleaning

### 2. Exploratory Data Analysis

* Disease distribution analysis
* Symptom frequency analysis
* Correlation analysis
* PCA visualization

### 3. Handling Imbalanced Data

SMOTE (Synthetic Minority Oversampling Technique) was applied to balance disease classes and improve model performance.

### 4. Model Training

Several machine learning models were tested:

* Random Forest
* Decision Tree
* K-Nearest Neighbors
* Logistic Regression
* Support Vector Machine

### 5. Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Cross-Validation Score
* Confusion Matrix
* ROC Curve

---

## Results

The Random Forest model achieved the best performance and was selected as the final model.

### Best Model

Random Forest Classifier

### Evaluation Metrics

* Accuracy: XX %
* Precision: XX %
* Recall: XX %
* F1-Score: XX %

(Replace XX with your real results)

---

## Running the Project

### Clone the repository

```bash
git clone https://github.com/yourusername/Disease-Prediction-System.git
cd Disease-Prediction-System
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Future Improvements

* Deep Learning implementation
* Web deployment
* Real-time prediction system
* Medical recommendation module
* Explainable AI (SHAP, LIME)

---

## Author

Mahfoudha Mohamedabbe

Data Science & Machine Learning Student
