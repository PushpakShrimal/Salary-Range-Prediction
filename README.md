# Salary-Range-Prediction
A classification project to predict shooting incident fatality using machine learning. Features data cleaning, SMOTE for imbalance, model comparison, and a Power BI dashboard.
An end-to-end classification project that predicts income levels based on demographic data. The repository includes a Jupyter notebook covering data cleaning, label encoding, and a comparative analysis of four machine learning models, with a TensorFlow-based ANN achieving the highest accuracy

# 💰 Salary Range Prediction using Census Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?logo=pandas)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Project Overview

This project aims to predict whether an individual's annual income is greater than $50,000 based on demographic and employment data from census records. This is a binary classification task that demonstrates a complete machine learning workflow, including data cleaning, feature engineering, model training, and comparative evaluation. Such a model has real-world applications in socio-economic analysis, financial services, and targeted marketing campaigns.

## 📊 The Dataset

The analysis is performed on the `salary.csv` dataset, which contains **32,561 entries** and **15 features**, including:
-   **Demographics:** `age`, `sex`, `race`, `marital-status`
-   **Education:** `education`, `education-num`
-   **Employment:** `workclass`, `occupation`, `hours-per-week`
-   **Target Variable:** `salary` (<=50K or >50K)

## 🛠️ Methodology & Workflow

The project followed a structured approach to build and evaluate the classification models:

1.  **Data Cleaning:** The dataset was loaded and inspected for missing values. Null entries in the `workclass`, `occupation`, and `native-country` columns were imputed using the mode (most frequent value).

2.  **Exploratory Data Analysis (EDA):** Visualizations were created to understand the distribution of various features. The target variable `salary` was found to be imbalanced, with about **76%** of individuals earning **<=50K** and **24%** earning **>50K**.

3.  **Data Preprocessing:**
    -   **Label Encoding:** All categorical features were converted into a numerical format using Label Encoding, making them suitable for machine learning models.
    -   **Feature Scaling:** All features were scaled using `StandardScaler` to ensure that no single feature disproportionately influences the model's predictions.

4.  **Model Training & Comparison:** The preprocessed data was split into training and testing sets. Four different classification models were trained and evaluated:
    -   Logistic Regression
    -   K-Nearest Neighbors (KNN)
    -   Random Forest Classifier
    -   **Artificial Neural Network (ANN)** using TensorFlow/Keras

## 🏆 Results & Model Performance

The models were evaluated based on their accuracy on the test set. The **Artificial Neural Network (ANN)** and **Random Forest** models delivered the best performance, with the ANN achieving a top accuracy of **84.65%**.

Here is a comparison of the accuracy scores for each model:

| Model                     | Accuracy on Test Set |
| ------------------------- | :------------------: |
| **Artificial Neural Network (ANN)** |      **84.65%** |
| Random Forest             |        84.46%        |
| Logistic Regression       |        81.28%        |
| K-Nearest Neighbors (KNN) |        80.99%        |

This chart visually represents the performance comparison:

*(You can add a screenshot of your model accuracy comparison bar chart here)*

The confusion matrix for the top-performing ANN model confirms its strong predictive capability for both salary classes.

*(You should add a screenshot of your ANN's confusion matrix here)*

## 🚀 How to Run this Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PushpakShrimal/Salary-Range-Prediction.git](https://github.com/PushpakShrimal/Salary-Range-Prediction.git)
    cd Salary-Range-Prediction
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook notebooks/Project_5_(Salary_Range_Prediction).ipynb
    ```

---
