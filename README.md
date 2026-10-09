# Customer Churn Prediction Using Machine Learning

## Overview

This project uses machine learning to predict whether a customer is likely to leave a service (churn) based on customer-related data. It uses Python, Pandas, and Scikit-learn to preprocess data, train a classification model, and evaluate its performance.

## Objectives

* Analyze customer data to identify churn patterns.
* Handle missing values in selected numerical features.
* Train a Decision Tree classification model.
* Evaluate model performance using accuracy.

## Technologies Used

* **Programming Language:** Python
* **Libraries:** Pandas, Scikit-learn
* **Machine Learning Algorithm:** Decision Tree Classifier
* **Tools:** VS Code, Git, GitHub

## Dataset

The dataset is stored in `customer_churn.csv`.

The model uses the following features:

* `monthly_charge`
* `months_as_customer`
* `support_tickets`
* `weekly_usage_hours`

**Target variable:** `churned`

Missing values in `support_tickets` and `weekly_usage_hours` are filled using their respective median values.

## Project Workflow

1. Load the customer dataset using Pandas.
2. Handle missing values.
3. Separate input features and the target variable.
4. Split the data into training and testing sets.
5. Train a Decision Tree Classifier.
6. Generate predictions on the test data.
7. Evaluate the model using accuracy.

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-prediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python customer_churn.py
```

## Model Evaluation

The model's performance is evaluated using accuracy on the test dataset. The script also displays actual and predicted churn values for comparison.

## Future Improvements

* Compare Decision Tree performance with other classification algorithms.
* Evaluate the model using precision, recall, F1-score, and a confusion matrix.
* Improve feature selection and model performance.

## Author

**Lohitha Reddy**

GitHub: https://github.com/Lohitha01456
