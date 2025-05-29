# Heart Disease Analysis Dashboard 🩺

![Dashboard Preview][image](https://github.com/user-attachments/assets/400497fa-5a6f-4aa5-b524-b59d4daa5de6)

![Dashboard Preview](images/dashboard_preview_dark.png#gh-dark-mode-only)

[![GitHub license](https://img.shields.io/github/license/yourusername/heart-disease-dashboard)](https://github.com/yourusername/heart-disease-dashboard/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10+-red)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Heart_2020-green)](https://www.cdc.gov/brfss/annual_data/annual_2020.html)

A powerful, interactive Streamlit dashboard for exploratory data analysis (EDA), statistical testing, and machine learning on a cleaned heart disease dataset from 2020. Analyze key health indicators, visualize trends, and build predictive models to understand heart disease risk factors.

## Table of Contents 📑
- [About](#about)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## About ℹ️
This project provides an interactive dashboard built with **Streamlit** to analyze the **2020 Heart Disease Dataset**. It offers tools for:
- **Exploratory Data Analysis (EDA)**: Visualize numerical and categorical features with histograms, box plots, pie charts, and correlation heatmaps.
- **Statistical Testing**: Perform normality tests (Kolmogorov-Smirnov), Mann-Whitney U tests, and Chi-Square tests to uncover relationships with heart disease.
- **Machine Learning**: Train and evaluate models like Logistic Regression, Random Forest, SVM, Gradient Boosting, MLP, and KNN, with optional SMOTE balancing and hyperparameter tuning.
- **Data Preprocessing**: Handle missing values, outliers, and encode categorical variables for robust analysis.

The dashboard is designed for researchers, data scientists, and healthcare enthusiasts to explore heart disease risk factors interactively.

## Features ✨
- **Data Overview**: View dataset structure, missing values, and duplicates with a user-friendly interface.
- **EDA Visualizations**:
  - Distribution plots (histograms, KDE, box plots, pie charts).
  - Feature correlations with heart disease.
  - Custom visualizations for BMI categories and more.
- **Statistical Tests**:
  - Kolmogorov-Smirnov for normality.
  - Mann-Whitney U for numerical feature comparisons.
  - Chi-Square for categorical feature associations.
- **Machine Learning**:
  - Train multiple classifiers with PCA for dimensionality reduction.
  - Optional SMOTE for handling imbalanced data.
  - GridSearchCV for hyperparameter tuning.
  - Visualized confusion matrices and classification reports.
- **Interactive Interface**: Select features, models, and preprocessing options via Streamlit's intuitive UI.
- **Outlier Removal**: Apply IQR-based outlier removal for numerical features like BMI and PhysicalHealth.

## Demo 🎥
Check out the dashboard in action:

![Demo GIF](images/demo.gif)

[Live Demo (if hosted)](https://your-hosted-url.streamlit.app/) *(Replace with your hosted URL if applicable)*

## Installation 🛠️

Follow these steps to set up the project locally:

### Prerequisites
- Python 3.8 or higher
- Git
- A code editor (e.g., VS Code, PyCharm)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/heart-disease-dashboard.git
