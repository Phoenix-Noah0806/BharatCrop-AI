# 🌾 BharatCropAI
### Intelligent Crop Production Analysis & Prediction Using Machine Learning

<p align="center">
  <strong>Turning historical agricultural data into meaningful production insights.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Database-MongoDB%20Atlas-green?logo=mongodb&logoColor=white" />
  <img src="https://img.shields.io/badge/Models-3-success" />
  <img src="https://img.shields.io/badge/Dataset-246K%2B%20Records-blueviolet" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

## 📌 Overview

**BharatCropAI** is a machine learning-based agricultural data analysis and crop production prediction system built using Python, Scikit-learn and MongoDB Atlas.

The project analyses over **246,000 historical agricultural records covering 124 crop varieties across 33 Indian states and union territories**.

It implements both regression and classification approaches to explore crop production patterns and generate predictions from agricultural inputs such as state, district, crop, season, year and cultivated area.

Rather than relying on a single algorithm, BharatCropAI implements and evaluates three machine learning models:

- 🌳 Random Forest Regressor – Predicts numerical crop production.
- 🌲 Random Forest Classifier – Classifies production into Low or High.
- 🌿 Decision Tree Classifier – Predicts the Low or High production category.

The project also integrates MongoDB Atlas to store the agricultural dataset, model evaluation metrics and generated prediction history.

---

## ✨ Key Highlights

| Feature | Implementation |
|---|---|
| Dataset size | 246,091 records |
| Crop varieties | 124 |
| States/UTs | 33 |
| Districts | 646 |
| Data preprocessing | Pandas and NumPy |
| Categorical encoding | OneHotEncoder |
| Regression | Random Forest Regressor |
| Classification | Random Forest and Decision Tree |
| Best classification accuracy | **91.13%** |
| Database | MongoDB Atlas |
| Prediction storage | PyMongo |
| Model persistence | Joblib |

### 🏆 Model Performance at a Glance

| Model | Accuracy / R² | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest Regressor | R²: 0.8122 | — | — | — |
| Random Forest Classifier | 85.07% | 87.43% | 80.65% | 83.90% |
| Decision Tree Classifier | **91.13%** | **92.02%** | **89.36%** | **90.67%** |

*Regression MAE: 235,603.85 recorded production units.*

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Dataset](#-dataset)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Machine Learning Methodology](#-machine-learning-methodology)
- [Model Evaluation](#-model-evaluation)
- [MongoDB Integration](#-mongodb-integration)
- [Prediction Workflow](#-prediction-workflow)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Project](#-running-the-project)
- [Sample Prediction](#-sample-prediction)
- [Key Learnings](#-key-learnings)
- [Limitations and Future Improvements](#-limitations-and-future-improvements)
- [Author](#-author)

---

## 🎯 Problem Statement

Agricultural production depends on several interconnected factors, including geographical location, crop variety, season, cultivated area and historical agricultural patterns.

Analysing large volumes of historical crop production data manually can make it difficult to identify trends and estimate production patterns.

BharatCropAI addresses this problem by applying machine learning techniques to historical Indian agricultural data.

The system explores two prediction problems:

**1. Regression Problem**

How much production can be expected for a given set of agricultural inputs?

**2. Classification Problem**

Will the production be relatively Low or High compared with the historical production benchmark for that crop?

The project demonstrates how historical agricultural datasets can be transformed into machine learning-ready data and used to generate predictions.

---

## 🎯 Project Objectives

- Analyse large-scale Indian crop production data.
- Identify missing values, duplicates and data quality issues.
- Perform data cleaning and feature preprocessing.
- Store agricultural records in MongoDB Atlas.
- Encode categorical variables for machine learning.
- Implement regression and classification algorithms.
- Evaluate models using appropriate performance metrics.
- Generate predictions using saved machine learning models.
- Store model evaluation results and prediction history in MongoDB.

---

## 📊 Dataset

**Dataset:** Crop Production in India

**Source:** Kaggle

🔗 [Access the Crop Production in India Dataset](https://www.kaggle.com/datasets/abhinand05/crop-production-in-india)

### Dataset Description

The dataset contains historical crop production information from different Indian states and districts.

| Attribute | Description |
|---|---|
| State_Name | Indian state or union territory |
| District_Name | District where the crop was cultivated |
| Crop_
