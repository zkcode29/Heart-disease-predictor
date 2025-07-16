# Heart Disease Prediction Web App

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

This project is repository.

You can copy and paste this text directly into the `README.md` file on GitHub. Just click the little "pencil" icon to edit the file, paste this content, and save the changes.

---

# Heart Disease Prediction Application

This project is an end-to-end machine learning application that predicts the likelihood of a patient having heart disease based on their medical attributes. The final model is deployed as an interactive web application using Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app) 
*(**Note:** Replace `https://your-app-url.streamlit.app` with your actual deployed Streamlit app URL!)*

## Table of Contents
- [Project Objective](#project-objective)
- [How it Works](#how-it-works)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Technologies Used](#technologies-used)
- [How to Run This Project Locally](#how-to-run-this-project-locally)
- [Project Structure](#project-structure)

## Project Objective
The primary goal is to provide a simple, accessible tool that leverages machine learning to assist healthcare professionals in making faster, data-driven decisions regarding a patient's heart disease risk. This serves as a clinical decision-support system, aiming for early diagnosis and intervention.

##  How it Works
The application functions as follows:
1.  A user (e.g., a doctor) enters a patient's medical information into the web interface.
2.  The input data is processed and scaled in the same way the model was trained.
3.  The trained Random Forest model predicts the probability an end-to-end machine learning application that predicts the likelihood of a patient having heart disease based on their medical attributes. The final model is deployed as a user-friendly, interactive web application using Streamlit.

[![Heart Disease Prediction App](https://i.imgur.com/your-screenshot-url.png)](https://your-app-url.streamlit.app/)
*Click the image above to visit the live application.*

---

## Table of Contents
* [Project Objective](#-project-objective)
* [How It Works](#-how-it-works)
* [Dataset](#-dataset)
* [Technical Workflow](#-technical-workflow)
* [How to Run This Project Locally](#-how-to-run-this-project-locally)
* [Technologies Used](#-technologies-used)
* [Project Structure](#-project-structure)

---

## Project Objective

The goal of this project is to build a reliable system that can assist healthcare professionals in making faster, more data-driven decisions. By providing a patient's key medical data, the application returns a risk assessment for heart disease, serving as a powerful clinical decision-support tool for early diagnosis and intervention.

---

##  How It Works

The web application provides an intuitive interface where a user can input 13 key medical features of a patient.

1.  **User Input:** The user enters the patient's data using the sliders and dropdown menus in the sidebar.
2.  **Prediction:** Upon clicking the "Predict" button, the data is sent to a pre-trained machine learning model hosted in the cloud.
3.  **Result:** The model returns a prediction ("Low Risk" or "High Risk") along with a confidence score, which is immediately displayed to the user.

---

##  Dataset

This project uses the well-known of the patient having heart disease.
4.  The application displays the result ("High Risk" or "Low Risk") along with the confidence score.

##  Dataset
This project uses the well-known **Heart Disease Dataset** from the UCI Machine Learning Repository.
- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Key Features:** `age`, `sex`, `chest pain type`, `resting blood pressure`, `cholesterol`, `max heart rate`, and other relevant medical indicators.

##  Workflow
The project was built following a standard machine learning workflow:
1.  **Data Preprocessing:** Handled missing values (imputed with the median) and ensured data was clean and numerical.
2.  **Exploratory Data Analysis (EDA):** Used visualizations like a correlation heatmap to understand feature relationships and identify key predictors.
3.  **Model Training & Comparison:** Trained and evaluated three different models: Logistic Regression, Decision Tree, and Random Forest.
4.  **Model Selection:** The **Random Forest Classifier** was selected as the best-performing model based on ROC-AUC score and overall accuracy.
5.  **Hyperparameter Tuning:** Used `GridSearchCV` to find the optimal settings for the Random Forest model, further enhancing its predictive power.
6.  **Deployment:** Built an interactive web application with Streamlit and deployed it on Streamlit Community Cloud.

##  Technologies Used
- **Programming Language:** Python 3
- **Data Science Libraries:** Pandas **"Heart Disease Dataset"** from the UCI Machine Learning Repository.

*   **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
*   **Source Institution:** Cleveland Clinic Foundation
*   **Key Features Used:** `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `thalach` (max heart rate achieved), etc.

The full analysis and model development process can be viewed in the included Jupyter/Colab notebook: `Heart_Disease_Analysis.ipynb`.

---

## Technical Workflow

The project followed a standard end-to-end machine learning pipeline:

1.  **Data Cleaning & Preprocessing:** Handled missing values (marked as '?') by imputing them with the column median.
2.  **Exploratory Data Analysis (EDA):** Used visualizations like correlation heatmaps to understand feature relationships and identify key predictors.
3.  **Model Training & Comparison:** Trained and evaluated three different
