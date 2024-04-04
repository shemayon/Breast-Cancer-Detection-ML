# Breast Cancer Detection Using Machine Learning: An Investigation and Algorithm Development

This repository contains code and documentation for a machine learning project focused on the detection of breast cancer. The project aims to develop an accurate and efficient algorithm for detecting breast cancer using available data.

## Introduction

Breast cancer is one of the most prevalent types of cancer, affecting millions of women worldwide. It is characterized by the uncontrolled growth of abnormal cells in the breast tissue, which can spread to other parts of the body if not detected and treated early. Early detection and accurate diagnosis are crucial for effective treatment and improved survival rates.

## Objective

The primary objective of this project is to investigate key features that affect breast cancer and develop a machine learning algorithm for accurate detection. The specific objectives are as follows:
- To analyze and identify important features associated with breast cancer.
- To explore various machine learning algorithms and determine the optimal one for breast cancer detection.
- To develop and evaluate a machine learning model that can predict breast cancer with high accuracy.

## Key Features
- **Dataset Exploration**: An in-depth exploration of the breast cancer dataset to understand its structure and features.
- **Feature Analysis**: Investigation into key features such as clump thickness, uniformity of cell size, uniformity of cell shape, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin, normal nucleoli, and mitoses.
- **Model Building**: Implementation of various machine learning algorithms including Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Naive Bayes, XGBoost, Multilayer Perceptron (MLP), and Artificial Neural Network (ANN).
- **Evaluation**: Comparison of model performance metrics such as accuracy, precision, recall, and F1-score.
- **Optimization**: Tuning hyperparameters to improve model performance and reduce overfitting.
- **Visualization**: Visual representation of data distributions, feature importance, and model evaluation metrics.
- **Summary**: Conclusion and recommendations based on the findings of the analysis and model evaluation.

## Technologies Used
- Python
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, keras

## Dataset
The project uses the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) from the UCI Machine Learning Repository. This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, and the target variable is the diagnosis (M = malignant, B = benign).

## Files
- `Breast_Cancer_Detection.ipynb`: Jupyter Notebook containing code for data analysis, model building, and evaluation.
- `breast_cancer_data.csv`: CSV file containing the breast cancer dataset.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/shemayon/Breast-Cancer-Detection.git
   ```
2. If you cant clone, download the notebook provided and try to run it !! 
3. Open and run the Jupyter Notebook `Breast_Cancer_Detection.ipynb` to explore the analysis, models, and results.

## Results
- **Logistic Regression**: Training Accuracy (96.87%), Testing Accuracy (97.85%)
- **Decision Tree**: Training Accuracy (98.00%), Testing Accuracy (92.85%)
- **Random Forest**: Training Accuracy (98.23%), Testing Accuracy (95.70%)
- **SVM**: Training Accuracy (99.00%), Testing Accuracy (94.20%)
- **KNN**: Training Accuracy (98.00%), Testing Accuracy (97.85%)
- **Naive Bayes**: Training Accuracy (96.19%), Testing Accuracy (95.70%)
- **XGBoost**: Training Accuracy (98.77%), Testing Accuracy (97.14%)
- **MLP**: Training Accuracy (95.30%), Testing Accuracy (95.00%)
- **ANN**: Training Accuracy (99.00%), Testing Accuracy (96.40%)

## Conclusion
- Logistic Regression and K-Nearest Neighbors (KNN) performed consistently well on both training and testing data.
- Decision Tree showed a noticeable drop in performance on the testing set compared to training.
- Support Vector Machine (SVM) achieved the highest training accuracy but a lower testing accuracy, indicating potential overfitting.
- Random Forest and XGBoost performed well but had slightly lower testing accuracy compared to LR and KNN.
- Multilayer Perceptron (MLP) had a good training accuracy but a lower testing accuracy, suggesting some degree of overfitting.
- Artificial Neural Network (ANN) also performed well with a high training accuracy, but slightly lower than LR and KNN on the testing set.
