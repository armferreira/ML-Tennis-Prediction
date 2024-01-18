# Predictive Modeling for ATP Tennis Matches

## Project Overview

This project focuses on predicting the number of sets in ATP tennis matches using various machine learning techniques. We employ a range of models including logistic regression, decision trees, neural networks, and AutoML approaches. Our goal is to accurately predict whether a match will be best of 3 or best of 5 sets, using historical match data.

## Required Libraries
- `dplyr`: Data manipulation and transformation
- `car`: Applied regression functions, diagnostic plots for regression models, outlier tests
- `ggplot2`: Data visualization library
- `ggeffects`: Regression visualization library
- `caret`: Classification and regression training
- `RSBID`: SMOTE-NC method for over-sampling
- `pROC`: Tools for ROC analysis, plotting ROC curves, computing AUC
- `partykit`: Creation and visualization of tree-based models
- `rpart`: Recursive partitioning and regression trees
- `rpart.plot`: Plots decision trees created with 'rpart'
- `fastDummies`: Quickly create dummy variables from categorical data
- `neuralnet`: Training neural networks using backpropagation
- `keras`: High-level neural networks API based on TensorFlow
- `tensorflow`: Interface to TensorFlow deep learning library
- `h2o`: Platform for building models with AutoML

## Project Structure

### 0. Data Pre-processing
- Seed setting for reproducibility
- Data division based on match type (best of 5 or 3 sets)
- Data splitting into training and testing sets
- Over-sampling and under-sampling techniques applied

### 1. Logistic Regression
- Model creation, evaluation, and optimization based on AIC
- Multicollinearity testing and confusion matrix analysis
- ROC curve plotting and AUC calculation

### 2. Decision Trees
- Decision tree creation using `rpart` and `caret`
- Model accuracy measurement and visualization
- Additional methods like bagging, random forest, and XGBoost
- Variable importance analysis

### 3. Neural Networks
- Dummy variable creation for categorical data
- Neural network training using `neuralnet` and `keras`
- Model evaluation and comparison

### 4. AutoML with H2O
- Initialization and data preparation for H2O
- AutoML model training and evaluation
- Leaderboard analysis and performance metrics
- Prediction and confusion matrix for the best model

## Usage
- Ensure all required libraries are installed
- Use the dataset 'atp'.
- Run each section in sequence for the complete analysis
