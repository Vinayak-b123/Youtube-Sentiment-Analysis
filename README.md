Overview

This project analyzes YouTube comments and classifies them into sentiment categories using Natural Language Processing (NLP) and Machine Learning techniques. The objective was to build an end-to-end text analytics pipeline, compare multiple feature engineering approaches, and identify the best-performing classification model.

Problem Statement

YouTube videos generate thousands of comments containing valuable user feedback and opinions. Manually analyzing such large volumes of text is impractical. This project automates sentiment classification to help understand audience reactions at scale.

Dataset
Source: YouTube comments dataset
Total Comments: 50,000+ comments
Target Variable: Sentiment Category
Preprocessing included text cleaning, URL removal, special character handling, and normalization.
Project Workflow
1. Data Preprocessing
Removed URLs, emojis, and unnecessary characters
Handled missing values
Performed text normalization and cleaning
Prepared data for vectorization
2. Feature Engineering

Implemented and compared multiple text representation techniques:

Bag of Words (BoW)
CountVectorizer
Tuned n-gram ranges and vocabulary size
TF-IDF Vectorization
Experimented with different feature limits and n-gram settings
Sentence Embeddings
Sentence Transformers
Generated dense semantic embeddings for comments
3. Handling Class Imbalance
Applied SMOTE oversampling on training data
Evaluated impact on model performance
4. Model Development

Trained and evaluated multiple machine learning models:

Logistic Regression
Linear SVM
Random Forest
XGBoost
LightGBM
Stacking Ensemble
5. Hyperparameter Optimization

Used Optuna and cross-validation to optimize:

Logistic Regression
XGBoost
LightGBM
Results
Model	Accuracy
Logistic Regression 68%
Linear SVM 68%
XGBoost	72%
LightGBM	69%
Stacking Ensemble	72%

Best Model: Stacking Ensemble
Best Accuracy: 72%

Tech Stack
Programming Language: Python

Libraries:  Pandas, NumPy, Scikit-learn, NLTK, Sentence Transformers, XGBoost, LightGBM, Optuna, Imbalanced-learn, Matplotlib, Seaborn.
Key Learnings

Built a complete NLP classification pipeline from data preprocessing to model deployment.
Compared sparse vectorization methods against transformer-based embeddings.
Applied hyperparameter optimization using Optuna.
Evaluated ensemble learning techniques such as stacking.
Worked with imbalanced datasets using SMOTE.

Future: deploying it through flask api
