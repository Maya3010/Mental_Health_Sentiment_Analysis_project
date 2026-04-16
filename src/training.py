import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_ingestion import import_data 
from src.data_preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib


#Data Loading
sentiment_df = import_data()
print("Data Loaded Successfully!")


#Data Cleaning
sentiment_df['statement'] = clean_text(sentiment_df['statement'])
print("Data Preprocessed Successfully!")


#Train Test Split
X = sentiment_df['statement']
y = sentiment_df['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#Vectorization
tfidf = TfidfVectorizer(
    ngram_range=(1,3),
    max_features=10000,
    min_df=2,
    max_df=0.9
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("Vectorization Completed!")


#Model Training
svm = LinearSVC(C=0.1,class_weight='balanced',random_state=42)
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)
print("Model Trained Successfully!")


#Evaluation
print(classification_report(y_test, y_pred_svm))


#Model Saving
joblib.dump(svm, "artifacts/svm_model.pkl")
joblib.dump(tfidf, "artifacts/tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully!")