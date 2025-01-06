import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
import zipfile
import os

# Downloading Dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(data_url)
zip_path = "smsspamcollection.zip"

with open(zip_path, "wb") as file:
    file.write(response.content)

# Extracting the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./")

os.remove(zip_path)  # Clean up the zip file

# Reading the dataset
data = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'], header=None)


data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Saving the results to a file
with open("model_results.txt", "w") as file:
    file.write(f"Accuracy: {accuracy}\n")
    file.write("\nClassification Report:\n")
    file.write(classification_report(y_test, y_pred))
    file.write("\nConfusion Matrix:\n")
    file.write(str(confusion_matrix(y_test, y_pred)))
