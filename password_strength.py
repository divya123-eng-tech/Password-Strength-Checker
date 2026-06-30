# password_strength.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("passwords.csv")  # your dataset file
print("Dataset loaded:")
print(data.head())

# Prepare data
X = data['password']
y = data['strength']

vectorizer = CountVectorizer(analyzer='char')
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Test new password
new_password = ["MyNewP@ss123"]
new_vec = vectorizer.transform(new_password)
print("Password Strength:", model.predict(new_vec)[0])