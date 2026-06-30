# password_checker_run.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("passwords.csv")

# Clean dataset: remove empty passwords or strengths
data = data.dropna(subset=['password', 'strength'])

# Convert strength to string (for ML)
data['strength'] = data['strength'].astype(str)

# Features and target
X = data['password']
y = data['strength']

# Convert passwords to numeric features
vectorizer = CountVectorizer(analyzer='char')
X = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")

# --- Interactive password check ---
while True:
    new_password = input("Enter a password to check (or type 'exit' to quit): ")
    if new_password.lower() == 'exit':
        print("Exiting Password Checker. Goodbye!")
        break
    new_vec = vectorizer.transform([new_password])
    strength = model.predict(new_vec)[0]
    print(f"Predicted Password Strength: {strength}\n")