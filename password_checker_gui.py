# password_checker_gui.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# --- Load dataset ---
data = pd.read_csv("passwords.csv")

# --- Clean dataset ---
data = data.dropna(subset=['password', 'strength'])

# --- Convert numeric strength to categories ---
def convert_strength(val):
    if val <= 4:
        return "Weak"
    elif val <= 7:
        return "Medium"
    else:
        return "Strong"

data['strength'] = data['strength'].apply(convert_strength)

# --- Features and labels ---
X = data['password']
y = data['strength']

# --- Convert text to numbers ---
vectorizer = CountVectorizer(analyzer='char')
X = vectorizer.fit_transform(X)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train better model ---
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --- Accuracy ---
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# --- GUI Function ---
def check_strength():
    password = password_entry.get()
    
    if not password:
        messagebox.showwarning("Input Error", "Please enter a password")
        return
    
    vec = vectorizer.transform([password])
    result = model.predict(vec)[0]

    # Color coding
    if result == "Weak":
        result_label.config(text="Weak", fg="red")
    elif result == "Medium":
        result_label.config(text="Medium", fg="orange")
    else:
        result_label.config(text="Strong", fg="green")

# --- GUI Setup ---
root = tk.Tk()
root.title("Password Strength Checker")
root.geometry("800x450")

tk.Label(root, text="Enter Password", font=("Arial", 14)).pack(pady=10)

password_entry = tk.Entry(root, width=30, font=("Arial", 12), show="*")
password_entry.pack(pady=5)

tk.Button(root, text="Check Strength", command=check_strength, font=("Arial", 12)).pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=20)

root.mainloop()