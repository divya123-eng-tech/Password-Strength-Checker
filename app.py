from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

# --- Load and train model ---
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
data = pd.read_csv(BASE_DIR / "passwords.csv")
data = data.dropna(subset=['password', 'strength'])

def convert_strength(val):
    if val <= 4:
        return "Weak"
    elif val <= 7:
        return "Medium"
    else:
        return "Strong"

data['strength'] = data['strength'].apply(convert_strength)

X = data['password']
y = data['strength']

vectorizer = CountVectorizer(analyzer='char')
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

# --- API route ---
@app.route('/check', methods=['POST'])
def check_password():
    data = request.get_json()   # get data from frontend
    if not data or 'password' not in data:
        return jsonify({"error": "Password is required"}), 400

    password = data.get('password')   # extract password
    if not password:
        return jsonify({"error": "Password is required"}), 400

    vec = vectorizer.transform([password])
    result = model.predict(vec)[0]

    return jsonify({"strength": result})
# --- Run server ---
if __name__ == '__main__':
    app.run(debug=False, port=5001)
