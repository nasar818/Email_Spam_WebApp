from flask import Flask, render_template, request
import joblib
from preprocess import clean_text

app = Flask(__name__)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email"]
    cleaned = clean_text(email_text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    result = "SPAM" if prediction == 1 else "NOT SPAM"
    return render_template("index.html", prediction=result, input=email_text)

if __name__ == "__main__":
    app.run(debug=True)
