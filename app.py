
from flask import Flask, request, render_template, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

emails = [
    "Verify your account now to avoid suspension.",
    "Join us for the team meeting tomorrow at 3.",
    "You've won a $1000 Walmart gift card. Click here.",
    "Reminder: doctor's appointment at 4 PM."
]
labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

stats = {"total": 0, "phishing": 0, "safe": 0}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", email_result=None, url_result=None, stats=stats)

@app.route("/check_email", methods=["POST"])
def check_email():
    text = request.form.get("email", "")
    pred = model.predict(vectorizer.transform([text]))[0]
    conf = model.predict_proba(vectorizer.transform([text]))[0][pred] * 100
    stats["total"] += 1
    if pred == 1:
        stats["phishing"] += 1
        result = f"❌ Phishing Detected! ({conf:.2f}%)"
    else:
        stats["safe"] += 1
        result = f"✅ Safe Message. ({conf:.2f}%)"
    return render_template("index.html", email_result=result, url_result=None, stats=stats)

@app.route("/check_url", methods=["POST"])
def check_url():
    url = request.form.get("url", "")
    pred = model.predict(vectorizer.transform([url]))[0]
    conf = model.predict_proba(vectorizer.transform([url]))[0][pred] * 100
    stats["total"] += 1
    if pred == 1:
        stats["phishing"] += 1
        result = f"❌ Suspicious URL Detected! ({conf:.2f}%)"
    else:
        stats["safe"] += 1
        result = f"✅ URL Looks Safe. ({conf:.2f}%)"
    return render_template("index.html", email_result=None, url_result=result, stats=stats)

if __name__ == "__main__":
    app.run(debug=True)
