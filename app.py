from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

app = Flask(__name__)


try:
    data = pd.read_csv('spam_ham_dataset.csv') 
    data = data.dropna(subset=['text', 'label']) 
except FileNotFoundError:
    raise FileNotFoundError("Dataset file 'spam_ham_dataset.csv' not found. Please place it in the project directory.")
except Exception as e:
    raise Exception(f"Error loading dataset: {e}")


train_emails = data['text'].values
train_labels = data['label'].values


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_emails)
classifier = MultinomialNB()
classifier.fit(X_train, train_labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    X_test = vectorizer.transform([email_text])
    prediction = classifier.predict(X_test)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', prediction=result, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)
