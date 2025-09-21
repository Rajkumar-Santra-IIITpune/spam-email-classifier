from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

app = Flask(__name__)

# Load dataset
data = pd.read_csv('spam_ham_dataset.csv')
data = data.dropna(subset=['text', 'label'])
train_emails = data['text'].values
train_labels = data['label_num'].values  # 0=ham, 1=spam


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
    prediction = classifier.predict(X_test)[0]  # numeric: 0 or 1
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', prediction=result, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)
