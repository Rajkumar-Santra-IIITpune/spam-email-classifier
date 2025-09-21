import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

@st.cache_data
def load_data():
    data = pd.read_csv('spam_ham_dataset.csv')
    data = data.dropna(subset=['text', 'label_num'])  # use numeric label
    return data

@st.cache_data
def train_model(data):
    train_emails = data['text'].values
    train_labels = data['label_num'].values  # 0=ham, 1=spam
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_emails)
    classifier = MultinomialNB()
    classifier.fit(X_train, train_labels)
    return vectorizer, classifier

def main():
    st.title("Spam Mail Classifier")
    st.write("Enter the email text below to check if it is Spam or Not Spam.")

    data = load_data()
    vectorizer, classifier = train_model(data)

    email_text = st.text_area("Email Text", height=150)

    if st.button("Predict"):
        if email_text.strip() == "":
            st.warning("Please enter some email text to classify.")
        else:
            X_test = vectorizer.transform([email_text])
            prediction = classifier.predict(X_test)[0]  # 0 or 1
            result = "Spam" if prediction == 1 else "Not Spam"
            st.success(f"Prediction: {result}")

if __name__ == "__main__":
    main()
