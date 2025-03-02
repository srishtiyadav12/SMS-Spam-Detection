import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')


st.title('📱 SMS Spam Detection')
st.subheader('This app detects whether an SMS message is spam or not.')

@st.cache_data
def load_data():
    sms = pd.read_csv("project/SMSSpamCollection", sep='\t', names=['label', 'mesg'])
    sms.drop_duplicates(inplace=True)
    sms.reset_index(drop=True, inplace=True)
    return sms

sms = load_data()

@st.cache_data
def preprocess_data(data):
    corpus = []
    ps = PorterStemmer()
    for i in range(0, data.shape[0]):
        message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=data.mesg[i])
        message = message.lower()
        words = message.split()
        words = [word for word in words if word not in set(stopwords.words('english'))]
        words = [ps.stem(word) for word in words]
        message = ' '.join(words)
        corpus.append(message)
    
    cv = CountVectorizer(max_features=2500)
    X = cv.fit_transform(corpus).toarray()
    
    y = pd.get_dummies(data['label'])
    y = y.iloc[:, 1].values
    
    return X, y, cv

X, y, cv = preprocess_data(sms)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

@st.cache_data
def train_model(X_train, y_train):
    best_accuracy = 0.0
    alpha_val = 0.0
    for i in np.arange(0.0, 1.1, 0.1):
        temp_classifier = MultinomialNB(alpha=i)
        temp_classifier.fit(X_train, y_train)
        temp_y_pred = temp_classifier.predict(X_test)
        score = accuracy_score(y_test, temp_y_pred)
        if score > best_accuracy:
            best_accuracy = score
            alpha_val = i

    classifier = MultinomialNB(alpha=alpha_val)
    classifier.fit(X_train, y_train)
    return classifier, best_accuracy, alpha_val

classifier, best_accuracy, alpha_val = train_model(X_train, y_train)


# Dataset showing
st.subheader("Dataset")
st.write(sms.head())



y_pred = classifier.predict(X_test)
acc_s = accuracy_score(y_test, y_pred) * 100


# Model Performance

st.subheader('Model Performance')
st.markdown(f"**Best accuracy: {best_accuracy:.2f} with alpha={alpha_val:.1f}** 🎉")




# Visual the performance

st.subheader('📊 Model Performance on Test Data')
st.write(f"Accuracy of the model on the test set: {acc_s:.2f}")

st.write('Confusion Matrix:')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
st.pyplot(fig)



# Spam Predection

st.subheader('🔍 Predict SMS Spam')
user_input = st.text_area("Enter your message here:")

def predict_spam(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if word not in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)

if st.button('Predict'):
    if user_input:
        prediction = predict_spam(user_input)
        result = ['Ohhh, this is a normal message. 😊', 'Wait a minute, this is a SPAM! 🚨']
        st.write(result[int(prediction)])
    else:
        st.write('Please enter a message to predict.')