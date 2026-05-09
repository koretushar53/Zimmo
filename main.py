from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

ps = PorterStemmer()

app = Flask(__name__)

model = pickle.load(open('model1.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer2.pkl', 'rb'))


def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    transformed_sms = transform_text(message)

    print("Original:", message)
    print("Processed:", transformed_sms)

    vector_input = vectorizer.transform([transformed_sms])

    prediction = model.predict(vector_input)[0]

    print("Prediction:", prediction)

    if prediction == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    return render_template('index.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)