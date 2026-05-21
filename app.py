import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Load model
model = pickle.load(open('spam_model.pkl', 'rb'))

# Load vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Stemming object
ps = PorterStemmer()


# Preprocessing function
def preprocess(text):

    text = text.lower()

    text = re.sub('[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]

    return ' '.join(words)


# Streamlit UI
st.title('📧 Spam Email Detection System')

st.write("Prof. Mariya Celin")

st.markdown("### Team Members")
st.write("""
- Mateti Karthik
- Avaneesh Reddy
""")

st.write('Enter a message below to check whether it is Spam or Ham.')

# User input
message = st.text_area('Enter Message')


# Predict button
if st.button('Predict'):

    processed_message = preprocess(message)

    vector_input = tfidf.transform([processed_message])

    prediction = model.predict(vector_input)

    if prediction[0] == 1:
        st.error('Spam Message')
    else:
        st.success('Ham Message')
