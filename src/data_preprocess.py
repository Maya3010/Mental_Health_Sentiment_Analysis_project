import pandas as pd
import nltk
nltk.download('punkt_tab')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_text(text):

    # If input is a single string (User Input)
    if isinstance(text,str):
        text = text.lower()

        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = word_tokenize(text)

        stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
        tokens = [word for word in tokens if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)

    # If input is pandas Series (training)
    elif isinstance(text, pd.Series):

        text = text.str.lower()

        text = text.apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
        text = text.apply(lambda x: re.sub(r'[^\w\s]', '', x))
        text = text.apply(lambda x: re.sub(r'\s+', ' ', x).strip())

        text = text.apply(word_tokenize)

        stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
        text = text.apply(lambda x: [word for word in x if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        text = text.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

        return text.apply(lambda x: ' '.join(x))

    else:
        raise ValueError("Input must be either a string or pandas Series")