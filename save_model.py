import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import joblib  # Import joblib for saving the model

# --- Your Preprocessing Functions (Re-used for consistency) ---
nltk.download('stopwords', quiet=True)
stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stopwords]
    return " ".join(no_stopword_text)


def clean_text(text):
    text = text.lower()
    # (Your full clean_text regex block goes here)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def stemming(sentence):
    # This must be run on the text AFTER cleaning and stopword removal
    stemmed_sentence = " ".join([stemmer.stem(word)
                                for word in sentence.split()])
    return stemmed_sentence


def full_preprocess(text):
    """Combines all preprocessing steps."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text


# --- Load and Preprocess Data ---
df = pd.read_csv('train.csv')
labels = df.columns[2:]

df['comment_text'] = df['comment_text'].apply(full_preprocess)

X = df['comment_text']
# Drop 'id' if still present
y = df.drop(columns=['comment_text', 'id'], axis=1)

# Train the final pipeline
SVC_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3))),
    ('svc_model', OneVsRestClassifier(
        LinearSVC(random_state=42, dual=False, C=0.4), n_jobs=-1))
])

print("Training SVC model...")
SVC_pipeline.fit(X, y)
print("Training complete.")

# --- Save the fitted pipeline and labels ---
joblib.dump(SVC_pipeline, 'toxic_classifier_svc.joblib')
joblib.dump(labels.tolist(), 'labels.joblib')
print("Model and labels saved successfully.")
