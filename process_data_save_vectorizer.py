from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import nltk

# Load the trained model
classifier = load('sentiment_model.joblib')

# Load the TfidfVectorizer used during training
vectorizer = TfidfVectorizer(max_features=3000)
vectorizer.fit([])  # Dummy fit, as we need to access vocabulary_ attribute
vectorizer.vocabulary_ = classifier.named_steps['tfidf'].vocabulary_

# Save the vectorizer using joblib
dump(vectorizer, 'vectorizer.joblib')
