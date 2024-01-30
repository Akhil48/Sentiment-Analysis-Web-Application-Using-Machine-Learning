# train_model.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump

# Load the IMDb dataset
data = pd.read_csv('D:\Dataset.csv')

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess the data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Apply preprocessing to the text column
data['preprocessed_text'] = data['review'].apply(preprocess_text)  # Assuming 'review' is the correct column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['preprocessed_text'], data['sentiment'], test_size=0.2, random_state=42  # Assuming 'sentiment' is the correct column name
)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer using joblib
dump(classifier, 'sentiment_model.joblib')
dump(vectorizer, 'vectorizer.joblib')

