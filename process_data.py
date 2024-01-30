# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Load the IMDb dataset
# Replace 'your_dataset.csv' with the actual filename and adjust column names accordingly
data = pd.read_csv('IMDB Dataset.csv')

# Download NLTK resources
nltk.download('stopwords')

# Preprocess the data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Apply preprocessing to the text column
data['preprocessed_text'] = data['text_column'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['preprocessed_text'], data['sentiment_label'], test_size=0.2, random_state=42
)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)  # You can adjust the number of features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model on the testing set
predictions = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

# Print accuracy and classification report
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save the trained model for later use
dump(classifier, 'sentiment_model.joblib')
