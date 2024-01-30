# sentiment_app.py
from flask import Flask, render_template, request
from pathlib import Path
from joblib import load
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
model_path = Path(__file__).resolve().parent / 'sentiment_model.joblib'
vectorizer_path = Path(__file__).resolve().parent / 'vectorizer.joblib'

# Check if the files exist
if not model_path.exists():
    print(f"Error: Model file does not exist at {model_path}")
if not vectorizer_path.exists():
    print(f"Error: Vectorizer file does not exist at {vectorizer_path}")

# Load the trained model and vectorizer
classifier = load(model_path)
vectorizer = load(vectorizer_path)

# Flask Routes
# ... (rest of the code)


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

# sentiment_app.py

# ... (previous code)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']

        # Preprocess the input review
        processed_review = preprocess_text(review)

        # Use the trained model and vectorizer to make a prediction
        prediction = classifier.predict(vectorizer.transform([processed_review]))[0]

        # Capitalize the first letter of the prediction
        prediction = prediction.capitalize()

        return render_template('index.html', review=review, prediction=prediction)

# ... (rest of the code)


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    processed_text = ' '.join(filtered_tokens)
    return processed_text

if __name__ == '__main__':
    app.run(debug=True)
