import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv('data/synthetic_address_subset_data.csv')

# Preprocessing: fill missing values in 'address' column
df['address'] = df['address'].fillna('')

# Define features (X) and target (y)
X = df['address']
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using character-level TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Build and train the balanced classifier
classifier = LogisticRegression(class_weight='balanced')
classifier.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer
joblib.dump(classifier, 'address_classifier_model.pkl')
joblib.dump(vectorizer, 'address_vectorizer.pkl')

# Make predictions and show report
y_pred = classifier.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))
