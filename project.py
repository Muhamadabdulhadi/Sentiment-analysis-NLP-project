import pandas as pd

# Splits data into training and testing
from sklearn.model_selection import train_test_split

# Converts text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

# Classifier
from sklearn.linear_model import LogisticRegression

# Pipeline (text → TF-IDF → classifier)
from sklearn.pipeline import Pipeline

# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Save model
import joblib


# Load dataset
df = pd.read_csv(r"C:\Users\muhamad abdul hadi\Downloads\IMDB Dataset.csv")

print(df.head())


# Convert labels to numbers
df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

X = df['review']
y = df['sentiment']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )),
    ('clf', LogisticRegression())
])


# Train model
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ⭐ Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Save model
joblib.dump(model, "sentiment_model.pkl")


# ⭐ Demo prediction
sample_review = ["This movie was absolutely fantastic and emotional"]
prediction = model.predict(sample_review)[0]

# ⭐ Prediction probability
probability = model.predict_proba(sample_review)[0]
confidence = max(probability)

print("\nSample Review Prediction:")
print("Review:", sample_review[0])
print("Sentiment:", "Positive 😊" if prediction == 1 else "Negative 😞")
print("Confidence Score:", confidence)