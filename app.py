## app.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(filepath: str) -> pd.DataFrame:
    """Load and clean the dataset."""
    df = pd.read_csv(filepath)
    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})  # If using text labels
    return df


def train_model(X_train, y_train):
    """Train and return a logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Print evaluation metrics."""
    y_pred = model.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))


def predict_sample(text: str, model, vectorizer):
    """Predict label for a sample text."""
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    print(f"\n🔍 Sample Prediction:\nInput: \"{text}\"\nPrediction: {label}")


def main():
    # 🔹 Load dataset
    df = load_data("fake_or_real_news.csv")

    # 🔹 Split data
    X = df['text'].astype(str)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 🔹 TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 🔹 Model training
    model = train_model(X_train_vec, y_train)

    # 🔹 Model evaluation
    evaluate_model(model, X_test_vec, y_test)

    # 🔹 Sample prediction
    sample_texts = [
        "The president announced a new economic reform package today.",
        "Click this link to win a free iPhone right now!",
        "Aliens are controlling the weather according to leaked documents."
    ]
    for text in sample_texts:
        predict_sample(text, model, vectorizer)


if __name__ == "__main__":
    main()
