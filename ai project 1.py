import os
import zipfile
import io
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import joblib

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_ZIP_NAME = "smsspamcollection.zip"
DATA_FILE_NAME = "SMSSpamCollection"
MODEL_PATH = "spam_model.joblib"

def download_and_extract(dest_dir="."):
    if os.path.exists(os.path.join(dest_dir, DATA_FILE_NAME)):
        return os.path.join(dest_dir, DATA_FILE_NAME)
    print("Downloading dataset...")
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extract(DATA_FILE_NAME, path=dest_dir)
    return os.path.join(dest_dir, DATA_FILE_NAME)

def load_dataset(path):
    # File format: label<TAB>message
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def train_and_save(df, model_path=MODEL_PATH):
    X = df['message']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    print("Evaluation on test set:")
    print("Accuracy:", metrics.accuracy_score(y_test, preds))
    print("Precision:", metrics.precision_score(y_test, preds))
    print("Recall:", metrics.recall_score(y_test, preds))
    print("F1:", metrics.f1_score(y_test, preds))
    print("ROC AUC:", metrics.roc_auc_score(y_test, probs))
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")
    return pipeline

def load_model(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def interactive_predict(pipeline):
    print("\nEnter email/SMS text to classify. Type 'quit' to exit.")
    while True:
        text = input("\nMessage: ").strip()
        if text.lower() in ('quit', 'exit'):
            break
        if not text:
            print("Please type a message.")
            continue
        prob = pipeline.predict_proba([text])[0][1]
        label = "SPAM" if prob >= 0.5 else "HAM"
        print(f"Prediction: {label} (spam probability: {prob:.2f})")

def main():
    # ensure dataset present
    data_path = download_and_extract(dest_dir=".")
    df = load_dataset(data_path)
    model = load_model()
    if model is None:
        model = train_and_save(df)
    else:
        print(f"Loaded saved model from {MODEL_PATH}")
    interactive_predict(model)

if __name__ == "__main__":
    main()
