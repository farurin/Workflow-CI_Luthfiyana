import pandas as pd
import os
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Setup Dagshub dan MLflow
DAGSHUB_USER = "farurin"
DAGSHUB_REPO = "Eksperimen_SML_Luthfiyana"
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment("Spam Filter Experiments")

# Load data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "spam_cleaned.zip")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['clean_text', 'target'])

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['target'], test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Hyperparameter Grid
param_grid = {
    'tfidf__max_features': [1000, 3000],
    'nb__alpha': [0.5, 1.0]
}

# Manual Logging Tuning

print("Memulai Hyperparameter Tuning...")
with mlflow.start_run(run_name="Advanced_Tuning_Manual"):
    
    # Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Manual Log Params
    print(f"Best Params: {best_params}")
    mlflow.log_params(best_params)
    
    # Manual Log Metrics
    y_pred = best_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }
    mlflow.log_metrics(metrics)
    
    # c. Manual Log Artifacts
    
    # Artefak 1: Confusion Matrix Image
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Artefak 2: Sample Prediction CSV
    sample_df = pd.DataFrame({'text': X_test[:10], 'actual': y_test[:10], 'pred': y_pred[:10]})
    sample_df.to_csv("sample_predictions.csv", index=False)
    mlflow.log_artifact("sample_predictions.csv")
    
    # Manual Log Model
    mlflow.sklearn.log_model(best_model, "model")
    
    print("Selesai! Cek DagsHub.")
    
    # Bersihkan file temp
    if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")
    if os.path.exists("sample_predictions.csv"): os.remove("sample_predictions.csv")