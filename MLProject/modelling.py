import pandas as pd
import numpy as np
import os
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# setup dagshub & mlflow
DAGSHUB_USER = "farurin" # username dagshub
DAGSHUB_REPO = "Eksperimen_SML_Luthfiyana" # nama repo dagshub

print("Menghubungkan ke DagsHub...")
# mlflow=True akan otomatis setting Tracking URI dan Credentials
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)

mlflow.set_experiment("Spam Filter Experiments")

# load dataset hasil preprocessing (zip)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(SCRIPT_DIR, "spam_cleaned.zip")

print(f"Memuat dataset dari {DATA_PATH}...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("File dataset tidak ditemukan! Pastikan path benar.")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['clean_text', 'target'])

X = df['clean_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# definisi pipeline model dan tuning
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

param_grid = {
    'tfidf__max_features': [1000, 3000, 5000],
    'nb__alpha': [0.1, 0.5, 1.0]
}

# Auto logging dengan mlflow
mlflow.autolog()

print("Memulai Hyperparameter Tuning...")
with mlflow.start_run(run_name="Advanced Tuning NB"):
    
    # Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best Params: {best_params}")
    mlflow.log_params(best_params)
    
    # Evaluasi
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print(f"Results -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec
    })
    
    # Log Artifacts
    
    # Artifact 1: Confusion Matrix
    print("Generating Confusion Matrix...")
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    
    # Artifact 2: Sample Data
    print("Saving sample data...")
    sample_path = "sample_test_data.csv"
    X_test.head(10).to_frame().to_csv(sample_path, index=False)
    mlflow.log_artifact(sample_path)
    
    # Log Model Sklearn
    print("Logging Model...")
    mlflow.sklearn.log_model(best_model, "model")
    
    # Simpan Run ID
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    print("Training Selesai! Cek DagsHub.")
    
    # Bersihkan file lokal
    if os.path.exists(cm_path): os.remove(cm_path)
    if os.path.exists(sample_path): os.remove(sample_path)