import pandas as pd
import os
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Setup Dagshub dan MLflow
DAGSHUB_USER = "farurin"
DAGSHUB_REPO = "Eksperimen_SML_Luthfiyana"
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)

mlflow.set_experiment("Spam Filter Experiments")

# Load data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "spam_cleaned.zip")

print(f"Memuat data dari: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['clean_text', 'target'])

X = df['clean_text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisi pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('nb', MultinomialNB())
])

# Autolog
mlflow.autolog() 

print("Memulai Training (Basic Autolog)...")
with mlflow.start_run(run_name="Basic_CI_Run"):
    
    # Fit model (param, metric, dan model dari mlflow.autolog())
    pipeline.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Training Selesai. Accuracy: {acc}")

    # Simpan Run ID
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    
    with open("run_id.txt", "w") as f:
        f.write(run_id)