import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from dagshub import dagshub_logger
import dagshub
from imblearn.over_sampling import SMOTE
import joblib

dagshub.init(repo_owner='KrisnaSantosa15',
             repo_name='stroke_prediction', mlflow=True)


# MLflow Tracking lokal
mlflow.set_tracking_uri("http://localhost:5000")

data = pd.read_csv("stroke_data_clean.csv")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("stroke", axis=1),
    data["stroke"],
    test_size=0.2,
    random_state=42,
    stratify=data["stroke"]
)


# Autolog
mlflow.autolog()

input_example = X_train[0:5]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"Random_Forest_{timestamp}"

with mlflow.start_run(run_name=run_name):
    n_estimators = 100
    random_state = 42

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    joblib.dump(model, "model.pkl")

print("\nTraining selesai. Metric accuracy:", accuracy)
