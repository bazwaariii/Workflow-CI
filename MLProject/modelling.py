import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Solana_Basic_Muhammad_Bazwa_Arigusna")
    mlflow.autolog()
    
    X_train = pd.read_csv("solana_forensic_preprocessing/X_train.csv")
    y_train = pd.read_csv("solana_forensic_preprocessing/y_train.csv")
    
    with mlflow.start_run():
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train.values.ravel())
        print("Training selesai!")

if __name__ == "__main__":
    main()
