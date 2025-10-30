import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    print("=== Titanic Logistic Regression (accuracy only) ===")

    # ---------- Load ----------
    print("\n[1] Loading datasets from ./data ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    gender_submission = pd.read_csv("data/gender_submission.csv")
    print(f"train.shape: {train.shape}")
    print(f"test.shape:  {test.shape}")
    print(f"gender_submission.shape: {gender_submission.shape}")

    # ---------- Define X_train, y_train, X_test, y_test ----------
    print("\n[2] Preparing data ...")
    y_train = train["Survived"]
    X_train = train.drop(columns=["Survived", "PassengerId"], errors="ignore")
    pid_test = test["PassengerId"]
    X_test = test.drop(columns=["PassengerId"], errors="ignore")
    y_test = gender_submission.set_index("PassengerId").loc[pid_test.values, "Survived"].values

    # ---------- Encoding ----------
    print("\n[3] Encoding categorical variables ...")

    # Sex
    if "Sex" in X_train.columns:
        print(" - Encoding 'Sex'")
        X_train["Sex"] = X_train["Sex"].map({"male": 0, "female": 1})
        X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})

    # Embarked
    if "Embarked" in X_train.columns:
        print(" - One-hot encoding 'Embarked' (drop_first=True)")
        X_train = pd.get_dummies(X_train, columns=["Embarked"], drop_first=True)
        X_test = pd.get_dummies(X_test, columns=["Embarked"], drop_first=True)

    # Drop text-like columns
    to_drop = [c for c in ["Name", "Ticket", "Cabin"] if c in X_train.columns]
    if to_drop:
        print(" - Dropping columns:", to_drop)
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop, errors="ignore")

    # Align test columns to train columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # ---------- Impute missing numeric values ----------
    print("\n[4] Imputing NAs in numeric columns with train medians ...")
    num_cols = X_train.select_dtypes(include="number").columns
    for c in num_cols:
        median_val = X_train[c].median()
        X_train[c] = X_train[c].fillna(median_val)
        X_test[c] = X_test[c].fillna(median_val)

    # ---------- Scale numeric columns ----------
    print("\n[5] Scaling numeric features using z-score scaling ...")
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    print(" - Scaling complete.")

    # ---------- Train model ----------
    print("\n[6] Training Logistic Regression ...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print(" - Model trained successfully.")

    # ---------- Evaluate ----------
    print("\n[7] Evaluating accuracy ...")
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy (vs gender_submission): {test_acc:.4f}")

    print("\n=== Done. ===")

if __name__ == "__main__":
    main()
