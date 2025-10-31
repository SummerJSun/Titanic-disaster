import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def main():
    print("=== Titanic Logistic Regression ===")

    # ---------- Load ----------
    print("\n[1] Loading datasets from ./data ...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"train.shape: {train.shape}")
    print(f"test.shape:  {test.shape}")

    # ---------- Define X_train, y_train, X_test ----------
    print("\n[2] Preparing data ...")
    y_train = train["Survived"]
    X_train = train.drop(columns=["Survived", "PassengerId"], errors="ignore")
    pid_test = test["PassengerId"]
    X_test = test.drop(columns=["PassengerId"], errors="ignore")

    # ---------- Encoding ----------
    print("\n[3] Encoding categorical variables ...")

    # Sex → {male:0, female:1}
    if "Sex" in X_train.columns:
        print(" - Encoding 'Sex'")
        X_train["Sex"] = X_train["Sex"].map({"male": 0, "female": 1})
        X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})

    # Embarked → one-hot (drop_first=True)
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
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

    # ---------- Scale numeric columns ----------
    print("\n[5] Scaling numeric features using z-score scaling ...")
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # ---------- Train model ----------
    print("\n[6] Training Logistic Regression ...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print(" - Model trained successfully.")

    # ---------- Training accuracy ----------
    print("\n[7] Evaluating training accuracy ...")
    train_pred = model.predict(X_train)
    train_acc = (train_pred == y_train).mean()
    print(f"Training Accuracy: {train_acc:.4f}")

    # ---------- Predict & Save ----------
    print("\n[8] Generating predictions on test set ...")
    test_pred = model.predict(X_test).astype(int)
    out = pd.DataFrame({"PassengerId": pid_test, "Survived": test_pred})
    out_path = "data/predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    print("\n=== Done. ===")

if __name__ == "__main__":
    main()

