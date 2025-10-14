from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, json, os

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    data = load_iris()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Test accuracy: {acc:.3f}")

    joblib.dump(clf, os.path.join(MODEL_DIR, "iris_rf.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    with open(os.path.join(MODEL_DIR, "target_names.json"), "w") as f:
        json.dump(target_names, f)

if __name__ == "__main__":
    main()
