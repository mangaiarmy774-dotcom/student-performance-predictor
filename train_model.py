"""Train a Decision Tree to predict Pass / Needs Support (UCI Student Performance - Maths)."""
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
PASS_MARK = 10  # final grade G3 (0-20) >= 10 means "Likely to Pass"

df = pd.read_csv("student_data.csv", sep=";")
print("Rows:", len(df), "| Missing values:", int(df.isna().sum().sum()))

X = df[FEATURES]
y = (df["G3"] >= PASS_MARK).astype(int)  # 1 = Likely to Pass, 0 = Needs Support

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
names = ["Needs Support", "Likely to Pass"]
print(f"\nTest accuracy: {acc:.3f}")
print(classification_report(y_test, pred, target_names=names))

joblib.dump(model, "model.joblib")
report = classification_report(y_test, pred, target_names=names, output_dict=True)
json.dump({"accuracy": acc, "train_rows": len(X_train), "test_rows": len(X_test),
           "report": report, "features": FEATURES}, open("metrics.json", "w"), indent=2)

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=names, cmap="Purples")
plt.title("Confusion Matrix (test set)")
plt.savefig("screenshots/confusion_matrix.png", dpi=150, bbox_inches="tight")

plt.figure(figsize=(18, 8))
plot_tree(model, feature_names=FEATURES, class_names=names, filled=True, rounded=True, fontsize=8)
plt.savefig("screenshots/decision_tree.png", dpi=130, bbox_inches="tight")
print("Saved model.joblib, metrics.json and charts in screenshots/")
