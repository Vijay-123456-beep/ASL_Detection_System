import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pickle.load(open("data.pickle", "rb"))
X = np.asarray(dataset["data"])
y = np.asarray(dataset["labels"])

print("Samples:", len(X))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42
)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train SVM Classifier
model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
print("Training SVM model...")
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.2f}%")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model + scaler
with open("model.p", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print("\nModel saved → model.p")