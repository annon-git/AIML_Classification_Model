from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Print first 5 rows of features
print("Features:")
print(X.head())

# Print count of each species class
print("\nSpecies counts:")
print(y.value_counts())

# Split the data: 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shape of resulting splits
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Create and train the classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# More detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display the confusion matrix as a plot
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 4, 6, None],
    'min_samples_split': [2, 4, 8]
}
search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
best_model = search.best_estimator_

# Evaluate the best model on the test set
y_pred_best = best_model.predict(X_test)
print("Test Accuracy (tuned):", accuracy_score(y_test, y_pred_best))
print("\nClassification Report (tuned):")
print(classification_report(y_test, y_pred_best))

# Feature importances from the best model (after tuning)
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(8, 5))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=45)
plt.tight_layout()
plt.show()

# Save the trained model to a file
joblib.dump(best_model, "final_iris_rf_model.pkl")
print("Trained model saved as final_iris_rf_model.pkl")

# Cross-validation scores for the best model
scores = cross_val_score(best_model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())

# Compare other classifiers
models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, pred)}")

# Pipeline with StandardScaler and SVM
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

pipe.fit(X_train, y_train)
print("Pipeline accuracy:", pipe.score(X_test, y_test))
