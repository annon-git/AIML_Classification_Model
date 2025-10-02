import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('final_iris_rf_model.pkl')

# Streamlit app layout
st.title("Iris Flower Classification")

sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 5.0, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 3.0, 1.5)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)

st.write(f'Predicted Iris Species: {prediction[0]}')

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 4, 6, None],
    'min_samples_split': [2, 4, 8]
}

search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)

search.fit(X_train, y_train)

best_model = search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("Best parameters:", search.best_params_)
print("Test Accuracy (tuned):", accuracy_score(y_test, y_pred_best))

from sklearn.model_selection import train_test_split

# Split the data into 70% training and 30% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Print the shape of the splits
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest classifier with a random state for reproducibility
clf = RandomForestClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

print("Model training complete.")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

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

from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 4, 6, None],
    'min_samples_split': [2, 4, 8]
}

# Create GridSearchCV object with Random Forest and 5-fold CV
search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)

# Fit on training data
search.fit(X_train, y_train)

# Best parameters found
print("Best parameters:", search.best_params_)
best_model = search.best_estimator_

# Evaluate best model on test data
y_pred_best = best_model.predict(X_test)
print("Test Accuracy (tuned):", accuracy_score(y_test, y_pred_best))
print("\nClassification Report (tuned):")
print(classification_report(y_test, y_pred_best))

import matplotlib.pyplot as plt
import numpy as np

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

