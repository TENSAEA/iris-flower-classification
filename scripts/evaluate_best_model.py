# scripts/evaluate_best_model.py

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load validation data and the best model
X_val = joblib.load('model/X_val.pkl')
y_val = joblib.load('model/y_val.pkl')
best_model = joblib.load('model/best_trained_model.pkl')

# Make predictions
y_pred_best = best_model.predict(X_val)

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred_best)
precision = precision_score(y_val, y_pred_best, average='weighted')
recall = recall_score(y_val, y_pred_best, average='weighted')
f1 = f1_score(y_val, y_pred_best, average='weighted')

print(f"Best Model Validation Accuracy: {accuracy:.4f}")
print(f"Best Model Validation Precision: {precision:.4f}")
print(f"Best Model Validation Recall: {recall:.4f}")
print(f"Best Model Validation F1 Score: {f1:.4f}")