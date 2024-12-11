# scripts/model_evaluation.py

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model():
    # Load validation data and the trained model
    X_val = joblib.load('model/X_val.pkl')
    y_val = joblib.load('model/y_val.pkl')
    model = joblib.load('model/trained_model.pkl')

    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    # Return output as a dictionary
    output = {
        'Validation Accuracy': accuracy,
        'Validation Precision': precision,
        'Validation Recall': recall,
        'Validation F1 Score': f1
    }

    return output