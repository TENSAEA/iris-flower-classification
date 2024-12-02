# scripts/model_training.py

import joblib
from sklearn.ensemble import RandomForestClassifier

# Load training data
X_train = joblib.load('model/X_train.pkl')
y_train = joblib.load('model/y_train.pkl')

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model/trained_model.pkl')

print("Model training completed and saved as 'trained_model.pkl'.")