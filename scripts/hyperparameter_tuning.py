# scripts/hyperparameter_tuning.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load training data
X_train = joblib.load('model/X_train.pkl')
y_train = joblib.load('model/y_train.pkl')

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Define hyperparameters grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

# Perform grid search
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best F1 Score: {grid_search.best_score_:.4f}")

# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'model/best_trained_model.pkl')

print("Hyperparameter tuning completed and best model saved as 'best_trained_model.pkl'.")