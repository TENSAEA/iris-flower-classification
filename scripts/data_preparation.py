# scripts/data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('data/iris.csv', header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Handle missing values (if any)
data = data.dropna()

# Encode categorical variables
data['species'] = data['species'].astype('category').cat.codes

# Features and target
X = data.drop('species', axis=1)
y = data['species']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, 'model/scaler.pkl')

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Save the splits
joblib.dump(X_train, 'model/X_train.pkl')
joblib.dump(X_val, 'model/X_val.pkl')
joblib.dump(X_test, 'model/X_test.pkl')
joblib.dump(y_train, 'model/y_train.pkl')
joblib.dump(y_val, 'model/y_val.pkl')
joblib.dump(y_test, 'model/y_test.pkl')

print("Data preparation completed and files saved in the 'model/' directory.")