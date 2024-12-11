# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
from PIL import Image
import scripts.model_evaluation as model_evaluation

# Evaluate the model
output = model_evaluation.evaluate_model()

# Set page configuration for better layout
st.set_page_config(page_title="🌸 Iris Flower Classification", layout="wide")

# Load the scaler and best-trained model
scaler = joblib.load('model/scaler.pkl')
model = joblib.load('model/best_trained_model.pkl')

# Load training history
history = joblib.load('model/training_history.pkl')

# Mapping of encoded species back to original labels
species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

# Title and description
st.title("🌸 Iris Flower Classification")
st.markdown("""
Predict the species of an Iris flower using either feature sliders or a natural language description.
""")

# Sidebar for user inputs
st.sidebar.header("🔧 Input Features")

# Choose input method via radio buttons
input_method = st.sidebar.radio("Select Input Method:", ("Sliders", "Natural Language"), index=0)

# Define input methods
def user_input_features_sliders():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.2)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

def parse_description(text):
    """
    Parses the natural language description to extract numerical features.
    """
    patterns = {
        'sepal_length': r'sepal length of ([\d\.]+) cm',
        'sepal_width': r'sepal width of ([\d\.]+) cm',
        'petal_length': r'petal length of ([\d\.]+) cm',
        'petal_width': r'petal width of ([\d\.]+) cm'
    }
    extracted = {}
    for feature, pattern in patterns.items():
        match = re.search(pattern, text.lower())
        if match:
            extracted[feature] = float(match.group(1))
        else:
            extracted[feature] = None  # Handle missing values if necessary
    return extracted

# Organize layout into two columns
col1, col2 = st.columns(2)

with col1:
    if input_method == "Sliders":
        st.subheader("🔧 User Input Features (Sliders)")
        input_df = user_input_features_sliders()
        st.table(input_df)
        input_scaled = scaler.transform(input_df)
    elif input_method == "Natural Language":
        st.subheader("📝 User Input Features (Natural Language)")
        description = st.text_area(
            "Describe the iris flower (e.g., 'The flower has a sepal length of 5.1 cm, sepal width of 3.5 cm, petal length of 1.4 cm, and petal width of 0.2 cm.')"
        )
        if description:
            parsed_features = parse_description(description)
            if None in parsed_features.values():
                st.error("❌ Please provide all four features in your description.")
                input_scaled = None
            else:
                input_df = pd.DataFrame([parsed_features])
                st.table(input_df)
                input_scaled = scaler.transform(input_df)
        else:
            input_scaled = None

    if input_scaled is not None:
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Display prediction
        st.subheader("🎯 Prediction")
        predicted_species = species_mapping[prediction[0]]
        st.write(f"**{predicted_species}**")

        # Display prediction probability
        st.subheader("📊 Prediction Probability")
        prob_df = pd.DataFrame(prediction_proba, columns=[species_mapping[i] for i in range(3)])
        st.write(prob_df)

with col2:
    if input_scaled is not None:
        # Display corresponding image
        st.subheader("🖼️ Image")
        image_path = f"images/{predicted_species}.jpg"
        image = Image.open(image_path)
        st.image(image, caption=predicted_species, use_container_width=True)

        # Display output in the app
st.subheader("Model Evaluation Results")
st.write(f"Validation Accuracy: {output['Validation Accuracy']:.4f}")
st.write(f"Validation Precision: {output['Validation Precision']:.4f}")
st.write(f"Validation Recall: {output['Validation Recall']:.4f}")
st.write(f"Validation F1 Score: {output['Validation F1 Score']:.4f}")

# Plot learning curves
st.subheader("📈 Learning Curves")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Plot training & validation accuracy values
ax[0].plot(history['accuracy'])
ax[0].plot(history['val_accuracy'])
ax[0].set_title('Model accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
ax[1].plot(history['loss'])
ax[1].plot(history['val_loss'])
ax[1].set_title('Model loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

st.pyplot(fig)

# Conclusion on Model Fit
st.subheader("🔍 Conclusion on Model Fit")
if history['accuracy'][-1] > history['val_accuracy'][-1]:
    st.write("The model may be overfitting. Consider using regularization techniques or gathering more data.")
elif history['accuracy'][-1] < history['val_accuracy'][-1]:
    st.write("The model may be underfitting. Consider increasing model complexity or improving feature engineering.")
else:
    st.write("The model appears to be well-fitted. Continue monitoring performance on unseen data.")

# Footer
st.markdown("---")
st.markdown("**Project by Tensae Aschalew**")
st.markdown("**ID: GSR/3976/17**")