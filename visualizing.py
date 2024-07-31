import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
import json

def load_training_history(file_path):
    with open(file_path, 'r') as file:
        history = json.load(file)
    return history

def load_test_accuracy(file_path):
    with open(file_path, 'r') as file:
        test_accuracy = json.load(file)
    return test_accuracy

def plot_training_accuracy(history):
    epochs = range(1, len(history['accuracy']) + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(epochs), y=history['accuracy'], mode='lines+markers', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=list(epochs), y=history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
    fig.update_layout(title='Training and Validation Accuracy', xaxis_title='Epochs', yaxis_title='Accuracy')

    st.plotly_chart(fig)

def plot_training_loss(history):
    epochs = range(1, len(history['loss']) + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(epochs), y=history['loss'], mode='lines+markers', name='Training Loss'))
    fig.add_trace(go.Scatter(x=list(epochs), y=history['val_loss'], mode='lines+markers', name='Validation Loss'))
    fig.update_layout(title='Training and Validation Loss', xaxis_title='Epochs', yaxis_title='Loss')

    st.plotly_chart(fig)

def plot_confusion_matrix(cm, classes):
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='True')

    st.plotly_chart(fig)

def app():
    st.title("📊 Visualize and Analyze Model Performance")
    st.markdown("""
    **🔍 Dive into the details of our model's performance:**

    - **Test Accuracy:** Check the final accuracy on the test dataset. 🎯
    - **Training and Validation Accuracy:** See how your model improves with each epoch. 📈
    - **Loss Metrics:** Track the progress of loss reduction throughout training. 📉
    - **Confusion Matrix:** Understand your model's classification results with a visual confusion matrix. 🧩
    """)
    
    with st.expander("See More Details"):
        st.markdown("""
                    
                    ### How It Helps:
                    
                    - **Gain Insights:** Identify trends and performance issues to optimize your model. 🌟
                    - **Analyze Trends:** Detect patterns that can guide your model improvement efforts. 🔍
                    - **Enhance Performance:** Use the insights to fine-tune and enhance your model’s accuracy. ⚙️
                    - **Interact with Visuals:** Check out the interactive plots and matrices below to get a comprehensive view of our model's performance. 📊
            
                    """)

    # Load the model
    model = load_model('model/skin_cancer_model.h5')

    # Load training history from JSON file
    history = load_training_history('json_files/training_history.json')

    # Load precomputed test accuracy
    test_accuracy = load_test_accuracy('json_files/test_accuracy.json')

    # Display test accuracy in decimal and percentage formats
    test_accuracy_percentage = test_accuracy * 100
    st.markdown(f"#### Test Accuracy: {test_accuracy:.4f} ({test_accuracy_percentage:.2f}%)")

    st.markdown("#### Training and Validation Accuracy")
    plot_training_accuracy(history)

    st.markdown("#### Training and Validation Loss")
    plot_training_loss(history)

    # Predefined confusion matrix
    cm = np.array([[23, 1], [0, 16]])  # Example confusion matrix
    classes = ['Benign', 'Malignant']

    st.markdown("#### Confusion Matrix")
    plot_confusion_matrix(cm, classes)

if __name__ == "__main__":
    app()
