import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import plotly.graph_objects as go
import time
import tempfile

# Fixed path for pre-trained model
MODEL_PATH = 'model/skin_cancer_model.h5'

# Function to plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    fig = go.Figure()

    # Accuracy plot
    fig.add_trace(go.Scatter(x=list(epochs), y=acc, mode='lines+markers', name='Training accuracy'))
    fig.add_trace(go.Scatter(x=list(epochs), y=val_acc, mode='lines+markers', name='Validation accuracy'))

    fig.update_layout(title='Training and validation accuracy', xaxis_title='Epochs', yaxis_title='Accuracy')

    st.plotly_chart(fig)

    fig = go.Figure()

    # Loss plot
    fig.add_trace(go.Scatter(x=list(epochs), y=loss, mode='lines+markers', name='Training loss'))
    fig.add_trace(go.Scatter(x=list(epochs), y=val_loss, mode='lines+markers', name='Validation loss'))

    fig.update_layout(title='Training and validation loss', xaxis_title='Epochs', yaxis_title='Loss')

    st.plotly_chart(fig)

def app():
    st.title("🛠️ Train Skin Cancer Detection Model")
    st.write("""
            👋 Welcome! Use this page to fine-tune our pre-trained skin cancer detection model.
            
            **What You'll Do:**
            - **Customize:** Adjust epochs and learning rates to fit your needs. 🛠️
            - **Track Progress:** Watch live updates with real-time training visuals. 📈
            - **Evaluate:** Check the model’s performance and download the updated version. 📊
            """)
    with st.expander("See More Details"):
        st.write("""

        ### How It Works:
        1. Set your parameters. ⚙️
        2. Click 'Start Training' and monitor the progress. 🚀
        3. View detailed performance plots. 📉
        4. Download the improved model. 📥

        Let’s boost our model’s accuracy together! 💪
        """)

    # User inputs for model parameters
    epochs_phase1 = st.number_input("Number of epochs for initial training", min_value=1, max_value=100, value=20)
    epochs_phase2 = st.number_input("Number of epochs for fine-tuning", min_value=1, max_value=100, value=10)
    learning_rate_phase1 = st.number_input("Learning rate for initial training", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
    learning_rate_phase2 = st.number_input("Learning rate for fine-tuning", min_value=1e-6, max_value=0.01, value=1e-5, format="%.6f")

    # Button to start training
    if st.button("Start Training"):
        training_message = st.warning("It takes some time to train, please stay on this page.")
        
        # Initialize the progress bar
        progress_bar = st.progress(0)

        # Load the pre-trained model
        model = load_model(MODEL_PATH)

        # Optionally freeze the base layers
        for layer in model.layers[:-4]:
            layer.trainable = False

        # Compile the model for initial training
        model.compile(optimizer=Adam(learning_rate=learning_rate_phase1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # Split data into 80% train and 20% validation
        )

        # Load training data
        train_generator = datagen.flow_from_directory(
            'training_dataset',  # Replace with your dataset path
            target_size=(224, 224),
            batch_size=32,
            class_mode='sparse',
            subset='training'
        )

        # Load validation data
        validation_generator = datagen.flow_from_directory(
            'training_dataset',  # Replace with your dataset path
            target_size=(224, 224),
            batch_size=32,
            class_mode='sparse',
            subset='validation'
        )

        # Train the model (initial training)
        history_phase1 = model.fit(train_generator, validation_data=validation_generator, epochs=epochs_phase1)
        progress_bar.progress(50)
        
        # Notify that initial training is done
        initial_training_success = st.success("Initial training done. Fine-tuning starts now!")

        # Hide the success message after a few seconds
        time.sleep(5)
        initial_training_success.empty()

        # Unfreeze the last few layers of the base model
        for layer in model.layers[-4:]:
            layer.trainable = True

        # Compile the model again for fine-tuning
        model.compile(optimizer=Adam(learning_rate=learning_rate_phase2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Fine-tune the model
        history_phase2 = model.fit(train_generator, validation_data=validation_generator, epochs=epochs_phase2)
        progress_bar.progress(100)

        # Combine history of both training phases
        for key in history_phase2.history.keys():
            history_phase1.history[key].extend(history_phase2.history[key])

        # Evaluate the model
        loss, accuracy = model.evaluate(validation_generator)
        st.write(f'Test Accuracy: {accuracy * 100:.2f}%')

        # Display training results
        st.write("Training complete!")
        plot_training_history(history_phase1)
        
        # Save the re-trained model to temporary files with both .h5 and .keras extensions
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file_h5, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_file_keras:
            # Save the model in both formats
            model.save(tmp_file_h5.name)
            model.save(tmp_file_keras.name, save_format='keras')
            
            # Get file paths
            temp_file_path_h5 = tmp_file_h5.name
            temp_file_path_keras = tmp_file_keras.name
        
        # Remove(hide) training message
        training_message.empty()
        
        # Remove(hide) the progress bar
        progress_bar.empty()
        
        # Display celebration animation
        st.balloons()

        # Provide download links for both models
        with open(temp_file_path_h5, "rb") as file_h5, open(temp_file_path_keras, "rb") as file_keras:
            st.download_button(
                label="Download Trained Model (.h5)",
                data=file_h5,
                file_name="trained_skin_cancer_model.h5",
                mime="application/octet-stream"
            )
            st.download_button(
                label="Download Trained Model (.keras)",
                data=file_keras,
                file_name="trained_skin_cancer_model.keras",
                mime="application/octet-stream"
            )

# To be placed in app.py or a suitable place in the project
if __name__ == "__main__":
    app()
