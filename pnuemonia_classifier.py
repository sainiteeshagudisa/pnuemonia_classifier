import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ========================================
# CONFIGURATION
# ========================================
# Update these paths to match your directory structure
TRAIN_DIR = os.getenv('TRAIN_DIR', './train')
VAL_DIR = os.getenv('VAL_DIR', './val')
TEST_IMAGE_PATH = os.getenv('TEST_IMAGE_PATH', './test/NORMAL/NORMAL2-IM-0337-0001.jpeg')

IMG_SIZE = (224, 224)  # ResNet50 standard input size
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2

# ========================================
# DATA PREPARATION
# ========================================
# Data augmentation for training - helps prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to 0-1
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of training data for validation
)

# Only rescale for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ========================================
# MODEL BUILDING WITH RESNET50
# ========================================
def create_model():
    """
    Creates a transfer learning model using ResNet50 as base.
    ResNet50 is pre-trained on ImageNet for better feature extraction.
    """
    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,  # Exclude final classification layer
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers to retain pre-trained weights
    base_model.trainable = False
    
    # Build custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Reduce dimensions
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer
    ])
    
    return model

# Create the model
model = create_model()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# ========================================
# TRAINING
# ========================================
print("\n========== Starting Training ==========\n")

# Define callbacks for better training
callbacks = [
    # Save best model based on validation accuracy
    keras.callbacks.ModelCheckpoint(
        'best_pneumonia_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Reduce learning rate when validation loss plateaus
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ========================================
# VISUALIZE TRAINING RESULTS
# ========================================
def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

plot_training_history(history)

# ========================================
# PREDICTION ON SAMPLE IMAGE
# ========================================
def predict_image(image_path, model):
    """
    Predict whether an X-ray image shows pneumonia or is normal.
    
    Args:
        image_path: Path to the image file
        model: Trained Keras model
    
    Returns:
        Prediction result with confidence
    """
    # Load and preprocess the image
    img = keras.preprocessing.image.load_img(
        image_path,
        target_size=IMG_SIZE
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Get class names from generator
    class_names = list(train_generator.class_indices.keys())
    result = class_names[predicted_class]
    
    return result, confidence, predictions[0]

# Test on sample image
print("\n========== Testing on Sample Image ==========\n")
try:
    result, confidence, probabilities = predict_image(TEST_IMAGE_PATH, model)
    
    print(f"Prediction: {result.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nClass Probabilities:")
    for class_name, prob in zip(train_generator.class_indices.keys(), probabilities):
        print(f"  {class_name}: {prob*100:.2f}%")
    
    # Display the test image with prediction
    img = plt.imread(TEST_IMAGE_PATH)
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {result.upper()} ({confidence:.2f}% confidence)")
    plt.axis('off')
    plt.savefig('prediction_result.png')
    print("\nPrediction visualization saved as 'prediction_result.png'")
    
except Exception as e:
    print(f"Error predicting image: {e}")
    print("Please ensure TEST_IMAGE_PATH is set correctly")

# ========================================
# SAVE FINAL MODEL
# ========================================
model.save('pneumonia_classifier_final.keras')
print("\n========== Model saved as 'pneumonia_classifier_final.keras' ==========")
