"""
Pneumonia Prediction Script
============================
This script loads a pre-trained pneumonia classification model and makes predictions
on chest X-ray images. It can process single images or batch process multiple images.

Author: Your Name
Date: February 25, 2026
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ========================================
# CONFIGURATION
# ========================================

# Path to your trained model - update this to match your model file
MODEL_PATH = 'best_pneumonia_model.keras'

# Input image size - must match the size the model was trained on
# ResNet50 uses 224x224 by default
IMG_SIZE = (224, 224)

# Class names - these should match your training data structure
# The order should match the class indices from training
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


# ========================================
# LOAD THE TRAINED MODEL
# ========================================

def load_model(model_path):
    """
    Load a pre-trained Keras model from disk.
    
    Args:
        model_path (str): Path to the saved model file (.keras or .h5)
    
    Returns:
        model: Loaded Keras model ready for predictions
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    try:
        # Load the model using Keras's load_model function
        # This works for both .keras and .h5 formats
        model = keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
        
        # Display model summary to verify it loaded correctly
        print("\nModel Architecture:")
        model.summary()
        
        return model
    
    except FileNotFoundError:
        print(f"✗ Error: Model file not found at {model_path}")
        print("Please ensure the model file exists and the path is correct.")
        sys.exit(1)
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)


# ========================================
# IMAGE PREPROCESSING
# ========================================

def preprocess_image(image_path, target_size=IMG_SIZE):
    """
    Load and preprocess a single image for prediction.
    
    The preprocessing steps must match exactly what was done during training:
    1. Load the image file
    2. Resize to target dimensions
    3. Convert to array format
    4. Normalize pixel values (0-1 range)
    5. Add batch dimension
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target image dimensions (height, width)
    
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
        PIL.Image: Original image for display purposes
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
    """
    try:
        # Load the image using Keras utilities
        # This handles various image formats (JPEG, PNG, etc.)
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=target_size  # Resize to model's expected input size
        )
        
        # Convert PIL Image to numpy array
        # Shape will be (height, width, channels) = (224, 224, 3)
        img_array = keras.preprocessing.image.img_to_array(img)
        
        # Normalize pixel values from 0-255 to 0-1 range
        # This matches the rescaling done during training with ImageDataGenerator
        img_array = img_array / 255.0
        
        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        # Models expect input shape: (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"✓ Image loaded and preprocessed: {image_path}")
        print(f"  Original image size: {img.size}")
        print(f"  Preprocessed array shape: {img_array.shape}")
        
        return img_array, img
    
    except FileNotFoundError:
        print(f"✗ Error: Image file not found at {image_path}")
        sys.exit(1)
    
    except Exception as e:
        print(f"✗ Error processing image: {e}")
        sys.exit(1)


# ========================================
# MAKE PREDICTIONS
# ========================================

def predict_single_image(model, image_path, class_names=CLASS_NAMES):
    """
    Predict pneumonia classification for a single chest X-ray image.
    
    Args:
        model: Trained Keras model
        image_path (str): Path to the chest X-ray image
        class_names (list): List of class names ['NORMAL', 'PNEUMONIA']
    
    Returns:
        dict: Dictionary containing prediction results with keys:
            - 'predicted_class': Name of predicted class
            - 'confidence': Confidence percentage for predicted class
            - 'probabilities': Dictionary of all class probabilities
    """
    print(f"\n{'='*60}")
    print(f"Making prediction for: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Preprocess the image
    img_array, original_img = preprocess_image(image_path)
    
    # Make prediction
    # model.predict() returns probability for each class
    # Shape: (1, num_classes) = (1, 2)
    predictions = model.predict(img_array, verbose=0)
    
    # Get the predicted class index (0 or 1)
    # np.argmax finds the class with highest probability
    predicted_class_idx = np.argmax(predictions[0])
    
    # Get the predicted class name
    predicted_class = class_names[predicted_class_idx]
    
    # Get confidence (probability) for the predicted class
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Create a dictionary of all class probabilities
    probabilities = {
        class_names[i]: predictions[0][i] * 100 
        for i in range(len(class_names))
    }
    
    # Print results
    print(f"\n{'─'*60}")
    print(f"PREDICTION RESULTS:")
    print(f"{'─'*60}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nClass Probabilities:")
    for class_name, prob in probabilities.items():
        print(f"  {class_name:12s}: {prob:6.2f}%")
    print(f"{'─'*60}")
    
    # Prepare results dictionary
    results = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities,
        'image': original_img
    }
    
    return results


def predict_batch(model, image_paths, class_names=CLASS_NAMES):
    """
    Predict pneumonia classification for multiple chest X-ray images.
    
    Args:
        model: Trained Keras model
        image_paths (list): List of paths to chest X-ray images
        class_names (list): List of class names
    
    Returns:
        list: List of prediction results dictionaries
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"BATCH PREDICTION: Processing {len(image_paths)} images")
    print(f"{'='*60}")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {Path(image_path).name}")
        result = predict_single_image(model, image_path, class_names)
        results.append(result)
    
    return results


# ========================================
# VISUALIZATION
# ========================================

def visualize_prediction(image_path, prediction_result, save_path=None):
    """
    Visualize the prediction result with the original image.
    
    Args:
        image_path (str): Path to the original image
        prediction_result (dict): Dictionary containing prediction results
        save_path (str, optional): Path to save the visualization
    """
    # Create figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Display the chest X-ray image
    img = plt.imread(image_path)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    ax1.set_title(f"Chest X-Ray: {Path(image_path).name}", fontsize=12, pad=10)
    
    # Right plot: Bar chart of class probabilities
    classes = list(prediction_result['probabilities'].keys())
    probabilities = list(prediction_result['probabilities'].values())
    
    # Create bar chart
    colors = ['green' if prob == max(probabilities) else 'steelblue' 
              for prob in probabilities]
    bars = ax2.bar(classes, probabilities, color=colors, alpha=0.7)
    
    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Probability (%)', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=12, pad=10)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add overall prediction as main title
    prediction_text = f"Prediction: {prediction_result['predicted_class']} " \
                     f"(Confidence: {prediction_result['confidence']:.2f}%)"
    fig.suptitle(prediction_text, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")
    else:
        plt.savefig('prediction_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: prediction_visualization.png")
    
    plt.close()


# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """
    Main function to demonstrate model usage.
    Modify this section to predict on your own images.
    """
    print("\n" + "="*60)
    print("PNEUMONIA CLASSIFICATION FROM CHEST X-RAY")
    print("="*60)
    
    # Step 1: Load the trained model
    model = load_model(MODEL_PATH)
    
    # Step 2: Define image path(s) to predict
    # OPTION A: Single image prediction
    image_path = './test/PNEUMONIA/person1651_virus_2855.jpeg'
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"\n✗ Warning: Test image not found at {image_path}")
        print("Please update IMAGE_PATH with a valid chest X-ray image path.")
        
        # Try to find any image in test directories
        test_dirs = [
            './test/NORMAL',
            './test/PNEUMONIA'
        ]
        
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                images = list(Path(test_dir).glob('*.jpeg')) + \
                        list(Path(test_dir).glob('*.jpg')) + \
                        list(Path(test_dir).glob('*.png'))
                if images:
                    image_path = str(images[0])
                    print(f"✓ Using alternative test image: {image_path}")
                    break
        else:
            print("\n✗ No test images found. Please specify a valid image path.")
            return
    
    # Step 3: Make prediction
    result = predict_single_image(model, image_path)
    
    # Step 4: Visualize the result
    visualize_prediction(image_path, result)
    
    
    # OPTION B: Batch prediction (uncomment to use)
    # image_paths = [
    #     '/path/to/image1.jpeg',
    #     '/path/to/image2.jpeg',
    #     '/path/to/image3.jpeg'
    # ]
    # results = predict_batch(model, image_paths)
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)


# ========================================
# USAGE EXAMPLES
# ========================================

def example_usage():
    """
    Examples of different ways to use this script.
    """
    # Example 1: Predict on a single image
    model = load_model('best_pneumonia_model.keras')
    result = predict_single_image(model, 'path/to/xray.jpeg')
    
    # Example 2: Predict on multiple images
    image_list = ['image1.jpeg', 'image2.jpeg', 'image3.jpeg']
    results = predict_batch(model, image_list)
    
    # Example 3: Predict and visualize
    result = predict_single_image(model, 'xray.jpeg')
    visualize_prediction('xray.jpeg', result, save_path='my_prediction.png')


if __name__ == "__main__":
    # Run the main prediction function
    main()
    
    # Uncomment below to see usage examples
    # example_usage()
