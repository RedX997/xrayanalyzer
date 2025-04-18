import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import json
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model(input_shape=(224, 224, 1)):
    """
    Create a CNN model for X-ray anomaly detection
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (normal vs. abnormal)
    ])
    
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for model input
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    
    return img

def create_data_generators(train_dir, val_dir, batch_size=32):
    """
    Create data generators for training and validation
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2,
        shear_range=0.2
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale'
    )
    
    return train_generator, val_generator

def train_model(model, train_generator, val_generator, epochs=50, batch_size=32, 
                checkpoint_path='model/checkpoint.h5', history_path='model/training_history.json',
                initial_epoch=0):
    """
    Train the model with checkpointing and early stopping
    """
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Load previous history if resuming training
    history = {}
    if os.path.exists(history_path) and initial_epoch > 0:
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    # Train the model
    new_history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        initial_epoch=initial_epoch
    )
    
    # Combine histories if resuming training
    if history:
        for key in new_history.history:
            if key in history:
                history[key].extend(new_history.history[key])
            else:
                history[key] = new_history.history[key]
    else:
        history = new_history.history
    
    # Save training history
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    return model, history

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data
    """
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int)
    y_true = test_generator.classes
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'])
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return report, conf_matrix

def plot_training_history(history, save_path='model/training_history.png'):
    """
    Plot training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train X-ray Anomaly Detection Model')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, required=True, help='Directory containing validation data')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='model/checkpoint.h5', help='Path to save model checkpoint')
    parser.add_argument('--final_model_path', type=str, default='model/xray_anomaly_detector.h5', help='Path to save final model')
    
    args = parser.parse_args()
    
    # Create data generators
    train_generator, val_generator = create_data_generators(args.train_dir, args.val_dir, args.batch_size)
    test_generator, _ = create_data_generators(args.test_dir, args.test_dir, args.batch_size)
    
    # Initialize or load model
    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"Resuming training from checkpoint: {args.checkpoint_path}")
        model = load_model(args.checkpoint_path)
        
        # Load history to determine initial epoch
        history_path = 'model/training_history.json'
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                initial_epoch = len(history['accuracy'])
                print(f"Resuming from epoch {initial_epoch}")
        else:
            initial_epoch = 0
    else:
        print("Creating new model")
        model = create_model()
        initial_epoch = 0
    
    # Train model
    model, history = train_model(
        model, 
        train_generator, 
        val_generator, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        initial_epoch=initial_epoch
    )
    
    # Evaluate model
    report, conf_matrix = evaluate_model(model, test_generator)
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    model.save(args.final_model_path)
    print(f"Model saved to {args.final_model_path}")
    
    # Save evaluation results
    with open('model/evaluation_results.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 