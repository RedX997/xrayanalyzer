import os
import shutil
import random
import argparse
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from tqdm import tqdm

def create_directory_structure(base_dir):
    """
    Create the directory structure for the dataset
    """
    # Create main directories
    os.makedirs(os.path.join(base_dir, 'train', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train', 'abnormal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', 'abnormal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test', 'abnormal'), exist_ok=True)
    
    return {
        'train': {
            'normal': os.path.join(base_dir, 'train', 'normal'),
            'abnormal': os.path.join(base_dir, 'train', 'abnormal')
        },
        'val': {
            'normal': os.path.join(base_dir, 'val', 'normal'),
            'abnormal': os.path.join(base_dir, 'val', 'abnormal')
        },
        'test': {
            'normal': os.path.join(base_dir, 'test', 'normal'),
            'abnormal': os.path.join(base_dir, 'test', 'abnormal')
        }
    }

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for the dataset
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image at {image_path}")
        return None
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    return img

def split_dataset(normal_dir, abnormal_dir, output_dirs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the dataset into train, validation, and test sets
    """
    # Get all image files
    normal_images = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    abnormal_images = [f for f in os.listdir(abnormal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split normal images
    normal_train, normal_temp = train_test_split(normal_images, train_size=train_ratio, random_state=seed)
    normal_val, normal_test = train_test_split(normal_temp, train_size=val_ratio/(val_ratio+test_ratio), random_state=seed)
    
    # Split abnormal images
    abnormal_train, abnormal_temp = train_test_split(abnormal_images, train_size=train_ratio, random_state=seed)
    abnormal_val, abnormal_test = train_test_split(abnormal_temp, train_size=val_ratio/(val_ratio+test_ratio), random_state=seed)
    
    # Copy files to respective directories
    print("Copying normal images...")
    for img in tqdm(normal_train):
        src = os.path.join(normal_dir, img)
        dst = os.path.join(output_dirs['train']['normal'], img)
        shutil.copy(src, dst)
    
    for img in tqdm(normal_val):
        src = os.path.join(normal_dir, img)
        dst = os.path.join(output_dirs['val']['normal'], img)
        shutil.copy(src, dst)
    
    for img in tqdm(normal_test):
        src = os.path.join(normal_dir, img)
        dst = os.path.join(output_dirs['test']['normal'], img)
        shutil.copy(src, dst)
    
    print("Copying abnormal images...")
    for img in tqdm(abnormal_train):
        src = os.path.join(abnormal_dir, img)
        dst = os.path.join(output_dirs['train']['abnormal'], img)
        shutil.copy(src, dst)
    
    for img in tqdm(abnormal_val):
        src = os.path.join(abnormal_dir, img)
        dst = os.path.join(output_dirs['val']['abnormal'], img)
        shutil.copy(src, dst)
    
    for img in tqdm(abnormal_test):
        src = os.path.join(abnormal_dir, img)
        dst = os.path.join(output_dirs['test']['abnormal'], img)
        shutil.copy(src, dst)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training: {len(normal_train)} normal, {len(abnormal_train)} abnormal")
    print(f"Validation: {len(normal_val)} normal, {len(abnormal_val)} abnormal")
    print(f"Testing: {len(normal_test)} normal, {len(abnormal_test)} abnormal")

def augment_dataset(dataset_dir, augmentation_factor=2):
    """
    Augment the dataset by applying transformations
    """
    for class_name in ['normal', 'abnormal']:
        class_dir = os.path.join(dataset_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Augmenting {class_name} images...")
        for img_file in tqdm(image_files):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Apply augmentations
            for i in range(augmentation_factor):
                # Random rotation
                angle = random.uniform(-20, 20)
                rotated = rotate_image(img, angle)
                
                # Random brightness adjustment
                brightness = random.uniform(0.8, 1.2)
                brightened = adjust_brightness(rotated, brightness)
                
                # Random contrast adjustment
                contrast = random.uniform(0.8, 1.2)
                contrasted = adjust_contrast(brightened, contrast)
                
                # Save augmented image
                aug_filename = f"{os.path.splitext(img_file)[0]}_aug_{i}.jpg"
                aug_path = os.path.join(class_dir, aug_filename)
                cv2.imwrite(aug_path, contrasted)
    
    print("Dataset augmentation completed!")

def rotate_image(image, angle):
    """
    Rotate an image by a given angle
    """
    height, width = image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return rotated

def adjust_brightness(image, factor):
    """
    Adjust the brightness of an image
    """
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_contrast(image, factor):
    """
    Adjust the contrast of an image
    """
    return cv2.convertScaleAbs(image, alpha=factor, beta=128*(1-factor))

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for X-ray Anomaly Detection')
    parser.add_argument('--normal_dir', type=str, required=True, help='Directory containing normal X-ray images')
    parser.add_argument('--abnormal_dir', type=str, required=True, help='Directory containing abnormal X-ray images')
    parser.add_argument('--output_dir', type=str, default='dataset', help='Output directory for the prepared dataset')
    parser.add_argument('--augment', action='store_true', help='Augment the dataset')
    parser.add_argument('--augmentation_factor', type=int, default=2, help='Number of augmented images to generate per original image')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of images to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of images to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of images to use for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create directory structure
    print("Creating directory structure...")
    output_dirs = create_directory_structure(args.output_dir)
    
    # Split dataset
    print("Splitting dataset...")
    split_dataset(
        args.normal_dir, 
        args.abnormal_dir, 
        output_dirs, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio, 
        args.seed
    )
    
    # Augment dataset if requested
    if args.augment:
        print("Augmenting dataset...")
        augment_dataset(os.path.join(args.output_dir, 'train'), args.augmentation_factor)
    
    print(f"Dataset preparation completed! Dataset saved to {args.output_dir}")

if __name__ == "__main__":
    main() 