# X-Ray Anomaly Detector - Training Guide

This guide explains how to train the X-Ray Anomaly Detection model with improved accuracy (targeting 95%) and the ability to resume training from checkpoints.

## Prerequisites

- Python 3.8+
- TensorFlow 2.8+
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- tqdm

Install the required packages:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn tqdm
```

## Dataset Preparation

Before training, you need to prepare your dataset. The dataset should be organized as follows:

```
dataset/
├── train/
│   ├── normal/
│   │   └── (normal X-ray images)
│   └── abnormal/
│       └── (abnormal X-ray images)
├── val/
│   ├── normal/
│   │   └── (normal X-ray images)
│   └── abnormal/
│       └── (abnormal X-ray images)
└── test/
    ├── normal/
    │   └── (normal X-ray images)
    └── abnormal/
        └── (abnormal X-ray images)
```

You can use the `prepare_dataset.py` script to organize your dataset:

```bash
python prepare_dataset.py --normal_dir /path/to/normal/images --abnormal_dir /path/to/abnormal/images --output_dir dataset
```

Optional arguments:
- `--augment`: Augment the dataset to increase the number of training samples
- `--augmentation_factor`: Number of augmented images to generate per original image (default: 2)
- `--train_ratio`: Ratio of images to use for training (default: 0.7)
- `--val_ratio`: Ratio of images to use for validation (default: 0.15)
- `--test_ratio`: Ratio of images to use for testing (default: 0.15)
- `--seed`: Random seed for reproducibility (default: 42)

## Training the Model

To train the model, use the `run_training.py` script:

```bash
python run_training.py --dataset_dir dataset --epochs 100 --batch_size 32
```

This script will:
1. Check if the dataset is properly structured
2. Run the training process using `train_model.py`
3. Save checkpoints during training
4. Evaluate the model on the test set
5. Save the final model

### Resuming Training

If training is interrupted, you can resume from the last checkpoint:

```bash
python run_training.py --dataset_dir dataset --epochs 100 --batch_size 32 --resume
```

The script will automatically detect the last checkpoint and continue training from there.

## Model Architecture

The model uses a deep convolutional neural network (CNN) with the following features:

- Multiple convolutional blocks with batch normalization and dropout
- Data augmentation to improve generalization
- Early stopping to prevent overfitting
- Learning rate reduction when validation loss plateaus
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for better image contrast

## Achieving 95% Accuracy

To achieve 95% accuracy, the training process includes:

1. **Data Augmentation**: The `prepare_dataset.py` script can augment the training data to increase the number of samples.

2. **Advanced Preprocessing**: Images are preprocessed using CLAHE to enhance contrast and improve feature visibility.

3. **Deep Architecture**: The model uses a deep CNN with multiple convolutional blocks and dense layers.

4. **Regularization**: Dropout and batch normalization are used to prevent overfitting.

5. **Early Stopping**: Training stops when validation accuracy stops improving, preventing overfitting.

6. **Learning Rate Scheduling**: The learning rate is reduced when validation loss plateaus, allowing fine-tuning.

## Monitoring Training

During training, the following information is displayed:

- Training and validation accuracy/loss for each epoch
- Best validation accuracy achieved
- Confusion matrix and classification report on the test set
- Training history plot saved as `model/training_history.png`

## Troubleshooting

If you encounter issues:

1. **Out of Memory**: Reduce the batch size using `--batch_size 16` or lower.

2. **Low Accuracy**: 
   - Increase the number of epochs
   - Use data augmentation with `--augment` and a higher `--augmentation_factor`
   - Collect more training data

3. **Overfitting**:
   - Increase dropout rates in the model
   - Use more data augmentation
   - Reduce model complexity

4. **Training Interruption**:
   - Use the `--resume` flag to continue from the last checkpoint
   - Checkpoints are saved automatically during training 