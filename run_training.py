import os
import argparse
import subprocess
import time
import signal
import sys

def signal_handler(sig, frame):
    print("\nTraining interrupted. You can resume later using the --resume flag.")
    sys.exit(0)

def main():
    # Register signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='Run X-ray Anomaly Detection Model Training')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Directory containing the prepared dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='model/checkpoint.h5', help='Path to save model checkpoint')
    parser.add_argument('--final_model_path', type=str, default='model/xray_anomaly_detector.h5', help='Path to save final model')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' does not exist.")
        print("Please run prepare_dataset.py first to prepare the dataset.")
        return
    
    # Check if train, val, and test directories exist
    train_dir = os.path.join(args.dataset_dir, 'train')
    val_dir = os.path.join(args.dataset_dir, 'val')
    test_dir = os.path.join(args.dataset_dir, 'test')
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        print("Error: Dataset is not properly structured.")
        print("Please run prepare_dataset.py first to prepare the dataset.")
        return
    
    # Check if model directory exists, create if not
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    # Build the command to run train_model.py
    cmd = [
        'python', 'train_model.py',
        '--train_dir', train_dir,
        '--val_dir', val_dir,
        '--test_dir', test_dir,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--checkpoint_path', args.checkpoint_path,
        '--final_model_path', args.final_model_path
    ]
    
    # Add resume flag if requested
    if args.resume:
        cmd.append('--resume')
    
    # Print the command
    print("Running command:")
    print(' '.join(cmd))
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
    except KeyboardInterrupt:
        print("\nTraining interrupted. You can resume later using the --resume flag.")

if __name__ == "__main__":
    main() 